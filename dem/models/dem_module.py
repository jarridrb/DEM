import time
from typing import Any, Dict, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from hydra.utils import get_original_cwd
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchmetrics import MeanMetric

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.data_utils import remove_mean
from dem.utils.logging_utils import fig_to_image

from .components.clipper import Clipper
from .components.cnf import CNF
from .components.distribution_distances import compute_distribution_distances
from .components.ema import EMAWrapper
from .components.lambda_weighter import BaseLambdaWeighter
from .components.mlp import TimeConder
from .components.noise_schedules import BaseNoiseSchedule
from .components.prioritised_replay_buffer import PrioritisedReplayBuffer
from .components.scaling_wrapper import ScalingWrapper
from .components.score_estimator import estimate_grad_Rt, wrap_for_richardsons
from .components.score_scaler import BaseScoreScaler
from .components.sde_integration import integrate_sde
from .components.sdes import VEReverseSDE


def t_stratified_loss(batch_t, batch_loss, num_bins=5, loss_name=None):
    """Stratify loss by binning t."""
    flat_losses = batch_loss.flatten().detach().cpu().numpy()
    flat_t = batch_t.flatten().detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = "loss"
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin + 1]
        t_range = f"{loss_name} t=[{bin_start:.2f},{bin_end:.2f})"
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


def get_wandb_logger(loggers):
    """Gets the wandb logger if it is the list of loggers otherwise returns None."""
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break

    return wandb_logger


class DEMLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        noise_schedule: BaseNoiseSchedule,
        lambda_weighter: BaseLambdaWeighter,
        buffer: PrioritisedReplayBuffer,
        num_init_samples: int,
        num_estimator_mc_samples: int,
        num_samples_to_generate_per_epoch: int,
        num_samples_to_sample_from_buffer: int,
        num_samples_to_save: int,
        eval_batch_size: int,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        nll_with_cfm: bool,
        nll_with_dem: bool,
        nll_on_buffer: bool,
        logz_with_cfm: bool,
        cfm_sigma: float,
        cfm_prior_std: float,
        use_otcfm: bool,
        nll_integration_method: str,
        use_richardsons: bool,
        compile: bool,
        prioritize_cfm_training_samples: bool = False,
        input_scaling_factor: Optional[float] = None,
        output_scaling_factor: Optional[float] = None,
        clipper: Optional[Clipper] = None,
        score_scaler: Optional[BaseScoreScaler] = None,
        partial_prior=None,
        clipper_gen: Optional[Clipper] = None,
        diffusion_scale=1.0,
        cfm_loss_weight=1.0,
        use_ema=False,
        use_exact_likelihood=False,
        debug_use_train_data=False,
        init_from_prior=False,
        compute_nll_on_train_data=False,
        use_buffer=True,
        tol=1e-5,
        version=1,
        negative_time=False,
        num_negative_time_steps=100,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param buffer: Buffer of sampled objects
        """
        super().__init__()
        # Seems to slow things down
        # torch.set_float32_matmul_precision('high')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net(energy_function=energy_function)
        self.cfm_net = net(energy_function=energy_function)

        if use_ema:
            self.net = EMAWrapper(self.net)
            self.cfm_net = EMAWrapper(self.cfm_net)
        if input_scaling_factor is not None or output_scaling_factor is not None:
            self.net = ScalingWrapper(self.net, input_scaling_factor, output_scaling_factor)

            self.cfm_net = ScalingWrapper(
                self.cfm_net, input_scaling_factor, output_scaling_factor
            )

        self.score_scaler = None
        if score_scaler is not None:
            self.score_scaler = self.hparams.score_scaler(noise_schedule)

            self.net = self.score_scaler.wrap_model_for_unscaling(self.net)
            self.cfm_net = self.score_scaler.wrap_model_for_unscaling(self.cfm_net)

        self.dem_cnf = CNF(
            self.net,
            is_diffusion=True,
            use_exact_likelihood=use_exact_likelihood,
            noise_schedule=noise_schedule,
            method=nll_integration_method,
            num_steps=num_integration_steps,
            atol=tol,
            rtol=tol,
        )
        self.cfm_cnf = CNF(
            self.cfm_net,
            is_diffusion=False,
            use_exact_likelihood=use_exact_likelihood,
            method=nll_integration_method,
            num_steps=num_integration_steps,
            atol=tol,
            rtol=tol,
        )

        self.nll_with_cfm = nll_with_cfm
        self.nll_with_dem = nll_with_dem
        self.nll_on_buffer = nll_on_buffer
        self.logz_with_cfm = logz_with_cfm
        self.cfm_prior_std = cfm_prior_std
        self.compute_nll_on_train_data = compute_nll_on_train_data

        flow_matcher = ConditionalFlowMatcher
        if use_otcfm:
            flow_matcher = ExactOptimalTransportConditionalFlowMatcher

        self.cfm_sigma = cfm_sigma
        self.conditional_flow_matcher = flow_matcher(sigma=cfm_sigma)

        self.nll_integration_method = nll_integration_method

        self.energy_function = energy_function
        self.noise_schedule = noise_schedule
        self.buffer = buffer
        self.dim = self.energy_function.dimensionality

        self.reverse_sde = VEReverseSDE(self.net, self.noise_schedule)

        grad_fxn = estimate_grad_Rt
        if use_richardsons:
            grad_fxn = wrap_for_richardsons(grad_fxn)

        self.clipper = clipper
        self.clipped_grad_fxn = self.clipper.wrap_grad_fxn(grad_fxn)

        self.dem_train_loss = MeanMetric()
        self.cfm_train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_nll_logdetjac = MeanMetric()
        self.test_nll_logdetjac = MeanMetric()
        self.val_nll_log_p_1 = MeanMetric()
        self.test_nll_log_p_1 = MeanMetric()
        self.val_nll = MeanMetric()
        self.test_nll = MeanMetric()
        self.val_nfe = MeanMetric()
        self.test_nfe = MeanMetric()
        self.val_energy_w2 = MeanMetric()
        self.val_dist_w2 = MeanMetric()
        self.val_dist_total_var = MeanMetric()

        self.val_dem_nll_logdetjac = MeanMetric()
        self.test_dem_nll_logdetjac = MeanMetric()
        self.val_dem_nll_log_p_1 = MeanMetric()
        self.test_dem_nll_log_p_1 = MeanMetric()
        self.val_dem_nll = MeanMetric()
        self.test_dem_nll = MeanMetric()
        self.val_dem_nfe = MeanMetric()
        self.test_dem_nfe = MeanMetric()
        self.val_dem_logz = MeanMetric()
        self.val_logz = MeanMetric()
        self.test_dem_logz = MeanMetric()
        self.test_logz = MeanMetric()

        self.val_buffer_nll_logdetjac = MeanMetric()
        self.val_buffer_nll_log_p_1 = MeanMetric()
        self.val_buffer_nll = MeanMetric()
        self.val_buffer_nfe = MeanMetric()
        self.val_buffer_logz = MeanMetric()
        self.test_buffer_nll_logdetjac = MeanMetric()
        self.test_buffer_nll_log_p_1 = MeanMetric()
        self.test_buffer_nll = MeanMetric()
        self.test_buffer_nfe = MeanMetric()
        self.test_buffer_logz = MeanMetric()

        self.val_train_nll_logdetjac = MeanMetric()
        self.val_train_nll_log_p_1 = MeanMetric()
        self.val_train_nll = MeanMetric()
        self.val_train_nfe = MeanMetric()
        self.val_train_logz = MeanMetric()
        self.test_train_nll_logdetjac = MeanMetric()
        self.test_train_nll_log_p_1 = MeanMetric()
        self.test_train_nll = MeanMetric()
        self.test_train_nfe = MeanMetric()
        self.test_train_logz = MeanMetric()

        self.num_init_samples = num_init_samples
        self.num_estimator_mc_samples = num_estimator_mc_samples
        self.num_samples_to_generate_per_epoch = num_samples_to_generate_per_epoch
        self.num_samples_to_sample_from_buffer = num_samples_to_sample_from_buffer
        self.num_integration_steps = num_integration_steps
        self.num_samples_to_save = num_samples_to_save
        self.eval_batch_size = eval_batch_size

        self.prioritize_cfm_training_samples = prioritize_cfm_training_samples
        self.lambda_weighter = self.hparams.lambda_weighter(self.noise_schedule)

        self.last_samples = None
        self.last_energies = None
        self.eval_step_outputs = []

        self.partial_prior = partial_prior

        self.clipper_gen = clipper_gen

        self.diffusion_scale = diffusion_scale
        self.init_from_prior = init_from_prior

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(t, x)

    def get_cfm_loss(self, samples: torch.Tensor) -> torch.Tensor:
        x0 = self.cfm_prior.sample(self.num_samples_to_sample_from_buffer)
        x1 = samples
        x1 = self.energy_function.unnormalize(x1)

        t, xt, ut = self.conditional_flow_matcher.sample_location_and_conditional_flow(x0, x1)

        if self.energy_function.is_molecule and self.cfm_sigma != 0:
            xt = remove_mean(
                xt, self.energy_function.n_particles, self.energy_function.n_spatial_dim
            )

        vt = self.cfm_net(t, xt)
        loss = (vt - ut).pow(2).mean(dim=-1)

        # if self.energy_function.normalization_max is not None:
        #    loss = loss / (self.energy_function.normalization_max ** 2)

        return loss

    def should_train_cfm(self, batch_idx: int) -> bool:
        return self.nll_with_cfm or self.hparams.debug_use_train_data

    def get_score_loss(
        self, times: torch.Tensor, samples: torch.Tensor, noised_samples: torch.Tensor
    ) -> torch.Tensor:
        predicted_score = self.forward(times, noised_samples)

        true_score = -(noised_samples - samples) / (
            self.noise_schedule.h(times).unsqueeze(1) + 1e-4
        )
        error_norms = (predicted_score - true_score).pow(2).mean(-1)
        return error_norms

    def get_loss(self, times: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        estimated_score = estimate_grad_Rt(
            times,
            samples,
            self.energy_function,
            self.noise_schedule,
            num_mc_samples=self.num_estimator_mc_samples,
        )

        if self.clipper is not None and self.clipper.should_clip_scores:
            if self.energy_function.is_molecule:
                estimated_score = estimated_score.reshape(
                    -1,
                    self.energy_function.n_particles,
                    self.energy_function.n_spatial_dim,
                )

            estimated_score = self.clipper.clip_scores(estimated_score)

            if self.energy_function.is_molecule:
                estimated_score = estimated_score.reshape(-1, self.energy_function.dimensionality)

        if self.score_scaler is not None:
            estimated_score = self.score_scaler.scale_target_score(estimated_score, times)

        predicted_score = self.forward(times, samples)

        error_norms = (predicted_score - estimated_score).pow(2).mean(-1)

        return self.lambda_weighter(times) * error_norms

    def training_step(self, batch, batch_idx):
        loss = 0.0
        if not self.hparams.debug_use_train_data:
            if self.hparams.use_buffer:
                iter_samples, _, _ = self.buffer.sample(self.num_samples_to_sample_from_buffer)
            else:
                iter_samples = self.prior.sample(self.num_samples_to_sample_from_buffer)
                # Uncomment for SM
                # iter_samples = self.energy_function.sample_train_set(self.num_samples_to_sample_from_buffer)

            times = torch.rand(
                (self.num_samples_to_sample_from_buffer,), device=iter_samples.device
            )

            noised_samples = iter_samples + (
                torch.randn_like(iter_samples) * self.noise_schedule.h(times).sqrt().unsqueeze(-1)
            )

            if self.energy_function.is_molecule:
                noised_samples = remove_mean(
                    noised_samples,
                    self.energy_function.n_particles,
                    self.energy_function.n_spatial_dim,
                )

            dem_loss = self.get_loss(times, noised_samples)
            # Uncomment for SM
            # dem_loss = self.get_score_loss(times, iter_samples, noised_samples)
            self.log_dict(
                t_stratified_loss(times, dem_loss, loss_name="train/stratified/dem_loss")
            )
            dem_loss = dem_loss.mean()
            loss = loss + dem_loss

            # update and log metrics
            self.dem_train_loss(dem_loss)
            self.log(
                "train/dem_loss",
                self.dem_train_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.should_train_cfm(batch_idx):
            if self.hparams.debug_use_train_data:
                cfm_samples = self.energy_function.sample_train_set(
                    self.num_samples_to_sample_from_buffer
                )
                times = torch.rand(
                    (self.num_samples_to_sample_from_buffer,), device=cfm_samples.device
                )
            else:
                cfm_samples, _, _ = self.buffer.sample(
                    self.num_samples_to_sample_from_buffer,
                    prioritize=self.prioritize_cfm_training_samples,
                )

            cfm_loss = self.get_cfm_loss(cfm_samples)
            self.log_dict(
                t_stratified_loss(times, cfm_loss, loss_name="train/stratified/cfm_loss")
            )
            cfm_loss = cfm_loss.mean()
            self.cfm_train_loss(cfm_loss)
            self.log(
                "train/cfm_loss",
                self.cfm_train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

            loss = loss + self.hparams.cfm_loss_weight * cfm_loss
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        if self.hparams.use_ema:
            self.net.update_ema()
            if self.should_train_cfm(batch_idx):
                self.cfm_net.update_ema()

    def generate_samples(
        self,
        reverse_sde: VEReverseSDE = None,
        num_samples: Optional[int] = None,
        return_full_trajectory: bool = False,
        diffusion_scale=1.0,
        negative_time=False,
    ) -> torch.Tensor:
        num_samples = num_samples or self.num_samples_to_generate_per_epoch

        samples = self.prior.sample(num_samples)

        return self.integrate(
            reverse_sde=reverse_sde,
            samples=samples,
            reverse_time=True,
            return_full_trajectory=return_full_trajectory,
            diffusion_scale=diffusion_scale,
            negative_time=negative_time,
        )

    def integrate(
        self,
        reverse_sde: VEReverseSDE = None,
        samples: torch.Tensor = None,
        reverse_time=True,
        return_full_trajectory=False,
        diffusion_scale=1.0,
        no_grad=True,
        negative_time=False,
    ) -> torch.Tensor:
        trajectory = integrate_sde(
            reverse_sde or self.reverse_sde,
            samples,
            self.num_integration_steps,
            self.energy_function,
            diffusion_scale=diffusion_scale,
            reverse_time=reverse_time,
            no_grad=no_grad,
            negative_time=negative_time,
            num_negative_time_steps=self.hparams.num_negative_time_steps,
        )
        if return_full_trajectory:
            return trajectory

        return trajectory[-1]

    def compute_nll(
        self,
        cnf,
        prior,
        samples: torch.Tensor,
    ):
        aug_samples = torch.cat(
            [samples, torch.zeros(samples.shape[0], 1, device=samples.device)], dim=-1
        )
        aug_output = cnf.integrate(aug_samples)[-1]
        x_1, logdetjac = aug_output[..., :-1], aug_output[..., -1]
        log_p_1 = prior.log_prob(x_1)
        log_p_0 = log_p_1 + logdetjac
        nll = -log_p_0
        return nll, x_1, logdetjac, log_p_1

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        if self.clipper_gen is not None:
            reverse_sde = VEReverseSDE(
                self.clipper_gen.wrap_grad_fxn(self.net), self.noise_schedule
            )
            self.last_samples = self.generate_samples(
                reverse_sde=reverse_sde, diffusion_scale=self.diffusion_scale
            )
            self.last_energies = self.energy_function(self.last_samples)
        else:
            self.last_samples = self.generate_samples(diffusion_scale=self.diffusion_scale)
            self.last_energies = self.energy_function(self.last_samples)

        self.buffer.add(self.last_samples, self.last_energies)

        self._log_energy_w2(prefix="val")

        if self.energy_function.is_molecule:
            self._log_dist_w2(prefix="val")
            self._log_dist_total_var(prefix="val")

    def _log_energy_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
            generated_energies = self.energy_function(generated_samples)
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            _, generated_energies = self.buffer.get_last_n_inserted(self.eval_batch_size)

        energies = self.energy_function(self.energy_function.normalize(data_set))
        energy_w2 = pot.emd2_1d(energies.cpu().numpy(), generated_energies.cpu().numpy())

        self.log(
            f"{prefix}/energy_w2",
            self.val_energy_w2(energy_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _log_dist_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)

        dist_w2 = pot.emd2_1d(
            self.energy_function.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
            self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1),
        )
        self.log(
            f"{prefix}/dist_w2",
            self.val_dist_w2(dist_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _log_dist_total_var(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)

        generated_samples_dists = (
            self.energy_function.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
        )
        data_set_dists = self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1)

        H_data_set, x_data_set = np.histogram(data_set_dists, bins=200)
        H_generated_samples, _ = np.histogram(generated_samples_dists, bins=(x_data_set))
        total_var = (
            0.5
            * np.abs(
                H_data_set / H_data_set.sum() - H_generated_samples / H_generated_samples.sum()
            ).sum()
        )

        self.log(
            f"{prefix}/dist_total_var",
            self.val_dist_total_var(total_var),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def compute_log_z(self, cnf, prior, samples, prefix, name):
        nll, _, _, _ = self.compute_nll(cnf, prior, samples)
        # energy function will unnormalize the samples itself
        logz = self.energy_function(self.energy_function.normalize(samples)) + nll
        logz_metric = getattr(self, f"{prefix}_{name}logz")
        logz_metric.update(logz)
        self.log(
            f"{prefix}/{name}logz",
            logz_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def compute_and_log_nll(self, cnf, prior, samples, prefix, name):
        cnf.nfe = 0.0
        nll, forwards_samples, logdetjac, log_p_1 = self.compute_nll(cnf, prior, samples)
        nfe_metric = getattr(self, f"{prefix}_{name}nfe")
        nll_metric = getattr(self, f"{prefix}_{name}nll")
        logdetjac_metric = getattr(self, f"{prefix}_{name}nll_logdetjac")
        log_p_1_metric = getattr(self, f"{prefix}_{name}nll_log_p_1")
        nfe_metric.update(cnf.nfe)
        nll_metric.update(nll)
        logdetjac_metric.update(logdetjac)
        log_p_1_metric.update(log_p_1)

        self.log_dict(
            {
                f"{prefix}/{name}_nfe": nfe_metric,
                f"{prefix}/{name}nll_logdetjac": logdetjac_metric,
                f"{prefix}/{name}nll_log_p_1": log_p_1_metric,
                # f"{prefix}/{name}logz": logz_metric,
            },
            on_epoch=True,
        )
        self.log(
            f"{prefix}/{name}nll",
            nll_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return forwards_samples

    def eval_step(self, prefix: str, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single eval step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if prefix == "test":
            batch = self.energy_function.sample_test_set(self.eval_batch_size)
        elif prefix == "val":
            batch = self.energy_function.sample_val_set(self.eval_batch_size)

        backwards_samples = self.last_samples

        # generate samples noise --> data if needed
        if backwards_samples is None or self.eval_batch_size > len(backwards_samples):
            backwards_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )

        # sample eval_batch_size from generated samples from dem to match dimensions
        # required for distribution metrics
        if len(backwards_samples) != self.eval_batch_size:
            indices = torch.randperm(len(backwards_samples))[: self.eval_batch_size]
            backwards_samples = backwards_samples[indices]

        if batch is None:
            print("Warning batch is None skipping eval")
            self.eval_step_outputs.append({"gen_0": backwards_samples})
            return

        times = torch.rand((self.eval_batch_size,), device=batch.device)

        noised_batch = batch + (
            torch.randn_like(batch) * self.noise_schedule.h(times).sqrt().unsqueeze(-1)
        )

        if self.energy_function.is_molecule:
            noised_batch = remove_mean(
                noised_batch,
                self.energy_function.n_particles,
                self.energy_function.n_spatial_dim,
            )

        loss = self.get_loss(times, noised_batch).mean(-1)

        # update and log metrics
        loss_metric = self.val_loss if prefix == "val" else self.test_loss
        loss_metric(loss)

        self.log(f"{prefix}/loss", loss_metric, on_step=True, on_epoch=True, prog_bar=True)

        to_log = {
            "data_0": batch,
            "gen_0": backwards_samples,
        }

        if self.nll_with_dem:
            batch = self.energy_function.normalize(batch)
            forwards_samples = self.compute_and_log_nll(
                self.dem_cnf, self.prior, batch, prefix, "dem_"
            )
            to_log["gen_1_dem"] = forwards_samples
            self.compute_log_z(self.cfm_cnf, self.prior, backwards_samples, prefix, "dem_")
        if self.nll_with_cfm:
            forwards_samples = self.compute_and_log_nll(
                self.cfm_cnf, self.cfm_prior, batch, prefix, ""
            )
            to_log["gen_1_cfm"] = forwards_samples

            iter_samples, _, _ = self.buffer.sample(self.eval_batch_size)

            # compute nll on buffer if not training cfm only
            if not self.hparams.debug_use_train_data and self.nll_on_buffer:
                forwards_samples = self.compute_and_log_nll(
                    self.cfm_cnf, self.cfm_prior, iter_samples, prefix, "buffer_"
                )

            if self.compute_nll_on_train_data:
                train_samples = self.energy_function.sample_train_set(self.eval_batch_size)
                forwards_samples = self.compute_and_log_nll(
                    self.cfm_cnf, self.cfm_prior, train_samples, prefix, "train_"
                )

        if self.logz_with_cfm:
            backwards_samples = self.cfm_cnf.generate(
                self.cfm_prior.sample(self.eval_batch_size),
            )[-1]
            # backwards_samples = self.generate_cfm_samples(self.eval_batch_size)
            self.compute_log_z(self.cfm_cnf, self.cfm_prior, backwards_samples, prefix, "")

        self.eval_step_outputs.append(to_log)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("val", batch, batch_idx)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("test", batch, batch_idx)

    def eval_epoch_end(self, prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)
        # convert to dict of tensors assumes [batch, ...]
        outputs = {
            k: torch.cat([dic[k] for dic in self.eval_step_outputs], dim=0)
            for k in self.eval_step_outputs[0]
        }

        unprioritized_buffer_samples, cfm_samples = None, None
        if self.nll_with_cfm:
            unprioritized_buffer_samples, _, _ = self.buffer.sample(
                self.eval_batch_size,
                prioritize=self.prioritize_cfm_training_samples,
            )

            cfm_samples = self.cfm_cnf.generate(
                self.cfm_prior.sample(self.eval_batch_size),
            )[-1]

            self.energy_function.log_on_epoch_end(
                self.last_samples,
                self.last_energies,
                wandb_logger,
                unprioritized_buffer_samples=unprioritized_buffer_samples,
                cfm_samples=cfm_samples,
                replay_buffer=self.buffer,
            )

        else:
            # Only plot dem samples
            self.energy_function.log_on_epoch_end(
                self.last_samples,
                self.last_energies,
                wandb_logger,
            )

        if "data_0" in outputs:
            # pad with time dimension 1
            names, dists = compute_distribution_distances(
                self.energy_function.unnormalize(outputs["gen_0"])[:, None],
                outputs["data_0"][:, None],
                self.energy_function,
            )
            names = [f"{prefix}/{name}" for name in names]
            d = dict(zip(names, dists))
            self.log_dict(d, sync_dist=True)

        self.eval_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self.eval_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        wandb_logger = get_wandb_logger(self.loggers)

        self.eval_epoch_end("test")
        # self._log_energy_w2(prefix="test")
        # if self.energy_function.is_molecule:
        #     self._log_dist_w2(prefix="test")
        #     self._log_dist_total_var(prefix="test")

        batch_size = 1000
        final_samples = []
        n_batches = self.num_samples_to_save // batch_size
        print("Generating samples")
        for i in range(n_batches):
            start = time.time()
            samples = self.generate_samples(
                num_samples=batch_size,
                diffusion_scale=self.diffusion_scale,
                negative_time=self.hparams.negative_time,
            )
            final_samples.append(samples)
            end = time.time()
            print(f"batch {i} took {end - start:0.2f}s")

            if i == 0:
                self.energy_function.log_on_epoch_end(
                    samples,
                    self.energy_function(samples),
                    wandb_logger,
                )

        final_samples = torch.cat(final_samples, dim=0)
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        path = f"{output_dir}/samples_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path)
        print(f"Saving samples to {path}")
        import os

        os.makedirs(self.energy_function.name, exist_ok=True)
        path2 = f"{self.energy_function.name}/samples_{self.hparams.version}_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path2)
        print(f"Saving samples to {path2}")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        def _grad_fxn(t, x):
            return self.clipped_grad_fxn(
                t,
                x,
                self.energy_function,
                self.noise_schedule,
                self.num_estimator_mc_samples,
            )

        reverse_sde = VEReverseSDE(_grad_fxn, self.noise_schedule)

        self.prior = self.partial_prior(device=self.device, scale=self.noise_schedule.h(1) ** 0.5)
        if self.init_from_prior:
            init_states = self.prior.sample(self.num_init_samples)
        else:
            init_states = self.generate_samples(
                reverse_sde, self.num_init_samples, diffusion_scale=self.diffusion_scale
            )
        init_energies = self.energy_function(init_states)

        self.buffer.add(init_states, init_energies)

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            self.cfm_net = torch.compile(self.cfm_net)

        if self.nll_with_cfm:
            self.cfm_prior = self.partial_prior(device=self.device, scale=self.cfm_prior_std)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": self.hparams.lr_scheduler_update_frequency,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = DEMLitModule(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
