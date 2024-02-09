import math
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchmetrics import MeanMetric

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.logging_utils import fig_to_image

from .components.clipper import Clipper
from .components.cnf import CNF
from .components.distribution_distances import compute_distribution_distances
from .components.ema import EMAWrapper
from .components.lambda_weighter import BaseLambdaWeighter
from .components.noise_schedules import BaseNoiseSchedule
from .components.prioritised_replay_buffer import PrioritisedReplayBuffer
from .components.scaling_wrapper import ScalingWrapper
from .components.score_estimator import estimate_grad_Rt
from .components.score_scaler import BaseScoreScaler
from .components.sde_integration import integrate_sde
from .components.sdes import SDE

logtwopi = math.log(2 * math.pi)


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


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


class PISLitModule(LightningModule):
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
        tnet: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        noise_schedule: BaseNoiseSchedule,
        lambda_weighter: BaseLambdaWeighter,
        buffer: PrioritisedReplayBuffer,
        num_init_samples: int,
        num_estimator_mc_samples: int,
        num_samples_to_generate_per_epoch: int,
        eval_batch_size: int,
        num_samples_to_sample_from_buffer: int,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        nll_with_cfm: bool,
        cfm_sigma: float,
        cfm_prior_std: float,
        compile: bool,
        nll_integration_method: str,
        use_richardsons: bool,
        logz_with_cfm: bool,
        use_exact_likelihood=False,
        use_buffer=True,
        prioritize_cfm_training_samples: bool = False,
        input_scaling_factor: Optional[float] = None,
        output_scaling_factor: Optional[float] = None,
        clipper: Optional[Clipper] = None,
        score_scaler: Optional[BaseScoreScaler] = None,
        partial_prior=None,
        clipper_gen: Optional[Clipper] = None,
        diffusion_scale=1.0,
        cfm_loss_weight=1.0,
        pis_scale=1.0,
        time_range=5.0,
        use_ema=False,
        debug_use_train_data=False,
        init_from_prior=False,
        compute_nll_on_train_data=False,
        use_otcfm=False,
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

        self.nll_with_cfm = nll_with_cfm
        self.cfm_prior_std = cfm_prior_std
        self.nll_integration_method = nll_integration_method

        flow_matcher = ConditionalFlowMatcher
        if use_otcfm:
            flow_matcher = ExactOptimalTransportConditionalFlowMatcher

        self.conditional_flow_matcher = flow_matcher(sigma=cfm_sigma)

        self.energy_function = energy_function
        self.buffer = buffer
        self.dim = self.energy_function.dimensionality
        self.time_range = time_range
        self.pis_scale = pis_scale
        self.diffusion_scale = diffusion_scale

        self.clipper = clipper
        self.clipped_grad_fxn = self.clipper.wrap_grad_fxn(estimate_grad_Rt)

        self.tcond = tnet()
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

        self.cfm_cnf = CNF(
            self.cfm_net, is_diffusion=False, use_exact_likelihood=use_exact_likelihood
        )

        self.num_init_samples = num_init_samples
        self.num_estimator_mc_samples = num_estimator_mc_samples
        self.num_samples_to_generate_per_epoch = num_samples_to_generate_per_epoch
        self.eval_batch_size = eval_batch_size
        self.num_samples_to_sample_from_buffer = num_samples_to_sample_from_buffer
        self.num_integration_steps = num_integration_steps

        self.prioritize_cfm_training_samples = prioritize_cfm_training_samples

        self.last_samples = None
        self.last_energies = None

        self.partial_prior = partial_prior
        self.clipper_gen = clipper_gen
        self.compute_nll_on_train_data = compute_nll_on_train_data

        self.outputs = {}

        self.pis_train_loss = MeanMetric()
        self.pis_prior_ll = MeanMetric()
        self.pis_sample_ll = MeanMetric()
        self.pis_reg_loss = MeanMetric()
        self.pis_term_loss = MeanMetric()
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
        self.val_logz = MeanMetric()
        self.test_logz = MeanMetric()

        self.val_gfn_elbo = MeanMetric()
        self.test_gfn_elbo = MeanMetric()

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

    def score(self, x):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                # TODO: should it be _gt_disc or _disc
                energy_output = self.energy_function(copy_x).sum()
                self.energy_function(copy_x).sum().backward()
                lgv_data = copy_x.grad.data
            return lgv_data

    def drift(self, t, x):
        state = x[:, :-1]
        state = torch.nan_to_num(state)
        grad = torch.clip(self.score(state), -1e4, 1e4)
        f = torch.clip(self.net(t, state), -1e4, 1e4)
        dx = torch.nan_to_num(f + self.tcond(t) * grad)
        norm = 0.5 * (dx**2).sum(dim=1, keepdim=True)
        return torch.cat([dx * self.pis_scale, norm], dim=-1)

    def diffusion(self, t, x):
        coeff = torch.ones_like(x) * self.pis_scale
        coeff[..., -1] = 0
        return coeff

    def fwd_traj(self):
        start_time = 0.0
        end_time = self.time_range
        dt = self.time_range / self.num_integration_steps

        times = torch.linspace(
            start_time, end_time, self.num_integration_steps + 1, device=self.device
        )[:-1]

        state = torch.zeros(self.eval_batch_size, self.dim + 1, device=self.device)

        logpf = torch.zeros(self.eval_batch_size, device=self.device)
        logpb = torch.zeros(self.eval_batch_size, device=self.device)

        for t in times:
            noise = torch.randn_like(state, device=self.device)
            dx = self.pis_sde.f(t, state)
            std = self.pis_sde.g(t, state)

            state_ = state + dt * dx + std * noise * np.sqrt(dt)
            logpf += -0.5 * (
                noise[..., :-1] ** 2 + logtwopi + np.log(dt) + torch.log(std[..., :-1] ** 2)
            ).sum(1)

            if t > 0:
                back_mean = state_[..., :-1] - dt * state_[..., :-1] / (t + dt)
                back_var = (self.pis_scale**2) * dt * t / (t + dt)
                noise_backward = (state[..., :-1] - back_mean) / torch.sqrt(back_var)
                logpb += -0.5 * (noise_backward**2 + logtwopi + torch.log(back_var)).sum(1)

            state = state_

        return state[..., :-1], logpf, logpb

    def bwd_traj(self, data):
        start_time = 0.0
        end_time = self.time_range
        dt = self.time_range / self.num_integration_steps

        times = torch.linspace(
            end_time, start_time, self.num_integration_steps + 1, device=self.device
        )[:-1]

        log_pf = torch.zeros(data.shape[0], device=self.device)
        log_pb = torch.zeros(data.shape[0], device=self.device)
        state = data.clone().to(self.device)

        with torch.no_grad():
            for t in times:
                if t > dt:
                    back_mean = state - dt * state / t
                    back_var = ((self.pis_scale**2) * dt * (t - dt)) / t
                    noise = torch.randn_like(state, device=self.device)
                    state_ = back_mean + torch.sqrt(back_var) * noise
                    log_pb += -0.5 * (noise**2 + logtwopi + torch.log(back_var)).sum(1)
                else:
                    state_ = torch.zeros_like(state, device=self.device)

                aug_state = torch.cat([state_, torch.zeros_like(state_[..., :1])], dim=-1)
                forward_mean = self.pis_sde.f(t - dt, aug_state)[..., :-1]
                forward_var = self.pis_sde.g(t - dt, aug_state)[..., :-1] ** 2

                noise = ((state - state_) - dt * forward_mean) / (
                    np.sqrt(dt) * torch.sqrt(forward_var)
                )
                log_pf += -0.5 * (noise**2 + logtwopi + np.log(dt) + torch.log(forward_var)).sum(
                    1
                )

                state = state_

        return log_pf, log_pb

    def step_with_uw(self, t, state, dt):
        noise = torch.randn_like(state) * np.sqrt(dt)
        f_value = self.pis_sde.f(t, state)
        g_coeff = self.pis_sde.g(t, state)
        new_state = state + f_value * dt + g_coeff * noise
        uw_term = (f_value[:, :-1] * noise[:, :-1]).sum(dim=1) / self.pis_scale
        return new_state, uw_term

    def get_cfm_loss(self, samples: torch.Tensor) -> torch.Tensor:
        x0 = self.cfm_prior.sample(self.num_samples_to_sample_from_buffer)
        # x0 = (
        #     torch.randn(
        #         self.num_samples_to_sample_from_buffer,
        #         self.energy_function.dimensionality,
        #         device=self.device,
        #     )
        #     * self.cfm_prior_std
        # )
        x1 = samples

        t, xt, ut = self.conditional_flow_matcher.sample_location_and_conditional_flow(x0, x1)

        vt = self.cfm_net(t, xt)
        return (vt - ut).pow(2).mean(dim=-1)

    def should_train_cfm(self, batch_idx: int) -> bool:
        return self.nll_with_cfm

    def get_loss(self):
        aug_prior_samples = torch.zeros(
            self.num_samples_to_sample_from_buffer, self.dim + 1, device=self.device
        )

        aug_output = self.integrate(
            self.pis_sde,
            aug_prior_samples,
            return_full_trajectory=True,
            no_grad=False,
            reverse_time=False,
            time_range=self.time_range,
        )[-1]
        x_1, quad_reg = aug_output[..., :-1], aug_output[..., -1]
        prior_ll = self.prior.log_prob(x_1).mean() / (self.dim + 1)
        sample_ll = self.energy_function(x_1).mean() / (self.dim + 1)
        term_loss = prior_ll - sample_ll
        quad_reg = (quad_reg).mean() / (self.dim + 1)
        pis_loss = term_loss + quad_reg

        return pis_loss, prior_ll, sample_ll, quad_reg, term_loss

    def generate_samples(
        self,
        sde,
        num_samples: Optional[int] = None,
        return_full_trajectory: bool = False,
        diffusion_scale=1.0,
    ) -> torch.Tensor:
        num_samples = num_samples or self.num_samples_to_generate_per_epoch
        samples = torch.zeros(num_samples, self.dim + 1, device=self.device)

        return self.integrate(
            sde=sde,
            samples=samples,
            reverse_time=False,
            return_full_trajectory=return_full_trajectory,
            diffusion_scale=diffusion_scale,
            time_range=self.time_range,
        )[..., :-1]

    def integrate(
        self,
        sde=None,
        samples: torch.Tensor = None,
        reverse_time=True,
        return_full_trajectory=False,
        diffusion_scale=1.0,
        no_grad=True,
        time_range=1.0,
    ) -> torch.Tensor:
        trajectory = integrate_sde(
            sde or self.pis_sde,
            samples,
            self.num_integration_steps,
            diffusion_scale=diffusion_scale,
            reverse_time=reverse_time,
            no_grad=no_grad,
            time_range=time_range,
        )
        if return_full_trajectory:
            return trajectory
        return trajectory[-1]

    def compute_nll(
        self,
        cnf,
        prior,
        samples: torch.Tensor,
        num_integration_steps=1,
        method="dopri5",
    ):
        aug_samples = torch.cat(
            [samples, torch.zeros(samples.shape[0], 1, device=samples.device)], dim=-1
        )

        num_integration_steps = self.num_integration_steps
        if self.nll_integration_method == "dopri5":
            num_integration_steps = 1

        aug_output = cnf.integrate(
            aug_samples,
            num_integration_steps=num_integration_steps,
            method=self.nll_integration_method,
        )[-1]
        x_1, logdetjac = aug_output[..., :-1], aug_output[..., -1]
        log_p_1 = prior.log_prob(x_1)
        log_p_0 = log_p_1 + logdetjac
        nll = -log_p_0
        return nll, x_1, logdetjac, log_p_1

    def training_step(self, batch, batch_idx):
        loss = 0.0
        pis_loss, prior_ll, sample_ll, quad_reg, term_loss = self.get_loss()
        self.pis_train_loss(pis_loss)
        self.pis_prior_ll(prior_ll)
        self.pis_sample_ll(sample_ll)
        self.pis_reg_loss(quad_reg)
        self.pis_term_loss(term_loss)

        loss = loss + pis_loss

        if self.should_train_cfm(batch_idx):
            if self.hparams.debug_use_train_data:
                cfm_samples = self.energy_function.sample_train_set(
                    self.num_samples_to_sample_from_buffer
                )
            else:
                cfm_samples, _, _ = self.buffer.sample(
                    self.num_samples_to_sample_from_buffer,
                    prioritize=self.prioritize_cfm_training_samples,
                )
            times = torch.rand(
                (self.num_samples_to_sample_from_buffer,), device=cfm_samples.device
            )

            cfm_loss = self.get_cfm_loss(cfm_samples)
            self.log_dict(
                t_stratified_loss(times, cfm_loss, loss_name="train/stratified/cfm_loss")
            )
            cfm_loss = cfm_loss.mean()
            self.cfm_train_loss(cfm_loss)
            loss = loss + self.hparams.cfm_loss_weight * cfm_loss
            self.log(
                "train/cfm_loss",
                self.cfm_train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

        self.log_dict(
            {
                "train/pis_loss": self.pis_train_loss,
                "train/pis_prior_ll": self.pis_prior_ll,
                "train/pis_sample_ll": self.pis_sample_ll,
                "train/pis_reg_loss": self.pis_reg_loss,
                "train/pis_term_loss": self.pis_term_loss,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss

    def compute_log_z(self, cnf, prior, samples, prefix, name):
        nll, forwards_samples, logdetjac, log_p_1 = self.compute_nll(cnf, prior, samples)
        logz = self.energy_function(samples) + nll
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

    def pis_logZ(self):
        times = torch.linspace(0.0, self.time_range, self.num_integration_steps + 1).to(
            self.device
        )[:-1]
        uw_term = 0
        dt = 1.0 / self.num_integration_steps
        state = torch.zeros(self.eval_batch_size, self.dim + 1).to(self.device)
        for t in times:
            state, cur_uw_term = self.step_with_uw(t, state, dt)
            uw_term += cur_uw_term

        loss = (
            state[:, -1]
            + uw_term
            + self.prior.log_prob(state[:, :-1])
            - self.energy_function(state[:, :-1])
        )

        log_weight = -loss + loss.mean()
        unnormal_weight = torch.exp(log_weight)
        weight = unnormal_weight / unnormal_weight.sum()
        half_unnormal_weight = torch.exp(log_weight / 2)
        half_weigh = half_unnormal_weight / half_unnormal_weight.sum()

        logZ_lb = -loss.mean()
        logZ_ub = torch.sum(-weight * loss)
        logZ_hb = torch.sum(-half_weigh * loss)
        logZ = torch.log(torch.mean(torch.exp(log_weight))) - loss.mean()

        return state[:, :-1], logZ_lb, logZ_ub, logZ_hb, logZ

    def gfn_log_Z(self):
        state, log_pf, log_pb = self.fwd_traj()
        log_r = self.energy_function(state)
        log_weight = log_r - log_pf + log_pb
        log_Z = logmeanexp(log_weight)
        log_Z_lb = log_weight.mean()

        return state, log_Z, log_Z_lb

    def get_elbo(self, data, prefix, name, num_evals=10):
        bsz = data.shape[0]
        data = data.view(bsz, 1, self.dim).repeat(1, num_evals, 1).view(bsz * num_evals, self.dim)
        log_pf, log_pb = self.bwd_traj(data)
        log_weight = (log_pf - log_pb).view(bsz, num_evals)
        log_weight = logmeanexp(log_weight, dim=1)
        elbo_metric = getattr(self, f"{prefix}_{name}gfn_elbo")
        elbo_metric.update(log_weight)
        self.log(
            f"{prefix}/gfn_elbo",
            elbo_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        if self.hparams.use_ema:
            self.net.update_ema()
            if self.should_train_cfm(batch_idx):
                self.cfm_net.update_ema()

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        # self.last_samples = self.generate_samples()
        # self.last_energies = self.energy_function(self.last_samples)
        self.last_samples = self.generate_samples(
            self.pis_sde, diffusion_scale=self.diffusion_scale
        )
        self.last_energies = self.energy_function(self.last_samples)
        self.buffer.add(self.last_samples, self.last_energies)

    def eval_step(self, prefix: str, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single eval step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch = self.energy_function.sample_test_set(self.num_samples_to_generate_per_epoch)
        loss = self.get_loss()[0]

        # update and log metrics
        loss_metric = self.val_loss if prefix == "val" else self.test_loss
        loss_metric(loss)

        if self.last_samples is None:
            self.last_samples = self.generate_samples(
                self.pis_sde, diffusion_scale=self.diffusion_scale
            )

        self.outputs[f"{prefix}/data"] = batch
        self.outputs[f"{prefix}/gen"] = self.last_samples

        self.log(f"{prefix}/loss", loss_metric, on_step=False, on_epoch=True, prog_bar=True)

        batch = self.energy_function.sample_test_set(self.eval_batch_size)

        self.get_elbo(batch, prefix, "")

        if self.nll_with_cfm:
            prior_samples = self.compute_and_log_nll(
                self.cfm_cnf, self.cfm_prior, batch, prefix, ""
            )
            self.outputs[f"{prefix}/cfm_prior"] = prior_samples

            if self.compute_nll_on_train_data:
                train_samples = self.energy_function.sample_train_set(self.eval_batch_size)
                _ = self.compute_and_log_nll(
                    self.cfm_cnf, self.cfm_prior, train_samples, prefix, "train_"
                )

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("val", batch, batch_idx)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("test", batch, batch_idx)

    def generate_cfm_samples(self, batch_size):
        def reverse_wrapper(model):
            def fxn(t, x, args=None):
                if t.ndim == 0:
                    t = t.unsqueeze(0)

                return model(t.repeat(len(x)), x)

            return fxn

        with torch.no_grad():
            shape = (
                batch_size,
                self.energy_function.dimensionality,
            )

            num_integration_steps = self.num_integration_steps
            if self.nll_integration_method == "dopri5":
                num_integration_steps = 2

            # noise = torch.randn(shape, device=self.device) * self.cfm_prior_std
            noise = self.cfm_prior.sample(batch_size)

            try:
                traj = odeint(
                    reverse_wrapper(self.cfm_net),
                    noise,
                    t=torch.linspace(0, 1, num_integration_steps, device=self.device),
                    method=self.nll_integration_method,
                )
                return traj[-1]
            except (RuntimeError, AssertionError) as e:
                print(e)
                print("Falling back on fixed-step integration")
                self.nfe = 0.0
                time = torch.linspace(0, 1, 1000 + 1, device=self.device)
                return odeint(reverse_wrapper(self.cfm_net), noise, t=time, method="euler")[-1]

    def scatter_prior(self, prefix, outputs):
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None:
            return
        fig, ax = plt.subplots()
        n_samples = outputs.shape[0]
        ax.scatter(*outputs.detach().cpu().T, label="Generated prior")
        ax.scatter(
            *self.prior.sample(n_samples).cpu().T,
            label="True prior",
            alpha=0.5,
        )
        ax.legend()
        wandb_logger.log_image(f"{prefix}/generated_prior", [fig_to_image(fig)])

    def eval_epoch_end(self, prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)

        samples_gfn, gfn_logZ, logZ_lb = self.gfn_log_Z()
        samples_pis, pis_logZ_lb, pis_logZ_ub, pis_logZ_hb, pis_logZ = self.pis_logZ()
        buffer_samples, _, _ = self.buffer.sample(self.eval_batch_size)

        metrics = {
            f"{prefix}/gfn_logZ": gfn_logZ,
            f"{prefix}/logZ_lb": logZ_lb,
            f"{prefix}/pis_logZ_lb": pis_logZ_lb,
            f"{prefix}/pis_logZ_ub": pis_logZ_ub,
            f"{prefix}/pis_logZ_hb": pis_logZ_hb,
            f"{prefix}/pis_logZ": pis_logZ,
        }

        self.energy_function.log_samples(
            samples_gfn, wandb_logger, f"{prefix}_samples/gfn_samples"
        )
        self.energy_function.log_samples(
            samples_pis, wandb_logger, f"{prefix}_samples/pis_samples"
        )
        self.energy_function.log_samples(
            buffer_samples, wandb_logger, f"{prefix}_samples/buffer_samples"
        )

        if self.nll_with_cfm:
            # Generate data from the CFM
            # Calculate logZ based on that data
            cfm_samples = self.generate_cfm_samples(self.eval_batch_size)
            self.compute_log_z(self.cfm_cnf, self.cfm_prior, cfm_samples, prefix, "")
            unprioritized_buffer_samples, _, _ = self.buffer.sample(
                self.eval_batch_size,
                prioritize=self.prioritize_cfm_training_samples,
            )
            if self.energy_function.dimensionality == 2:
                self.scatter_prior(prefix + "_samples", self.outputs[f"{prefix}/cfm_prior"])
            self.energy_function.log_samples(
                cfm_samples, wandb_logger, f"{prefix}_samples/cfm_samples"
            )
            self.energy_function.log_samples(
                unprioritized_buffer_samples,
                wandb_logger,
                f"{prefix}_samples/unprioritized_buffer_samples",
            )

        # pad with time dimension 1
        names, dists = compute_distribution_distances(
            self.outputs[f"{prefix}/gen"][:, None], self.outputs[f"{prefix}/data"][:, None]
        )
        names = [f"{prefix}/{name}" for name in names]
        d = dict(zip(names, dists))
        metrics.update(d)
        self.log_dict(metrics, sync_dist=True)
        self.outputs = {}

    def on_validation_epoch_end(self) -> None:
        self.eval_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self.eval_epoch_end("test")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        self.prior = self.partial_prior(
            device=self.device, scale=self.pis_scale * np.sqrt(self.time_range)
        )
        self.net = self.net.to(self.device)
        self.tcond = self.tcond.to(self.device)
        self.pis_sde = SDE(self.drift, self.diffusion).to(self.device)
        init_states = self.generate_samples(
            self.pis_sde, self.num_init_samples, diffusion_scale=self.diffusion_scale
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
