from typing import Optional

import matplotlib.pyplot as plt
import torch
from fab.target_distributions import gmm
from fab.utils.plotting import plot_contours, plot_marginal_pair
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.replay_buffer import ReplayBuffer
from dem.utils.logging_utils import fig_to_image


class GMM(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        n_mixes=40,
        loc_scaling=40,
        log_var_scaling=1.0,
        device="cpu",
        true_expectation_estimation_n_samples=int(1e5),
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=50,
        train_set_size=100000,
        test_set_size=2000,
        val_set_size=2000,
    ):
        use_gpu = device != "cpu"
        torch.manual_seed(0)  # seed of 0 for GMM problem
        self.gmm = gmm.GMM(
            dim=dimensionality,
            n_mixes=n_mixes,
            loc_scaling=loc_scaling,
            log_var_scaling=log_var_scaling,
            use_gpu=use_gpu,
            true_expectation_estimation_n_samples=true_expectation_estimation_n_samples,
        )

        self.curr_epoch = 0
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.should_unnormalize = should_unnormalize
        self.data_normalization_factor = data_normalization_factor

        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size

        self.name = "gmm"

        super().__init__(
            dimensionality=dimensionality,
            normalization_min=-data_normalization_factor,
            normalization_max=data_normalization_factor,
        )

    def setup_test_set(self):
        # test_sample = self.gmm.sample((self.test_set_size,))
        # return test_sample
        return self.gmm.test_set

    def setup_train_set(self):
        train_samples = self.gmm.sample((self.train_set_size,))
        return self.normalize(train_samples)

    def setup_val_set(self):
        val_samples = self.gmm.sample((self.val_set_size,))
        return val_samples

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        if self.should_unnormalize:
            samples = self.unnormalize(samples)

        return self.gmm.log_prob(samples)

    @property
    def dimensionality(self):
        return 2

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            if self.should_unnormalize:
                # Don't unnormalize CFM samples since they're in the
                # unnormalized space
                if latest_samples is not None:
                    latest_samples = self.unnormalize(latest_samples)

                if unprioritized_buffer_samples is not None:
                    unprioritized_buffer_samples = self.unnormalize(unprioritized_buffer_samples)

            if unprioritized_buffer_samples is not None:
                buffer_samples, _, _ = replay_buffer.sample(self.plotting_buffer_sample_size)
                if self.should_unnormalize:
                    buffer_samples = self.unnormalize(buffer_samples)

                samples_fig = self.get_dataset_fig(buffer_samples, latest_samples)

                wandb_logger.log_image(f"{prefix}unprioritized_buffer_samples", [samples_fig])

            if cfm_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(unprioritized_buffer_samples, cfm_samples)

                wandb_logger.log_image(f"{prefix}cfm_generated_samples", [cfm_samples_fig])

            if latest_samples is not None:
                fig, ax = plt.subplots()
                ax.scatter(*latest_samples.detach().cpu().T)

                wandb_logger.log_image(f"{prefix}generated_samples_scatter", [fig_to_image(fig)])
                img = self.get_single_dataset_fig(latest_samples, "dem_generated_samples")
                wandb_logger.log_image(f"{prefix}generated_samples", [img])

            plt.close()

        self.curr_epoch += 1

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
        should_unnormalize: bool = False,
    ) -> None:
        if wandb_logger is None:
            return

        if self.should_unnormalize and should_unnormalize:
            samples = self.unnormalize(samples)
        samples_fig = self.get_single_dataset_fig(samples, name)
        wandb_logger.log_image(f"{name}", [samples_fig])

    def get_single_dataset_fig(self, samples, name, plotting_bounds=(-1.4 * 40, 1.4 * 40)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        self.gmm.to("cpu")
        plot_contours(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=ax,
            n_contour_levels=50,
            grid_width_n_points=200,
        )

        plot_marginal_pair(samples, ax=ax, bounds=plotting_bounds)
        ax.set_title(f"{name}")

        self.gmm.to(self.device)

        return fig_to_image(fig)

    def get_dataset_fig(self, samples, gen_samples=None, plotting_bounds=(-1.4 * 40, 1.4 * 40)):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        self.gmm.to("cpu")
        plot_contours(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=axs[0],
            n_contour_levels=50,
            grid_width_n_points=200,
        )

        # plot dataset samples
        plot_marginal_pair(samples, ax=axs[0], bounds=plotting_bounds)
        axs[0].set_title("Buffer")

        if gen_samples is not None:
            plot_contours(
                self.gmm.log_prob,
                bounds=plotting_bounds,
                ax=axs[1],
                n_contour_levels=50,
                grid_width_n_points=200,
            )
            # plot generated samples
            plot_marginal_pair(gen_samples, ax=axs[1], bounds=plotting_bounds)
            axs[1].set_title("Generated samples")

        # delete subplot
        else:
            fig.delaxes(axs[1])

        self.gmm.to(self.device)

        return fig_to_image(fig)
