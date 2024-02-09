import numpy as np
import torch

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.clipper import Clipper
from dem.models.components.noise_schedules import BaseNoiseSchedule


def wrap_for_richardsons(score_estimator):
    def _fxn(t, x, energy_function, noise_schedule, num_mc_samples):
        bigger_samples = score_estimator(t, x, energy_function, noise_schedule, num_mc_samples)

        smaller_samples = score_estimator(
            t, x, energy_function, noise_schedule, int(num_mc_samples / 2)
        )

        return (2 * bigger_samples) - smaller_samples

    return _fxn


def log_expectation_reward(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    clipper: Clipper = None,
):
    repeated_t = t.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)
    repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)

    h_t = noise_schedule.h(repeated_t).unsqueeze(1)

    samples = repeated_x + (torch.randn_like(repeated_x) * h_t.sqrt())

    log_rewards = energy_function(samples)

    if clipper is not None and clipper.should_clip_log_rewards:
        log_rewards = clipper.clip_log_rewards(log_rewards)

    return torch.logsumexp(log_rewards, dim=-1) - np.log(num_mc_samples)


def estimate_grad_Rt(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
):
    if t.ndim == 0:
        t = t.unsqueeze(0).repeat(len(x))

    grad_fxn = torch.func.grad(log_expectation_reward, argnums=1)
    vmapped_fxn = torch.vmap(grad_fxn, in_dims=(0, 0, None, None, None), randomness="different")

    return vmapped_fxn(t, x, energy_function, noise_schedule, num_mc_samples)
