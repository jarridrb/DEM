from abc import ABC, abstractmethod

import torch

from .noise_schedules import BaseNoiseSchedule


class BaseScoreScaler(ABC):
    @abstractmethod
    def scale_target_score(self, target_score: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def wrap_model_for_unscaling(self, model: torch.nn.Module) -> torch.nn.Module:
        pass


class BadScoreScaler(BaseScoreScaler):
    def __init__(
        self,
        noise_schedule: BaseNoiseSchedule,
        constant_scaling_factor: float,
        epsilon: float,
    ):
        self.noise_schedule = noise_schedule
        self.constant_scaling_factor = constant_scaling_factor
        self.epsilon = epsilon

    def _get_scale_factor(self, score: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        # call view to expand h_t to the number of dimensions of target_score
        h_t = self.noise_schedule.h(times).view(-1, *(1 for _ in range(score.ndim - times.ndim)))

        h_t = h_t * self.constant_scaling_factor

        return (h_t * self.constant_scaling_factor) + self.epsilon

    def scale_target_score(self, target_score: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return target_score * self._get_scale_factor(target_score, times)

    def _build_wrapper_class(self):
        class _ScalingOutputWrapper(torch.nn.Module):
            def __init__(inner_self, model: torch.nn.Module):
                super().__init__()
                inner_self.model = model

            def forward(inner_self, t, x):
                out_score = inner_self.model(t, x)
                return out_score / self._get_scale_factor(out_score, t)

        return _ScalingOutputWrapper

    def wrap_model_for_unscaling(self, model: torch.nn.Module) -> torch.nn.Module:
        wrapper_class = self._build_wrapper_class()
        return wrapper_class(model)
