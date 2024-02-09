from abc import ABC, abstractmethod

import torch

from .noise_schedules import BaseNoiseSchedule


class BaseLambdaWeighter(ABC):
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BasicLambdaWeighter(BaseLambdaWeighter):
    def __init__(self, noise_schedule: BaseNoiseSchedule, epsilon: float = 1e-3):
        self.noise_schedule = noise_schedule
        self.epsilon = epsilon

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.noise_schedule.h(t) + self.epsilon


class NoLambdaWeighter(BasicLambdaWeighter):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1
