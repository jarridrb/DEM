from typing import Optional

import torch


class ScalingWrapper(torch.nn.Module):
    """(Tries to) normalize data and blah blah blah."""

    def __init__(
        self,
        network: torch.nn.Module,
        input_scaling_factor: Optional[float] = None,
        output_scaling_factor: Optional[float] = None,
    ):
        super().__init__()

        self.network = network
        self.input_scaling_factor = input_scaling_factor
        self.output_scaling_factor = output_scaling_factor

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.input_scaling_factor is not None:
            x = x / self.input_scaling_factor

        out = self.network(t, x)
        if self.output_scaling_factor is not None:
            out = out / self.output_scaling_factor

        return out
