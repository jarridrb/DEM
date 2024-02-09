import copy
from typing import Optional

import torch

from .mlp import TimeConder


class PISNN(torch.nn.Module):
    def __init__(
        self,
        f_func: torch.nn.Module,
        nn_clip: float = 1e2,
        lgv_clip: float = 1e2,
        energy_function=None,
        f_format: Optional[str] = None,
    ):
        super().__init__()
        self.energy_function = energy_function
        self.f_func = f_func
        self.nn_clip = nn_clip
        self.lgv_clip = lgv_clip
        self.select_f(f_format)

    def select_f(self, f_format=None):
        if f_format == "f":

            def _fn(t, x):
                return torch.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)

        elif f_format == "t_tnet_grad":
            self.lgv_coef = TimeConder(64, 1, 3)

            def _fn(t, x):
                grad_fxn = torch.vmap(torch.func.grad(self.energy_function.__call__))
                grad = torch.clip(grad_fxn(x), -self.lgv_clip, self.lgv_clip)
                f = torch.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)
                return f - self.lgv_coef(t) * grad

        elif f_format == "nn_grad":

            def _fn(t, x):
                x_dot = torch.clip(self.energy_function.score(x), -self.lgv_clip, self.lgv_clip)
                f_x = torch.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)
                return f_x * x_dot

        elif f_format == "comp_grad":
            self.grad_net = copy.deepcopy(self.f_func)

            def _fn(t, x):
                x_dot = torch.clip(self.energy_function(x), -self.lgv_clip, self.lgv_clip)
                f_x = torch.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)
                f_x_dot = torch.clip(self.grad_net(t, x_dot), -self.nn_clip, self.nn_clip)
                return f_x + f_x_dot

        else:
            _fn = self.f_func

        self.param_fn = _fn

    def forward(self, t, x):
        return self.param_fn(t, x)
