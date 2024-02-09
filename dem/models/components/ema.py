"""Exponential moving average wrapper for torch.nn.Module."""
import torch


class EMAWrapper(torch.nn.Module):
    """Implements exponential moving average model wrapper.

    Wraps a model where `ema.update_ema()` can be manually called to update ema weights which are
    separately saved.

    with `ema.eval()` activates the EMA weights of the model for eval mode and backs up the current
    weights.

    with `ema.train()` restores current weights.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        warmup_denominator: float = 10,
    ):
        super().__init__()
        self.model = model
        self.decay = decay
        self.warmup_denominator = warmup_denominator

        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))

        self.shadow_params = torch.nn.ParameterList(
            [
                torch.nn.Parameter(p.clone().detach(), requires_grad=False)
                for p in model.parameters()
                if p.requires_grad
            ]
        )
        self.backup_params = []

    def train(self, mode: bool = True) -> None:
        """Switch between training and eval mode."""
        use_training_mode = mode
        if self.training and not use_training_mode:
            self.backup()
            self.copy_to_model()
        elif not self.training and use_training_mode:
            self.restore_to_model()
        # else: N/A; we only care if self.training != use_training_mode
        super().train(use_training_mode)

    def _get_decay(self, num_updates: int) -> float:
        """Decay warmup magic from meta."""
        return min(self.decay, (1 + num_updates) / (self.warmup_denominator + num_updates))

    def update_ema(self) -> None:
        """Update the shadow params with a new EMA update."""
        self.num_updates += 1  # pylint: disable=no-member
        num_updates = self.num_updates.item()  # pylint: disable=no-member
        decay = self._get_decay(num_updates)
        with torch.no_grad():
            params = [p for p in self.model.parameters() if p.requires_grad]
            for shadow, param in zip(self.shadow_params, params):
                shadow.sub_((1 - decay) * (shadow - param))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def copy_to_model(self) -> None:
        """Copy the shadow params to the model in-place."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        for shadow, param in zip(self.shadow_params, params):
            param.data.copy_(shadow.data)

    def backup(self) -> None:
        """Create a backup of the model current params by creating a new copy or copying in-
        place."""
        if len(self.backup_params) > 0:
            for p, b in zip(self.model.parameters(), self.backup_params):
                b.data.copy_(p.data)
        else:
            self.backup_params = [param.clone() for param in self.model.parameters()]

    def restore_to_model(self) -> None:
        """Move the backup parameters into the current model's parameters in-place."""
        for param, backup in zip(self.model.parameters(), self.backup_params):
            param.data.copy_(backup.data)
