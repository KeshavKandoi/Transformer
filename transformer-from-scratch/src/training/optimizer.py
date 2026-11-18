"""AdamW construction for the trainable model parameters."""

from torch import nn
from torch.optim import AdamW

from .config import OptimizerConfig


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> AdamW:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("AdamW needs at least one trainable parameter")
    return AdamW(
        parameters, lr=config.learning_rate, betas=(config.beta1, config.beta2),
        eps=config.epsilon, weight_decay=config.weight_decay,
    )
