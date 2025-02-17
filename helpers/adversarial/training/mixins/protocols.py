from typing import Protocol, Any, Callable
from accelerate import Accelerator
from torch.optim import Optimizer
from torch import nn


class TrainerProtocol(Protocol):
    accelerator: Accelerator
    config: Any
    state: dict
    transformer: nn.Module
    discriminator: nn.Module
    optimizer: Optimizer
    train_loss: float
    phase: Any
    timesteps_buffer: list
    _get_trainable_parameters: Callable
    lycoris_wrapped_network: nn.Module