from typing import Protocol, Any
from accelerate import Accelerator
from torch.optim import Optimizer

class TrainerProtocol(Protocol):
    accelerator: Accelerator
    config: Any
    state: dict
    transformer: Any
    discriminator: Any
    optimizer: Optimizer
    train_loss: float
    phase: Any
    timesteps_buffer: list
    # etc... 