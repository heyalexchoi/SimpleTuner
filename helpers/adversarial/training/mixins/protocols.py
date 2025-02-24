from typing import Protocol, Any, Callable
from accelerate import Accelerator
from torch.optim import Optimizer
from torch import nn
from helpers.adversarial.core.constants import Phase
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel  # assuming this is the generator model
from helpers.adversarial.core.network.flux_discriminator import FluxTransformer2DDiscriminator


class TrainerProtocol(Protocol):
    accelerator: Accelerator
    config: Any
    state: dict
    transformer: nn.Module
    optimizer: Optimizer
    train_loss: float
    grad_norm: float
    phase: Any
    timesteps_buffer: list
    _get_trainable_parameters: Callable
    lycoris_wrapped_network: nn.Module

class AdversarialTrainerProtocol(TrainerProtocol):
    transformer: FluxTransformer2DModel
    discriminator: FluxTransformer2DDiscriminator
    generator_optimizer: Optimizer
    discriminator_optimizer: Optimizer
    phase: Phase
    current_step_loss_components: dict