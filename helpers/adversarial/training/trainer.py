"""
Key functions to drop into helpers/training/trainer.py

Adversarial trainer for Flux lycoris lokr.
This trainer extends the original Trainer by adding an adversarial training loop:
- The generator is composed of the pre-trained flux transformer (with lycoris lokr for adapter training).
- The generator's main parameters are frozen.
- The discriminator is constructed using FluxTransformer2DDiscriminator (from transformer_D.py) that hooks into
  the generator's transformer for feature extraction.
  
Two optimizers are created:
  • One for the generator (update only the lycoris adapter parameters).
  • One for the discriminator (with a configurable learning rate).

The training loop alternates between discriminator updates and generator updates.
Placeholders for the adversarial loss computations are included as TODO comments.
"""
from enum import Enum
import torch
from helpers.adversarial.core.network.transformer_D import FluxTransformer2DDiscriminator
from diffusers.models.transformers import FluxTransformer2DModel
from typing import Union
from helpers.training.trainer import Trainer

class Phase(Enum):
    D = "D"  # Discriminator phase
    G = "G"  # Generator phase

# PREDICTION
def model_predict(
        trainer: Trainer,
        prepared_batch: dict,
        ) -> torch.Tensor:
    phase = trainer.phase
    if phase == Phase.D:
        return discriminator_predict(trainer=trainer, prepared_batch=prepared_batch)
    elif phase == Phase.G:
        return generator_predict(trainer=trainer, prepared_batch=prepared_batch)
    else:
        raise ValueError(f"Invalid phase: {phase}")

def generator_predict(
        trainer: Trainer,
        prepared_batch: dict,
        ) -> torch.Tensor:
    # TODO: implement the generator prediction
    return torch.randn(prepared_batch["latents"].shape)

def discriminator_predict(
        trainer: Trainer,
        prepared_batch: dict,
        ) -> torch.Tensor:
    # TODO: implement the discriminator prediction
    return torch.randn(prepared_batch["latents"].shape)

# LOSS
def calculate_loss(
        trainer: Trainer,
        prepared_batch: dict,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        ):
    phase = trainer.phase
    if phase == Phase.D:
        return calculate_discriminator_loss(prepared_batch, model_pred, target)
    elif phase == Phase.G:
        return calculate_generator_loss(prepared_batch, model_pred, target)
    else:
        raise ValueError(f"Invalid phase: {phase}")

def calculate_generator_loss(
        prepared_batch: dict,
        model_pred,
        target,
        ):
    pass

def calculate_discriminator_loss(
        prepared_batch: dict,
        model_pred,
        target,
        ):
    pass
