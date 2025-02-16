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
from typing import Union, TYPE_CHECKING, cast
from helpers.training.trainer import Trainer

class Phase(Enum):
    D = "D"  # Discriminator phase
    G = "G"  # Generator phase

if TYPE_CHECKING:
    from helpers.training.trainer import Trainer
    from helpers.adversarial.core.network.transformer_D import FluxTransformer2DDiscriminator
    from diffusers.models.transformers import FluxTransformer2DModel

# PREDICTION
def model_predict(
        trainer: "Trainer",  # quotes because of circular import
        prepared_batch: dict,
        ) -> torch.Tensor:
    """Generate predictions based on current phase"""
    assert trainer.config.enable_adversarial_training, "Adversarial training must be enabled"
    assert trainer.discriminator is not None, "Discriminator must be initialized"
    
    if trainer.phase == Phase.D:
        return discriminator_predict(trainer, prepared_batch)
    elif trainer.phase == Phase.G:
        return generator_predict(trainer, prepared_batch)
    else:
        raise ValueError(f"Invalid phase: {trainer.phase}")

def generator_predict(
        trainer: Trainer,
        prepared_batch: dict,
        ) -> torch.Tensor:
    """Generate predictions using the generator (transformer/unet)"""
    assert trainer.config.flow_matching, "Flow matching must be enabled"
    assert trainer.transformer is not None, "Transformer must be initialized"
    
    # AC: double check the logic here. we are using flux which is flow matching.
    noisy_latents = prepared_batch["noisy_latents"]
    timesteps = prepared_batch["timesteps"]
    encoder_hidden_states = prepared_batch["encoder_hidden_states"]
    added_cond_kwargs = prepared_batch.get("added_cond_kwargs", {})

    return trainer.transformer(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]


def discriminator_predict(
        trainer: Trainer,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        ) -> torch.Tensor:
    """Generate predictions using the discriminator"""
    assert trainer.discriminator is not None, "Discriminator must be initialized"
    return trainer.discriminator.forward(
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        latent_image_ids=latent_image_ids,
    )

# LOSS
def calculate_loss(
        trainer: Trainer,
        prepared_batch: dict,
        ):
    phase = trainer.phase
    if phase == Phase.D:
        return calculate_discriminator_loss(trainer=trainer, prepared_batch=prepared_batch)
    elif phase == Phase.G:
        return calculate_generator_loss(trainer=trainer, prepared_batch=prepared_batch)
    else:
        raise ValueError(f"Invalid phase: {phase}")

def calculate_generator_loss(
        trainer: Trainer,
        prepared_batch: dict,
        ):
    """Calculate generator loss combining reconstruction and adversarial components"""
    # Reconstruction loss (e.g., L2 loss)
    # recon_loss = torch.nn.functional.mse_loss(model_pred, target)
    
    # Adversarial loss - fool the discriminator
    fake_pred = discriminator_predict(
        trainer=trainer,
        prepared_batch=prepared_batch,
    )
    adv_loss = -torch.mean(fake_pred)
    
    return adv_loss

def calculate_discriminator_loss(
        trainer: Trainer,
        prepared_batch: dict,
        ):
    """Calculate discriminator loss using hinge loss formulation"""
    # Get real and fake predictions
    assert trainer.discriminator is not None, "Discriminator must be initialized"

    # AC: this seems like it needs to be discriminator prediction of real and generated images
    # should use prepared batch to generate image
    # it's not clear to me what is real vs fake here. seems important in loss.
    # also do i want to use seaweed's R1 approximation?
    
    # AC: the gen_prediction probably has to be converted into something equivalent to a noised image latent since we use flux diffusion network for feature extraction.
    generator_prediction = generator_predict(trainer=trainer, prepared_batch=prepared_batch)

    # we will have to noise the real image latent
    real_item = torch.Tensor(prepared_batch["images"])

    real_pred = trainer.discriminator()
    fake_pred = trainer.discriminator()
    
    # Hinge loss
    real_loss = torch.nn.functional.relu(1.0 - real_pred).mean()
    fake_loss = torch.nn.functional.relu(1.0 + fake_pred).mean()
    
    return real_loss + fake_loss
