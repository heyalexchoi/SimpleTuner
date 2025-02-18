import torch
from .protocols import AdversarialTrainerProtocol
from helpers.adversarial.training.helpers import Phase
from helpers.models.flux import pack_latents, prepare_latent_image_ids

class AdversarialLossMixin(AdversarialTrainerProtocol):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_adversarial_loss(self, model_pred, prepared_batch) -> torch.Tensor:
        if self.phase == Phase.G:
            return self.calculate_generator_loss(model_pred, prepared_batch)
        elif self.phase == Phase.D:
            return self.calculate_discriminator_loss(model_pred, prepared_batch)
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

    
    def calculate_generator_loss(self, prepared_batch, generator_prediction):
        """
        Makes discriminator prediction from generator prediction then calculates loss for generator
        
        From paper:
            The functions fD , fG , and gG are the output functions. 
            Here, we use the simple non-saturating variant [16]: fD(x) = gG(x) = log σ(x) 
            and fG(x) = log(1 − σ(x)), where σ(x) is the sigmoid function.

        need generator prediction
        turn into image latent

        give that to discriminator with timesteps, conditioning, etc (prepared_batch)
        get discriminator prediction

        apply sigmoid and calculate log probabilities

        
        Args:
            discriminator_outputs: Tensor of shape (batch_size, 1) 
                                 containing discriminator predictions on generated samples
        
        Returns:
            loss: Single tensor value representing generator loss
        """
        # copied from trainer.model_predict
        # based on flux w/ guidance mode constant
        
        # remember that flux transformer prefers packed latents
        # this includes flux discriminator
        # probably want to pack the real images too
        
        packed_generator_prediction = pack_latents(
                    latents=generator_prediction,
                    batch_size=prepared_batch["latents"].shape[0],
                    num_channels_latents=prepared_batch["latents"].shape[1],
                    height=prepared_batch["latents"].shape[2],
                    width=prepared_batch["latents"].shape[3],
                ).to(
                    dtype=self.config.base_weight_dtype,
                    device=self.accelerator.device,
                )

        flux_transformer_kwargs = self.extract_flux_transformer_kwargs(prepared_batch)
       
        discriminator_outputs = self.discriminator.forward(
            **flux_transformer_kwargs,
            hidden_states=packed_generator_prediction,
            guidance_scale=self.config.flux_guidance_value,
        )
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(discriminator_outputs)
        
        # Calculate log probabilities
        # We want to maximize log(D(G(z))) which is equivalent to minimizing -log(D(G(z)))
        loss = -torch.mean(torch.log(probs + 1e-8))  # Add small epsilon to prevent log(0)
        
        return loss
    
    def calculate_discriminator_loss(self, discriminator_outputs, prepared_batch) -> torch.Tensor:
        return torch.tensor(0.0)


    def extract_flux_transformer_kwargs(self, prepared_batch):
        timesteps = prepared_batch.get("timesteps")
        noisy_latents = prepared_batch.get("noisy_latents")

        img_ids = prepare_latent_image_ids(
                    prepared_batch["latents"].shape[0],
                    prepared_batch["latents"].shape[2],
                    prepared_batch["latents"].shape[3],
                    self.accelerator.device,
                    self.config.weight_dtype,
                )
        
        timesteps = (
            torch.tensor(timesteps)
            .expand(noisy_latents.shape[0])
            .to(device=self.accelerator.device)
            / 1000
        )

        text_ids = torch.zeros(
            prepared_batch["prompt_embeds"].shape[1],
            3,
        ).to(
            device=self.accelerator.device,
            dtype=self.config.base_weight_dtype,
        )

        pooled_projections = prepared_batch.get('add_text_embeds')
        # confusingly this is set twice in model_predict, but this is the final value passed to flux transformer:
        encoder_hidden_states = prepared_batch.get('prompt_embeds')

        return {
            "noisy_latents": noisy_latents,
            "text_ids": text_ids,
            "img_ids": img_ids,
            "timesteps": timesteps,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
        }