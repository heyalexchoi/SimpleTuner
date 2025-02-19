import torch
from .protocols import AdversarialTrainerProtocol
from helpers.adversarial.core.constants import Phase
from helpers.models.flux import pack_latents, prepare_latent_image_ids
from helpers.adversarial.core.logging import get_adversarial_logger
from contextlib import contextmanager

logger = get_adversarial_logger()

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

    def convert_model_prediction_to_clean_image_latent(self, prepared_batch: dict, unpacked_generator_predicted_noise: torch.Tensor):
        """
        Converts the unpacked noise prediction into "clean" image latent
        """
        # convert to predicted "clean image" latent by inverting flow matching target
        # target = prepared_batch["noise"] - prepared_batch["latents"]
        predicted_clean_image_latent = prepared_batch["noise"] - unpacked_generator_predicted_noise 
        return predicted_clean_image_latent
    
    def calculate_generator_loss(self, prepared_batch: dict, unpacked_generator_predicted_noise: torch.Tensor):
        """
        Does the following:
        - Takes in *unpacked* latents (to `unpacked_generator_predicted_noise`) from `trainer.model_predict`` corresponding to the generator's prediction
            of noise to remove from batch's noised latents. These latents are unpacked in `trainer.model_predict` in preparation
            for calculating L2 loss.
        - Converts the noise prediction into "clean" image latent
        - Repacks the latent to be processed by Flux transformer discriminator
        - Gets prediction from Flux transformer discriminator. 
            Batch timesteps are passed in with 'clean images' following Seaweed APT paper's use of "ensemble of different timestep values". 
            Rest of batch data including text conditioning is also passed in.
        - Calculates and returns loss for generator

        Returns:
            loss: Single tensor value representing generator loss

        From paper:
            The functions fD , fG , and gG are the output functions. 
            Here, we use the simple non-saturating variant [16]: fD(x) = gG(x) = log σ(x) 
            and fG(x) = log(1 − σ(x)), where σ(x) is the sigmoid function.
                
        """
        # convert to predicted "clean image" latent by inverting flow matching target
        # target = prepared_batch["noise"] - prepared_batch["latents"]
        predicted_clean_image_latent = self.convert_model_prediction_to_clean_image_latent(
            prepared_batch=prepared_batch, 
            unpacked_generator_predicted_noise=unpacked_generator_predicted_noise,
            )

        packed_predicted_clean_image_latent = pack_latents(
                    latents=predicted_clean_image_latent,
                    batch_size=prepared_batch["latents"].shape[0],
                    num_channels_latents=prepared_batch["latents"].shape[1],
                    height=prepared_batch["latents"].shape[2],
                    width=prepared_batch["latents"].shape[3],
                ).to(
                    dtype=self.config.base_weight_dtype,
                    device=self.accelerator.device,
                )

        flux_transformer_kwargs = self.extract_flux_transformer_kwargs(prepared_batch)
       
        with self.temporarily_detach_lycoris():
            discriminator_outputs = self.discriminator.forward(
                **flux_transformer_kwargs,
                hidden_states=packed_predicted_clean_image_latent,
                guidance_scale=self.config.flux_guidance_value,
            )
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(discriminator_outputs)
        
        # Calculate log probabilities
        # We want to maximize log(D(G(z))) which is equivalent to minimizing -log(D(G(z)))
        loss = -torch.mean(torch.log(probs + 1e-8))  # Add small epsilon to prevent log(0)
        
        # Store loss components for logging
        self.current_step_loss_components.update({
            "g_loss": loss.item(),
        })
        
        return loss
    
    def calculate_discriminator_loss(self, prepared_batch: dict, unpacked_generator_predicted_noise: torch.Tensor) -> torch.Tensor:
        """
        Compute the full discriminator loss including adversarial and R1 terms.

        From Seaweed APT:
        We propose an approximated R1 loss, written as:

        ```
        LaR1 = ‖D(x, c) - D(N(x, σI), c)‖²₂
        ```

        Specifically, we perturb the real data with Gaussian noise of small variance σ. 
        The loss encourages the discriminator's predictions to be close between the real data 
        and its perturbation, thereby reducing the discriminator gradient on real data and 
        achieving a consistent objective as the original R1 regularization. 
        Therefore, the final discriminator loss LD is defined as:

        ```
        LD = E[fD(D(x, c))] + E[fG(D(G(z, c), c))] + λ E[‖D(x, c) - D(N(x, σI), c)‖²₂]
            x,c∼T        z∼N               x,c∼T
                        c∼T
        ```

        In our experiments, we use λ = 100, σ = 0.01 for images and σ = 0.1 for videos. 
        The generator and the discriminator are optimized in alternating steps, 
        in which the approximated R1 is applied on every discriminator step.
        """
        # lambda_r1 (float): Coefficient for the approximated R1 regularization
        # sigma (float): Standard deviation for Gaussian noise in approximated R1
        lambda_r1 = 100.0
        sigma = 0.01

        predicted_clean_image_latent = self.convert_model_prediction_to_clean_image_latent(
            prepared_batch=prepared_batch, 
            unpacked_generator_predicted_noise=unpacked_generator_predicted_noise,
            )

        # pack generator prediction latents for flux transformer
        fake_samples = pack_latents(
                    latents=predicted_clean_image_latent,
                    batch_size=prepared_batch["latents"].shape[0],
                    num_channels_latents=prepared_batch["latents"].shape[1],
                    height=prepared_batch["latents"].shape[2],
                    width=prepared_batch["latents"].shape[3],
                )
        # pack real image latents for flux transformer
        real_samples = pack_latents(
            latents=prepared_batch["latents"],
            batch_size=prepared_batch["latents"].shape[0],
            num_channels_latents=prepared_batch["latents"].shape[1],
            height=prepared_batch["latents"].shape[2],
            width=prepared_batch["latents"].shape[3],
        )
        
        extracted_flux_transformer_kwargs = self.extract_flux_transformer_kwargs(prepared_batch)
        guidance_scale = self.config.flux_guidance_value

        with self.temporarily_detach_lycoris():
            # Get discriminator predictions for real image and fake generated image
            disc_real = self.discriminator(
                **extracted_flux_transformer_kwargs,
                hidden_states=real_samples,
                guidance_scale=guidance_scale,
            )
            disc_fake = self.discriminator(
                **extracted_flux_transformer_kwargs,
                hidden_states=fake_samples,
                guidance_scale=guidance_scale,
            )
        # Real loss: want D(real) -> 1
        real_loss = -torch.mean(torch.log(torch.sigmoid(disc_real) + 1e-8))
        
        # Fake loss: want D(fake) -> 0 
        fake_loss = -torch.mean(torch.log(1 - torch.sigmoid(disc_fake) + 1e-8))
        
        # R1 regularization approximation from seaweed APT
        noise = torch.randn_like(real_samples) * sigma
        noised_samples = real_samples + noise
        with torch.no_grad(), self.temporarily_detach_lycoris():
            noised_outputs = self.discriminator(
                **extracted_flux_transformer_kwargs,
                hidden_states=noised_samples,
                guidance_scale=guidance_scale,
                )
        r1_penalty = torch.mean((disc_real - noised_outputs) ** 2)
        
        total_loss = real_loss + fake_loss + lambda_r1 * r1_penalty

        # Store loss components for logging
        self.current_step_loss_components.update({
            "d_real_loss": real_loss.item(),
            "d_fake_loss": fake_loss.item(),
            "d_r1_penalty": r1_penalty.item(),
            "d_total_loss": total_loss.item()
        })
        
        return total_loss


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
    
    @contextmanager
    def temporarily_detach_lycoris(self):
        """
        Context manager for temporarily detaching LyCORIS adapter during operations.
        The discriminator and generator share the same frozen flux transformer,
        with the LyCORIS adapter serving as the generator-specific weights.
        We want to detach the LyCORIS adapter during the discriminator forward pass.
        """
        if hasattr(self.accelerator, '_lycoris_wrapped_network'):
            logger.debug("Detaching LyCORIS adapter.")
            self.accelerator._lycoris_wrapped_network.set_multiplier(0.0)  # type: ignore
        
        try:
            yield
        finally:
            if hasattr(self.accelerator, '_lycoris_wrapped_network'):
                logger.debug("Re-attaching LyCORIS adapter.")
                self.accelerator._lycoris_wrapped_network.set_multiplier(1.0)  # type: ignore