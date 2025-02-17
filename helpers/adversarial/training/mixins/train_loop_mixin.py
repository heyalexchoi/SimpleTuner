import logging
import torch
from accelerate.logging import get_logger
from ..helpers import Phase
from .base import TrainerMixinBase
from typing import TypeVar
from .protocols import TrainerProtocol

logger = get_logger(__name__)

T = TypeVar('T', bound=TrainerProtocol)

class TrainLoopMixin(TrainerMixinBase):
    """Mixin class that handles the core training loop logic"""
    
    def __init__(self):
        required_attrs = [
            'accelerator', 'config', 'state', 'transformer',
            'discriminator', 'optimizer', 'train_loss', 'phase',
            'timesteps_buffer'
        ]
        missing = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing:
            raise TypeError(
                f"{self.__class__.__name__} requires {missing} attributes"
            )

    def _train_loop_step(self: T, step, prepared_batch, training_luminance_values):
        """Execute a single training step"""
        if self.phase == Phase.G:
            training_models = [self.transformer]
        else:
            training_models = [self.discriminator]

        with self.accelerator.accumulate(training_models):
            bsz = prepared_batch["latents"].shape[0]
            training_logger.debug("Sending latent batch to GPU.")

            if int(bsz) != int(self.config.train_batch_size):
                logger.error(
                    f"Received {bsz} latents, but expected {self.config.train_batch_size}. Processing short batch."
                )
            training_logger.debug(f"Working on batch size: {bsz}")
            
            # Store timesteps for visualization
            for timestep in prepared_batch["timesteps"].tolist():
                self.timesteps_buffer.append(
                    (self.state["global_step"], timestep)
                )

            encoder_hidden_states = prepared_batch["encoder_hidden_states"]
            training_logger.debug(
                f"Encoder hidden states: {encoder_hidden_states.shape}"
            )

            add_text_embeds = prepared_batch["add_text_embeds"]
            training_logger.debug(
                f"Pooled embeds: {add_text_embeds.shape if add_text_embeds is not None else None}"
            )
            
            # Get prediction target and compute loss
            target = self.get_prediction_target(prepared_batch)
            model_pred = self.model_predict(prepared_batch=prepared_batch)
            loss = self._calculate_loss(
                prepared_batch, model_pred, target, apply_conditioning_mask=True
            )

            # Gather losses and compute average
            avg_loss = self.accelerator.gather(
                loss.repeat(self.config.train_batch_size)
            ).mean()
            self.train_loss += (
                avg_loss.item() / self.config.gradient_accumulation_steps
            )

            # Handle backpropagation and optimization
            self._handle_backward_step(loss)

        return loss, avg_loss, training_luminance_values

    def _handle_backward_step(self, loss):
        """Handle backpropagation and gradient updates"""
        self.grad_norm = None
        if not self.config.disable_accelerator:
            training_logger.debug("Backwards pass.")
            self.accelerator.backward(loss)

            if (
                self.config.optimizer != "adam_bfloat16"
                and self.config.gradient_precision == "fp32"
            ):
                # Convert gradients to fp32 for stable accumulation
                for param in self.params_to_optimize:
                    if param.grad is not None:
                        param.grad.data = param.grad.data.to(torch.float32)

            self.grad_norm = self._max_grad_value()
            if (
                self.accelerator.sync_gradients
                and self.config.optimizer not in ["optimi-stableadamw", "prodigy"]
                and self.config.max_grad_norm > 0
            ):
                if self.config.grad_clip_method == "norm":
                    self.grad_norm = self.accelerator.clip_grad_norm_(
                        self._get_trainable_parameters(),
                        self.config.max_grad_norm,
                    )
                elif self.config.use_deepspeed_optimizer:
                    pass  # deepspeed handles norm clipping internally
                elif self.config.grad_clip_method == "value":
                    self.accelerator.clip_grad_value_(
                        self._get_trainable_parameters(),
                        self.config.max_grad_norm,
                    )
                else:
                    raise ValueError(
                        f"Unknown grad clip method: {self.config.grad_clip_method}. Supported methods: value, norm"
                    )

            self._handle_optimizer_step()

    def _handle_optimizer_step(self):
        """Handle optimizer stepping and gradient release"""
        if self.config.optimizer_release_gradients:
            step_offset = 0  # simpletuner indexes steps from 1
            should_not_release_gradients = (
                self.state["global_step"] + step_offset
            ) % self.config.gradient_accumulation_steps != 0
            training_logger.debug(
                f"step: {self.state['global_step']}, should_not_release_gradients: {should_not_release_gradients}"
            )
            self.optimizer.optimizer_accumulation = should_not_release_gradients
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=self.config.set_grads_to_none) 