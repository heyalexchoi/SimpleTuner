import sys
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
import logging

from enum import Enum
from diffusers.models.transformers import FluxTransformer2DModel  # assuming this is the generator model
from helpers.adversarial.core.network.transformer_D import FluxTransformer2DDiscriminator
from helpers.adversarial.training.helpers import (
    Phase,
    calculate_loss,  # contains calculate_generator_loss & calculate_discriminator_loss helpers with TODOs
)
import os
from helpers.data_backend.factory import random_dataloader_iterator
from helpers.training.state_tracker import StateTracker
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger

from .mixins.protocols import TrainerProtocol

logger = get_logger(
    "SimpleTuner.AdversarialTrainer", log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
)

class AdversarialTrainerMixin(TrainerProtocol):
    """
    Trainer mixin for Flux adversarial training with LyCORIS/LOKR adapter.

    TODO: Insert prediction target generation via get_prediction_target() if needed.
    """

    transformer: FluxTransformer2DModel
    discriminator: FluxTransformer2DDiscriminator
    discriminator_optimizer: Optimizer
    phase: Phase

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = Phase.G

    # [x] hooked load discriminator into init_load_base_model w/ transformer init
    # [x] need to init discriminator optimizer
    # [x] hooked accelerator prepare discriminator and discriminator optimizer in init_prepare_models
    # [x] all under config.use_adversarial_loss

    # [x ] load checkpoint
    # [x ] save checkpoint
    # likely these two are taken care of with accelerator save and load state

    # [ ] discriminator.train()
    # [x ] training_models = [self.discriminator]
    # [ ] set / switch optimizer for step? looks like optimizer step is based on self.optimizer. I could put reference to first transformer in another ivar and switch between the two

    # [x ] need to hook into _get_trainable_parameters()? used in grad norm clip
    # [ ] hook into mark_optimizer_eval and mark_optimizer_train which seems to turn off optimizer for evals?
    # [ ] verify training_models refers to transformer and this includes for lycoris
    # [ ] init_freeze_models freezes the transformer when training lora (requires_grad_(False))
    # [ ] hook into model predict and calculate loss
    # - once we have loss, seems everything else should be the same
    # [ ] I refer to discriminator and generator loss in the logs so i need to set those values
    # [x ] save discriminator weights? there's stage at end where model is unwrapped and saved in diffusers format? not sure if this matters. no it doesn't bc i won't use discriminator in diffusrs pipeline
    def load_discriminator(self, config):
        return FluxTransformer2DDiscriminator(
            transformer=self.transformer,
        )
    
    def determine_adversarial_params_to_optimize(self):
        return [param for param in self.discriminator.parameters() if param.requires_grad]

    def get_training_models(self):
        """
        Returns list of models currently being trained
        Gets passed into accelerator.accumulate()
        """
        if self.phase == Phase.G:
            return [self.transformer]
        else:
            return [self.discriminator]
        
    def get_step_logs(self):
        return {
            "discriminator_loss": self.discriminator_loss,
            "generator_loss": self.generator_loss,
        }
    
    def _get_discriminator_trainable_parameters(self):
        """
        fork from trainer._get_trainable_parameters()
        which seems to correspond to parameters of currently training_models
        """
        return self.discriminator.parameters()
    
    def adversarial_step_will_begin(self):
        pass
    
    def adversarial_step_will_end(self):
        if self.phase == Phase.G:
            self.phase = Phase.D
        else:
            self.phase = Phase.G
