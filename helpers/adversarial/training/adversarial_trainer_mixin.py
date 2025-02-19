import sys
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
import logging

from enum import Enum
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel  # assuming this is the generator model
from helpers.adversarial.core.network.flux_discriminator import FluxTransformer2DDiscriminator
from helpers.adversarial.core.constants import Phase
import os
from helpers.data_backend.factory import random_dataloader_iterator
from helpers.training.state_tracker import StateTracker
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger

from .mixins.protocols import AdversarialTrainerProtocol
from .mixins.loss import AdversarialLossMixin

logger = get_logger(
    "SimpleTuner.AdversarialTrainer", log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
)

class AdversarialTrainerMixin(AdversarialLossMixin, AdversarialTrainerProtocol):
    """
    Trainer mixin for Flux adversarial training with LyCORIS/LOKR adapter.

    Enable with config.use_adversarial_loss = True

    Behavior outside of flux lycoris / lokr is undefined.
    Deepspeed training not implemented.

    TODO: Insert prediction target generation via get_prediction_target() if needed.
    """

    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = Phase.G
        self.current_step_loss_components = {}


    # [ ] may need to detach the lycoris when using discriminator transformer. looks like they set the multiplier to zero in some cases where they want to deactivate it? then set back to 1 after predict.
    # [ ] hook into model predict, get_prediction_target, and calculate loss
    # - model predict I can probably actually just keep as is. both G and D use G prediction, right? not sure about D phase. is that 50/50 split?
    # - should be able to insert D prediction during G phase
    # - and ... tbd on D phase.
    # [ ] does discriminator phase alternate between using G prediction and just using training data? hows that work
    # - once we have loss, seems everything else should be the same
    # [ ] I refer to discriminator and generator loss in the logs so i need to set those values
    # [ ] make sure config supports my new use_adversarial_loss option
    

    # [x] hooked load discriminator into init_load_base_model w/ transformer init
    # [x] need to init discriminator optimizer
    # [x] hooked accelerator prepare discriminator and discriminator optimizer in init_prepare_models
    # [x] all under config.use_adversarial_loss

    # [x ] load checkpoint
    # [x ] save checkpoint
    # likely these two are taken care of with accelerator save and load state
    # [x ] training_models = [self.discriminator]
    # [x ] save discriminator weights? there's stage at end where model is unwrapped and saved in diffusers format? not sure if this matters. no it doesn't bc i won't use discriminator in diffusrs pipeline

    # [x ] discriminator.train()
    # [x ] seems like self.lycoris_wrapped_network is the lycoris weights. _get_trainable_parameters() refers to lycoris weights when not in D phase.
    # [ x] set / switch optimizer for step? looks like optimizer step is based on self.optimizer. I could put reference to first transformer in another ivar and switch between the two

    # [x ] need to hook into _get_trainable_parameters()? used in grad norm clip
    # [ x] hook into mark_optimizer_eval and mark_optimizer_train which seems to turn off optimizer for evals?
    # [x ] do i need to "move" the discriminator to device w/ dtype? doesnt seem like it. accelerate is configured already and will handle that
    # [x ] verify training_models refers to transformer and this includes for lycoris. seems like once lycoris is init w/ model and called apply_to, it becomes part of model
    # [x ] init_freeze_models freezes the transformer when training lora (requires_grad_(False))
   
    def load_discriminator(self, config):
        return FluxTransformer2DDiscriminator(
            transformer=self.transformer,
        )
    
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
            **self.current_step_loss_components,
        }
    
    def _get_discriminator_trainable_parameters(self):
        """
        Returns parameters of discriminator heads, excluding transformer parameters
        """
        return self.discriminator.heads.parameters()
    
    # these are behind checks for hasattr 'eval' and 'train'
    def mark_adversarial_optimizers_train(self):
        self.generator_optimizer.train() # type: ignore
        self.discriminator_optimizer.train() # type: ignore

    def mark_adversarial_optimizers_eval(self):
        self.generator_optimizer.eval() # type: ignore
        self.discriminator_optimizer.eval() # type: ignore

    def freeze_discriminator_trainable_parameters(self):
        for param in self._get_discriminator_trainable_parameters():
            param.requires_grad = False

    def freeze_lycoris_parameters(self):
        for param in self.lycoris_wrapped_network.parameters():
            param.requires_grad = False

    def adversarial_training_will_begin(self):
        # keep this reference since we will be swapping self.optimizer each phase
        self.generator_optimizer = self.optimizer
        self.transformer.requires_grad_(False)
        self.discriminator.train()

    def adversarial_step_will_begin(self):
        if self.phase == Phase.G:
            self.freeze_discriminator_trainable_parameters()
            self.optimizer = self.generator_optimizer
        else:
            self.freeze_lycoris_parameters()
            self.optimizer = self.discriminator_optimizer
        
        for param in self._get_trainable_parameters():
            param.requires_grad = True
    
    def adversarial_step_will_end(self):
        self.current_step_loss_components = {}
        if self.phase == Phase.G:
            self.phase = Phase.D
        else:
            self.phase = Phase.G
