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
from ..core.logging import get_adversarial_logger

from .mixins.protocols import AdversarialTrainerProtocol
from .mixins.loss import AdversarialLossMixin

logger = get_adversarial_logger()

class AdversarialTrainerMixin(AdversarialLossMixin, AdversarialTrainerProtocol):
    """
    Trainer mixin for Flux adversarial training with LyCORIS/LOKR adapter.

    Enable with config.use_adversarial_loss = True

    Behavior outside of flux lycoris / lokr is undefined.
    Deepspeed training not implemented.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = Phase.G
        self.current_step_loss_components = {}

    def load_discriminator(self, config):
        logger.debug("Loading discriminator")
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
        logger.debug(f"get_step_logs: {self.current_step_loss_components}")
        return self.current_step_loss_components
    
    def _get_discriminator_trainable_parameters(self):
        """
        Returns parameters of discriminator heads, excluding transformer parameters
        """
        #
        for name, param in self.discriminator.heads.named_parameters():
            logger.debug(f"_get_discriminator_trainable_parameters: {name}")
        #
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
        #
        for name, param in self.lycoris_wrapped_network.named_parameters():
            logger.debug(f"lycoris_wrapped_network named parameters: {name}")
        #
        for param in self.lycoris_wrapped_network.parameters():
            param.requires_grad = False

    def adversarial_training_will_begin(self):
        # keep this reference since we will be swapping self.optimizer each phase
        self.generator_optimizer = self.optimizer
        self.transformer.requires_grad_(False)
        self.discriminator.train()

    def adversarial_step_will_begin(self):
        if self.phase == Phase.G:
            logger.debug("adversarial_step_will_begin: Phase G. Freezing discriminator and switching to G optimizer")
            # freeze discriminator
            # unfreeze generator lycoris
            # ensure generator lycoris multiplier == 1.0 for generator forward pass
            # but set lycoris multiplier to 0.0 for discriminator forward pass
            self.freeze_discriminator_trainable_parameters()
            self.optimizer = self.generator_optimizer
        else:
            logger.debug("adversarial_step_will_begin: Phase D. Freezing generator lycoris and switching to D optimizer")
            # Phase D
            # freeze generator lycoris
            # unfreeze discriminator heads
            # generator lycoris should be activated during generator forward pass, then turned off for discriminator forward pass
            self.freeze_lycoris_parameters()
            self.optimizer = self.discriminator_optimizer
        logger.debug("Unfreezing trainable parameters")
        # WARNING: is there an issue here with accessing lycoris wrapped network ivar reference vs accelerator attribute to get parameters during training operations?
        for param in self._get_trainable_parameters():
            param.requires_grad = True
    
    def adversarial_step_will_end(self):
        self.current_step_loss_components = {}
        if self.phase == Phase.G:
            self.phase = Phase.D
        else:
            self.phase = Phase.G
