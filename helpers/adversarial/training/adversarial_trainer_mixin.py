from helpers.adversarial.core.network.flux_discriminator import FluxTransformer2DDiscriminator
from helpers.adversarial.core.constants import Phase
from ..core.logging import get_adversarial_logger

from .mixins.protocols import AdversarialTrainerProtocol
from .mixins.loss import AdversarialLossMixin

import torch

# debugging
from torch.profiler import profile, ProfilerActivity

logger = get_adversarial_logger()

class AdversarialTrainerMixin(AdversarialLossMixin, AdversarialTrainerProtocol):
    """
    Trainer mixin for Flux adversarial training with LyCORIS/LOKR adapter.

    Enable with config.use_adversarial_loss = True

    Behavior outside of flux lycoris / lokr is undefined.
    Deepspeed training not implemented.
    LR schedules besides constant are not implemented.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = Phase.G
        self.current_step_loss_components = {}

    def load_discriminator(self, config):
        logger.debug("Loading discriminator")
        return FluxTransformer2DDiscriminator(
            transformer=self.transformer,
            torch_dtype=self.config.weight_dtype,
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
        
    def get_step_logs(self, wandb_logs: dict):
        """
        Returns dictionary of loss components for current step.
        Called immediately before trainer calls accelerator.log(wandb_updates)
        """
        # add phase label to train_loss and grad_absmax
        train_loss = wandb_logs["train_loss"]
        grad_absmax = wandb_logs["grad_absmax"]
        self.current_step_loss_components.update({
            f"{self.phase.value}_train_loss": train_loss,
            f"{self.phase.value}_grad_absmax": grad_absmax,
        })
        logger.info(f"get_step_logs: {self.current_step_loss_components}")
        return self.current_step_loss_components
    
    def _get_discriminator_trainable_parameters(self):
        """
        Returns parameters of discriminator heads, excluding transformer parameters
        """
        # parameters actually return exhaustible generator, not just iterator. wrap in list
        return list(self.discriminator.heads.parameters())
    
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

    def prepare_adversarial_items(self):
        self.discriminator, self.discriminator_optimizer = self.accelerator.prepare(
            self.discriminator,
            self.discriminator_optimizer,
        )

    def freeze_lycoris_parameters(self):
        for param in self.lycoris_wrapped_network.parameters():
            param.requires_grad = False

    def adversarial_training_will_begin(self):
        # keep this reference since we will be swapping self.optimizer each phase
        self.generator_optimizer = self.optimizer
        self.transformer.requires_grad_(False)
        self.discriminator.train()

    def adversarial_step_will_begin(self):
        """
        Sets models to appropriate modes and freezes/unfreezes parameters
        Toggling of lycoris is handled granularly in loss calculation functions
        """
        self.verify_transformer_and_lycoris_parameters_exclusive()
        self.verify_lycoris_identities()
        if self.phase == Phase.G:
            logger.info("adversarial_step_will_begin: Phase G. Freezing discriminator and switching to G optimizer")
            # Set models to appropriate modes
            self.transformer.train()
            self.discriminator.eval()
            # freeze discriminator
            # unfreeze generator lycoris
            # ensure generator lycoris multiplier == 1.0 for generator forward pass
            # but set lycoris multiplier to 0.0 for discriminator forward pass
            self.freeze_discriminator_trainable_parameters()
            self.optimizer = self.generator_optimizer
        else:
            logger.info("adversarial_step_will_begin: Phase D. Freezing generator lycoris and switching to D optimizer")
            # Phase D
            # Set models to appropriate modes
            self.transformer.eval()
            self.discriminator.train()
            # freeze generator lycoris
            # unfreeze discriminator heads
            # generator lycoris should be activated during generator forward pass, then turned off for discriminator forward pass
            self.freeze_lycoris_parameters()
            self.optimizer = self.discriminator_optimizer
        logger.info("Unfreezing trainable parameters")
        # WARNING: is there an issue here with accessing lycoris wrapped network ivar reference vs accelerator attribute to get parameters during training operations?
        for param in self._get_trainable_parameters():
            param.requires_grad = True
        
        # DEBUG
        self.create_profiler()
        self.profiler.start()
        # Check params trainable
        if self.phase == Phase.G:
            # plenty of these params. they are not in transformer. i think.
            # for name, param in self.lycoris_wrapped_network.named_parameters():
            #     logger.info(f"Phase G lycoris_wrapped_network param {name}: requires_grad={param.requires_grad}")
            # Verify in optimizer
            for group in self.optimizer.param_groups:
                lycoris_params = [p for p in group['params'] if any(id(p) == id(param) 
                    for name, param in self.lycoris_wrapped_network.named_parameters() 
                    )]
                logger.info(f"Lycoris params in optimizer group: {len(lycoris_params)}")


    
    def adversarial_step_will_end(self):
        self.verify_transformer_and_lycoris_parameters_exclusive()
        self.current_step_loss_components = {}
        if self.phase == Phase.G:
            self.phase = Phase.D
        else:
            self.phase = Phase.G
        
        # DEBUG
        self.profiler.stop()
        logger.info(f"Snapshot {self.phase.value}:")
        logger.info(self.profiler.key_averages(group_by_stack_n=10).table(sort_by="self_cuda_memory_usage", row_limit=20))
        self.profiler.export_chrome_trace(f"trace_{self.phase.value}.json")
        

    # DEBUG
    def verify_transformer_and_lycoris_parameters_exclusive(self):
        transformer_params = list(self.transformer.parameters())
        lycoris_params = list(self.lycoris_wrapped_network.parameters())
        # Instead of comparing tensors directly, compare their ids
        lycoris_param_ids = {id(p) for p in lycoris_params}
        assert not any(id(p) in lycoris_param_ids for p in transformer_params)
        logger.info("CONFIRMED transformer and lycoris parameters are exclusive")

    def verify_lycoris_identities(self):
        ivar_lycoris_wrapped_network = self.lycoris_wrapped_network
        accelerator_lycoris_wrapped_network = self.accelerator._lycoris_wrapped_network # type: ignore
        assert ivar_lycoris_wrapped_network == accelerator_lycoris_wrapped_network
        logger.info("CONFIRMED ivar_lycoris_wrapped_network == accelerator_lycoris_wrapped_network")
        # compare their params
        ivar_lycoris_param_ids = {id(p) for p in ivar_lycoris_wrapped_network.parameters()}
        accelerator_lycoris_param_ids = {id(p) for p in accelerator_lycoris_wrapped_network.parameters()}
        assert ivar_lycoris_param_ids == accelerator_lycoris_param_ids
        logger.info(f"CONFIRMED ivar_lycoris_param_ids == accelerator_lycoris_param_ids {len(ivar_lycoris_param_ids)} params")

    def create_profiler(self):
        self.profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
              profile_memory=True, record_shapes=True, with_stack=True)
