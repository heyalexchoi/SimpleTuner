import sys
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
import logging

from enum import Enum
from diffusers.models.transformers import FluxTransformer2DModel  # assuming this is the generator model
from helpers.adversarial.core.network.transformer_D import FluxTransformer2DDiscriminator
from helpers.training.trainer import Trainer
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

logger = get_logger(
    "SimpleTuner", log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
)

filelock_logger = get_logger("filelock")
connection_logger = get_logger("urllib3.connectionpool")
training_logger = get_logger("training-loop")

# More important logs.
target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)
training_logger_level = os.environ.get("SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL", "INFO")
training_logger.setLevel(training_logger_level)

# Less important logs.
filelock_logger.setLevel("WARNING")
connection_logger.setLevel("WARNING")

class AdversarialTrainer(Trainer):
    """
    Trainer subclass for Flux adversarial training with LyCORIS/LOKR adapter.
    This subclass removes extra non-Flux, controlnet, EMA, and standard LoRA logic,
    retaining feature extraction methods (e.g. VAE & text encoders), checkpointing,
    data backend handling, multi-GPU support via Accelerate and adversarial loss computation.
    
    NOTE: Regularisation data handling (e.g. detaching/re-attaching adapters) has been removed.
    TODO: Insert prediction target generation via get_prediction_target() if needed.
    """

    accelerator: Accelerator
    transformer: FluxTransformer2DModel
    optimizer: Optimizer
    discriminator: FluxTransformer2DDiscriminator
    discriminator_optimizer: Optimizer
    phase: Phase

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_adversarial_components()

    def init_prepare_models(self, lr_scheduler):
        """
        Prepare generator (Flux transformer) for adversarial training.
        This method strips out controlnet and unet logic and only loads the transformer.
        It also loads the dataloader(s) from the data backend.
        """
        logger.info("Preparing models for adversarial training...")

        # Setup train dataloaders from data backends
        self.train_dataloaders = []
        from helpers.training.state_tracker import StateTracker  # assumed to exist
        for _, backend in StateTracker.get_data_backends().items():
            if "train_dataloader" in backend:
                self.train_dataloaders.append(backend["train_dataloader"])
                break
        if len(self.train_dataloaders) == 0:
            logger.error("No dataloaders were configured.")
            sys.exit(0)

        if self.config.disable_accelerator:
            logger.warning("Accelerator disabled. Skipping model preparation.")
            return

        logger.info("Initializing accelerator and moving weights to GPU...")
        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False

        # We always use the transformer for Flux (generator)
        primary_model = self.transformer
        # We deliberately skip any controlnet or unet branches.
        results = self.accelerator.prepare(
            primary_model, lr_scheduler, self.optimizer, self.train_dataloaders[0]
        )
        self.transformer = results[0]
        self.lr_scheduler = results[1]
        self.optimizer = results[2]
        # The rest of the entries are dataloaders:
        self.train_dataloaders = [results[3:]]
        logger.info("Model preparation complete.")

    def resume_and_prepare(self):
        """
        Initialize optimizer, lr scheduler, hooks, prepare models, resume checkpoint and do post-load freezing.
        This has been modified to eventually handle checkpoint states for both generator and discriminator.
        """
        self.init_optimizer()
        lr_scheduler = self.init_lr_scheduler()
        self.init_hooks()
        self.init_prepare_models(lr_scheduler=lr_scheduler)
        # Adjust lr_scheduler state with checkpoint if available.
        lr_scheduler = self.init_resume_checkpoint(lr_scheduler=lr_scheduler)
        self.init_post_load_freeze()

    def init_adversarial_components(self):
        """
        Initialize discriminator and its optimizer for adversarial training.
        """
        logger.info("Initializing adversarial training components...")
        # Initialize discriminator (Flux transformer based discriminator)
        self.discriminator = FluxTransformer2DDiscriminator(
            config=self.config,
            # TODO: Add any additional parameters if needed
        )
        self.discriminator.to(
            device=self.accelerator.device,
            dtype=self.config.weight_dtype
        )
        self.discriminator_optimizer = AdamW(
            self.discriminator.parameters(),
            lr=self.config.discriminator_learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )
        # Prepare discriminator with accelerator
        self.discriminator, self.discriminator_optimizer = self.accelerator.prepare(
            self.discriminator, self.discriminator_optimizer
        )

    def train(self):
        self.init_trackers()
        self._train_initial_msg()
        self.mark_optimizer_train()

        self.transformer.train()
        self.discriminator.train()

        # Only show the progress bar once on each machine.
        show_progress_bar = True
        if not self.accelerator.is_local_main_process:
            show_progress_bar = False
        progress_bar = tqdm(
            range(0, self.config.max_train_steps),
            disable=not show_progress_bar,
            initial=self.state["global_step"],
            desc=f"Epoch {self.state['first_epoch']}/{self.config.num_train_epochs} Steps",
            ncols=125,
        )
        self.accelerator.wait_for_everyone()

        # Some values that are required to be initialised later.
        step = self.state["global_step"]
        training_luminance_values = []
        current_epoch_step = None
        self.bf, fetch_thread = None, None
        iterator_fn = random_dataloader_iterator
        num_epochs_to_track = self.config.num_train_epochs + 1
        if self.config.ignore_final_epochs:
            num_epochs_to_track += 1000000
        for epoch in range(self.state["first_epoch"], num_epochs_to_track):
            if (
                self.state["current_epoch"] > self.config.num_train_epochs + 1
                and not self.config.ignore_final_epochs
            ):
                # This might immediately end training, but that's useful for simply exporting the model.
                logger.info(
                    f"Training run is complete ({self.config.num_train_epochs}/{self.config.num_train_epochs} epochs, {self.state['global_step']}/{self.config.max_train_steps} steps)."
                )
                break
            self._epoch_rollover(epoch)

            # removed controlnet and unet logic
            # removed standard lora and train text encoder logic

            if current_epoch_step is not None:
                # We are resetting to the next epoch, if it is not none.
                current_epoch_step = 0
            else:
                # If it's None, we need to calculate the current epoch step based on the current global step.
                current_epoch_step = (
                    self.state["global_step"] % self.config.num_update_steps_per_epoch
                )
            train_backends = {}
            for backend_id, backend in StateTracker.get_data_backends().items():
                if (
                    StateTracker.backend_status(backend_id)
                    or "train_dataloader" not in backend
                ):
                    # Exclude exhausted backends.
                    logger.debug(
                        f"Excluding backend: {backend_id}, as it is exhausted? {StateTracker.backend_status(backend_id)} or not found {('train_dataloader' not in backend)}"
                    )
                    continue
                train_backends[backend_id] = backend["train_dataloader"]
            
            iterator_args = [train_backends]
            # remove dataloader prefetch stuff

            # --- Regularisation handling removed ---
            # Previously, there was special handling for regularisation data (e.g. temporarily detaching
            # the adapter parameters) which is no longer required for adversarial training.

            # TODO: If needed in the future, insert prediction target calculation:
            # target = self.get_prediction_target(prepared_batch)

            # TODO remember each phase setup
            while True:
                self._exit_on_signal()
                step += 1
                prepared_batch = self.prepare_batch(iterator_fn(step, *iterator_args))
                
                training_logger.debug(f"Iterator: {iterator_fn}")
                if self.config.lr_scheduler == "cosine_with_restarts":
                    self.extra_lr_scheduler_kwargs["step"] = self.state["global_step"]

                if self.accelerator.is_main_process:
                    progress_bar.set_description(
                        f"Epoch {self.state['current_epoch']}/{self.config.num_train_epochs}, Steps"
                    )

                # If we receive a False from the enumerator, we know we reached the next epoch.
                if prepared_batch is False:
                    logger.debug(f"Reached the end of epoch {epoch}")
                    break

                if prepared_batch is None:
                    import traceback

                    raise ValueError(
                        f"Received a None batch, which is not a good thing. Traceback: {traceback.format_exc()}"
                    )
                
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
                    # Prepare the data for the scatter plot
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
                    # Get the target for loss depending on the prediction type
                    target = self.get_prediction_target(prepared_batch)

                    added_cond_kwargs = prepared_batch.get("added_cond_kwargs")

                    # Predict the noise residual and compute loss
                    
                    # removed special treatment of regularization data

                    training_logger.debug("Predicting noise residual.")
                    model_pred = self.model_predict(
                        prepared_batch=prepared_batch,
                    )
                    loss = self._calculate_loss(
                        prepared_batch, model_pred, target, apply_conditioning_mask=True
                    )

                    # removed regularisation loss handling

                    # Gather the losses across all processes for logging (if using distributed training)
                    avg_loss = self.accelerator.gather(
                        loss.repeat(self.config.train_batch_size)
                    ).mean()
                    self.train_loss += (
                        avg_loss.item() / self.config.gradient_accumulation_steps
                    )
                    # Backpropagate
                    self.grad_norm = None
                    if not self.config.disable_accelerator:
                        training_logger.debug("Backwards pass.")
                        self.accelerator.backward(loss)

                        if (
                            self.config.optimizer != "adam_bfloat16"
                            and self.config.gradient_precision == "fp32"
                        ):
                            # After backward, convert gradients to fp32 for stable accumulation
                            for param in self.params_to_optimize:
                                if param.grad is not None:
                                    param.grad.data = param.grad.data.to(torch.float32)

                        self.grad_norm = self._max_grad_value()
                        if (
                            self.accelerator.sync_gradients
                            and self.config.optimizer
                            not in ["optimi-stableadamw", "prodigy"]
                            and self.config.max_grad_norm > 0
                        ):
                            # StableAdamW/Prodigy do not need clipping, similar to Adafactor.
                            if self.config.grad_clip_method == "norm":
                                self.grad_norm = self.accelerator.clip_grad_norm_(
                                    self._get_trainable_parameters(),
                                    self.config.max_grad_norm,
                                )
                            elif self.config.use_deepspeed_optimizer:
                                # deepspeed can only do norm clipping (internally)
                                pass
                            elif self.config.grad_clip_method == "value":
                                self.accelerator.clip_grad_value_(
                                    self._get_trainable_parameters(),
                                    self.config.max_grad_norm,
                                )
                            else:
                                raise ValueError(
                                    f"Unknown grad clip method: {self.config.grad_clip_method}. Supported methods: value, norm"
                                )
                        training_logger.debug("Stepping components forward.")
                        if self.config.optimizer_release_gradients:
                            step_offset = 0  # simpletuner indexes steps from 1.
                            should_not_release_gradients = (
                                step + step_offset
                            ) % self.config.gradient_accumulation_steps != 0
                            training_logger.debug(
                                f"step: {step}, should_not_release_gradients: {should_not_release_gradients}, self.config.optimizer_release_gradients: {self.config.optimizer_release_gradients}"
                            )
                            self.optimizer.optimizer_accumulation = (
                                should_not_release_gradients
                            )
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad(
                            set_to_none=self.config.set_grads_to_none
                        )

                # Checks if the accelerator has performed an optimization step behind the scenes
                wandb_logs = {}
                if self.accelerator.sync_gradients:
                    try:
                        if "prodigy" in self.config.optimizer:
                            self.lr_scheduler.step(**self.extra_lr_scheduler_kwargs)
                            self.lr = self.optimizer.param_groups[0]["d"]
                        elif self.config.is_lr_scheduler_disabled:
                            # hackjob method of retrieving LR from accelerated optims
                            self.lr = StateTracker.get_last_lr()
                        else:
                            self.lr_scheduler.step(**self.extra_lr_scheduler_kwargs)
                            self.lr = self.lr_scheduler.get_last_lr()[0]
                    except Exception as e:
                        logger.error(
                            f"Failed to get the last learning rate from the scheduler. Error: {e}"
                        )
                    wandb_logs.update(
                        {
                            "train_loss": self.train_loss,
                            "optimization_loss": loss,
                            "learning_rate": self.lr,
                            "epoch": epoch,
                        }
                    )
                    if parent_loss is not None:
                        wandb_logs["regularisation_loss"] = parent_loss
                    if self.config.model_family == "flux" and self.guidance_values_list:
                        # avg the values
                        guidance_values = torch.tensor(self.guidance_values_list).mean()
                        wandb_logs["mean_cfg"] = guidance_values.item()
                        self.guidance_values_list = []
                    if self.grad_norm is not None:
                        if self.config.grad_clip_method == "norm":
                            wandb_logs["grad_norm"] = self.grad_norm
                        else:
                            wandb_logs["grad_absmax"] = self.grad_norm
                    if self.validation is not None and hasattr(
                        self.validation, "evaluation_result"
                    ):
                        eval_result = self.validation.get_eval_result()
                        if eval_result is not None and type(eval_result) == dict:
                            # add the dict to wandb_logs
                            self.validation.clear_eval_result()
                            wandb_logs.update(eval_result)

                    progress_bar.update(1)
                    self.state["global_step"] += 1
                    current_epoch_step += 1
                    StateTracker.set_global_step(self.state["global_step"])

                    ema_decay_value = "None (EMA not in use)"
                    if self.config.use_ema:
                        if self.ema_model is not None:
                            self.ema_model.step(
                                parameters=self._get_trainable_parameters(),
                                global_step=self.state["global_step"],
                            )
                            wandb_logs["ema_decay_value"] = self.ema_model.get_decay()
                            ema_decay_value = wandb_logs["ema_decay_value"]
                        self.accelerator.wait_for_everyone()

                    # Log scatter plot to wandb
                    if (
                        self.config.report_to == "wandb"
                        and self.accelerator.is_main_process
                    ):
                        # Prepare the data for the scatter plot
                        data = [
                            [iteration, timestep]
                            for iteration, timestep in self.timesteps_buffer
                        ]
                        table = wandb.Table(
                            data=data, columns=["global_step", "timestep"]
                        )
                        wandb_logs["timesteps_scatter"] = wandb.plot.scatter(
                            table,
                            "global_step",
                            "timestep",
                            title="Timestep distribution by step",
                        )

                    # Clear buffers
                    self.timesteps_buffer = []

                    # Average out the luminance values of each batch, so that we can store that in this step.
                    avg_training_data_luminance = sum(training_luminance_values) / len(
                        training_luminance_values
                    )
                    wandb_logs["train_luminance"] = avg_training_data_luminance

                    logger.debug(
                        f"Step {self.state['global_step']} of {self.config.max_train_steps}: loss {loss.item()}, lr {self.lr}, epoch {epoch}/{self.config.num_train_epochs}, ema_decay_value {ema_decay_value}, train_loss {self.train_loss}"
                    )
                    webhook_pending_msg = f"Step {self.state['global_step']} of {self.config.max_train_steps}: loss {round(loss.item(), 4)}, lr {self.lr}, epoch {epoch}/{self.config.num_train_epochs}, ema_decay_value {ema_decay_value}, train_loss {round(self.train_loss, 4)}"

                    # Reset some values for the next go.
                    training_luminance_values = []
                    self.train_loss = 0.0

                    if (
                        self.config.webhook_reporting_interval is not None
                        and self.state["global_step"]
                        % self.config.webhook_reporting_interval
                        == 0
                    ):
                        structured_data = {
                            "state": self.state,
                            "loss": round(self.train_loss, 4),
                            "parent_loss": parent_loss,
                            "learning_rate": self.lr,
                            "epoch": epoch,
                            "final_epoch": self.config.num_train_epochs,
                        }
                        self._send_webhook_raw(
                            structured_data=structured_data, message_type="train"
                        )
                    if self.state["global_step"] % self.config.checkpointing_steps == 0:
                        self._send_webhook_msg(
                            message=f"Checkpoint: `{webhook_pending_msg}`",
                            message_level="info",
                        )
                        if self.accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if self.config.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(self.config.output_dir)
                                checkpoints = [
                                    d for d in checkpoints if d.startswith("checkpoint")
                                ]
                                checkpoints = sorted(
                                    checkpoints, key=lambda x: int(x.split("-")[1])
                                )

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if (
                                    len(checkpoints)
                                    >= self.config.checkpoints_total_limit
                                ):
                                    num_to_remove = (
                                        len(checkpoints)
                                        - self.config.checkpoints_total_limit
                                        + 1
                                    )
                                    removing_checkpoints = checkpoints[0:num_to_remove]
                                    logger.debug(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.debug(
                                        f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                    )

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(
                                            self.config.output_dir, removing_checkpoint
                                        )
                                        try:
                                            shutil.rmtree(
                                                removing_checkpoint, ignore_errors=True
                                            )
                                        except Exception as e:
                                            logger.error(
                                                f"Failed to remove directory: {removing_checkpoint}"
                                            )
                                            print(e)

                        if (
                            self.accelerator.is_main_process
                            or self.config.use_deepspeed_optimizer
                        ):
                            save_path = os.path.join(
                                self.config.output_dir,
                                f"checkpoint-{self.state['global_step']}",
                            )
                            print("\n")
                            # schedulefree optim needs the optimizer to be in eval mode to save the state (and then back to train after)
                            self.mark_optimizer_eval()
                            self.accelerator.save_state(save_path)
                            self.mark_optimizer_train()
                            for _, backend in StateTracker.get_data_backends().items():
                                if "sampler" in backend:
                                    logger.debug(f"Backend: {backend}")
                                    backend["sampler"].save_state(
                                        state_path=os.path.join(
                                            save_path,
                                            self.model_hooks.training_state_path,
                                        ),
                                    )

                    if (
                        self.config.accelerator_cache_clear_interval is not None
                        and self.state["global_step"]
                        % self.config.accelerator_cache_clear_interval
                        == 0
                    ):
                        reclaim_memory()

                    # here we might run eval loss calculations.
                    if self.evaluation is not None and self.evaluation.would_evaluate(
                        self.state
                    ):
                        self.mark_optimizer_eval()
                        all_accumulated_losses = self.evaluation.execute_eval(
                            prepare_batch=self.prepare_batch,
                            model_predict=self.model_predict,
                            calculate_loss=self._calculate_loss,
                            get_prediction_target=self.get_prediction_target,
                            noise_scheduler=self._get_noise_scheduler(),
                        )
                        tracker_table = self.evaluation.generate_tracker_table(
                            all_accumulated_losses=all_accumulated_losses
                        )
                        print(f"Tracking information: {tracker_table}")
                        wandb_logs.update(tracker_table)
                        self.mark_optimizer_train()

                    self.accelerator.log(
                        wandb_logs,
                        step=self.state["global_step"],
                    )

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": float(self.lr),
                }
                if self.grad_norm is not None:
                    if self.config.grad_clip_method == "norm":
                        logs["grad_norm"] = float(self.grad_norm.clone().detach())
                    elif self.config.grad_clip_method == "value":
                        logs["grad_absmax"] = self.grad_norm

                progress_bar.set_postfix(**logs)

                if self.validation is not None:
                    if self.validation.would_validate():
                        self.mark_optimizer_eval()
                        self.enable_sageattention_inference()
                        self.disable_gradient_checkpointing()
                    self.validation.run_validations(
                        validation_type="intermediary", step=step
                    )
                    if self.validation.would_validate():
                        self.disable_sageattention_inference()
                        self.enable_gradient_checkpointing()
                        self.mark_optimizer_train()
                if (
                    self.config.push_to_hub
                    and self.config.push_checkpoints_to_hub
                    and self.state["global_step"] % self.config.checkpointing_steps == 0
                    and step % self.config.gradient_accumulation_steps == 0
                    and self.state["global_step"] > self.state["global_resume_step"]
                ):
                    if self.accelerator.is_main_process:
                        try:
                            self.hub_manager.upload_latest_checkpoint(
                                validation_images=(
                                    getattr(self.validation, "validation_images")
                                    if self.validation is not None
                                    else None
                                ),
                                webhook_handler=self.webhook_handler,
                            )
                        except Exception as e:
                            logger.error(
                                f"Error uploading to hub: {e}, continuing training."
                            )
                self.accelerator.wait_for_everyone()

                if self.state["global_step"] >= self.config.max_train_steps or (
                    epoch > self.config.num_train_epochs
                    and not self.config.ignore_final_epochs
                ):
                    logger.info(
                        f"Training has completed."
                        f"\n -> global_step = {self.state['global_step']}, max_train_steps = {self.config.max_train_steps}, epoch = {epoch}, num_train_epochs = {self.config.num_train_epochs}",
                    )
                    break
            if self.state["global_step"] >= self.config.max_train_steps or (
                epoch > self.config.num_train_epochs
                and not self.config.ignore_final_epochs
            ):
                logger.info(
                    f"Exiting training loop. Beginning model unwind at epoch {epoch}, step {self.state['global_step']}"
                )
                break

        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        validation_images = None
        if self.accelerator.is_main_process:
            self.mark_optimizer_eval()
            if self.validation is not None:
                self.enable_sageattention_inference()
                self.disable_gradient_checkpointing()
                validation_images = self.validation.run_validations(
                    validation_type="final",
                    step=self.state["global_step"],
                    force_evaluation=True,
                    skip_execution=True,
                ).validation_images
                # we don't have to do this but we will anyway.
                self.disable_sageattention_inference()
            if self.unet is not None:
                self.unet = unwrap_model(self.accelerator, self.unet)
            if self.transformer is not None:
                self.transformer = unwrap_model(self.accelerator, self.transformer)
            if (
                "lora" in self.config.model_type
                and "standard" == self.config.lora_type.lower()
            ):
                if self.transformer is not None:
                    transformer_lora_layers = get_peft_model_state_dict(
                        self.transformer
                    )
                elif self.unet is not None:
                    unet_lora_layers = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(self.unet)
                    )
                else:
                    raise Exception(
                        "Couldn't locate the unet or transformer model for export."
                    )

                if self.config.train_text_encoder:
                    self.text_encoder_1 = self.accelerator.unwrap_model(
                        self.text_encoder_1
                    )
                    self.text_encoder_lora_layers = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(self.text_encoder_1)
                    )
                    if self.text_encoder_2 is not None:
                        self.text_encoder_2 = self.accelerator.unwrap_model(
                            self.text_encoder_2
                        )
                        text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(self.text_encoder_2)
                        )
                        if self.text_encoder_3 is not None:
                            text_encoder_3 = self.accelerator.unwrap_model(
                                self.text_encoder_3
                            )
                else:
                    text_encoder_lora_layers = None
                    text_encoder_2_lora_layers = None

                if self.config.model_family == "flux":
                    from diffusers.pipelines import FluxPipeline

                    FluxPipeline.save_lora_weights(
                        save_directory=self.config.output_dir,
                        transformer_lora_layers=transformer_lora_layers,
                        text_encoder_lora_layers=text_encoder_lora_layers,
                    )
                elif self.config.model_family == "sd3":
                    StableDiffusion3Pipeline.save_lora_weights(
                        save_directory=self.config.output_dir,
                        transformer_lora_layers=transformer_lora_layers,
                        text_encoder_lora_layers=text_encoder_lora_layers,
                        text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                    )
                else:
                    StableDiffusionXLPipeline.save_lora_weights(
                        save_directory=self.config.output_dir,
                        unet_lora_layers=unet_lora_layers,
                        text_encoder_lora_layers=text_encoder_lora_layers,
                        text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                    )

                del self.unet
                del self.transformer
                del text_encoder_lora_layers
                del text_encoder_2_lora_layers
                reclaim_memory()
            elif (
                "lora" in self.config.model_type
                and "lycoris" == self.config.lora_type.lower()
            ):
                if (
                    self.accelerator.is_main_process
                    or self.config.use_deepspeed_optimizer
                ):
                    logger.info(
                        f"Saving final LyCORIS checkpoint to {self.config.output_dir}"
                    )
                    # Save final LyCORIS checkpoint.
                    if (
                        getattr(self.accelerator, "_lycoris_wrapped_network", None)
                        is not None
                    ):
                        from helpers.publishing.huggingface import (
                            LORA_SAFETENSORS_FILENAME,
                        )

                        self.accelerator._lycoris_wrapped_network.save_weights(
                            os.path.join(
                                self.config.output_dir, LORA_SAFETENSORS_FILENAME
                            ),
                            list(
                                self.accelerator._lycoris_wrapped_network.parameters()
                            )[0].dtype,
                            {
                                "lycoris_config": json.dumps(self.lycoris_config)
                            },  # metadata
                        )
                        shutil.copy2(
                            self.config.lycoris_config,
                            os.path.join(self.config.output_dir, "lycoris_config.json"),
                        )

            elif self.config.use_ema:
                if self.unet is not None:
                    self.ema_model.copy_to(self.unet.parameters())
                if self.transformer is not None:
                    self.ema_model.copy_to(self.transformer.parameters())

            if self.config.model_type == "full":
                # Now we build a full SDXL Pipeline to export the model with.
                if self.config.model_family == "sd3":
                    self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                        self.config.pretrained_model_name_or_path,
                        text_encoder=self.text_encoder_1
                        or (
                            self.text_encoder_cls_1.from_pretrained(
                                self.config.pretrained_model_name_or_path,
                                subfolder="text_encoder",
                                revision=self.config.revision,
                                variant=self.config.variant,
                            )
                            if self.config.save_text_encoder
                            else None
                        ),
                        tokenizer=self.tokenizer_1,
                        text_encoder_2=self.text_encoder_2
                        or (
                            self.text_encoder_cls_2.from_pretrained(
                                self.config.pretrained_model_name_or_path,
                                subfolder="text_encoder_2",
                                revision=self.config.revision,
                                variant=self.config.variant,
                            )
                            if self.config.save_text_encoder
                            else None
                        ),
                        tokenizer_2=self.tokenizer_2,
                        text_encoder_3=self.text_encoder_3
                        or (
                            self.text_encoder_cls_3.from_pretrained(
                                self.config.pretrained_model_name_or_path,
                                subfolder="text_encoder_3",
                                revision=self.config.revision,
                                variant=self.config.variant,
                            )
                            if self.config.save_text_encoder
                            else None
                        ),
                        tokenizer_3=self.tokenizer_3,
                        vae=self.vae
                        or (
                            self.vae_cls.from_pretrained(
                                self.config.vae_path,
                                subfolder=(
                                    "vae"
                                    if self.config.pretrained_vae_model_name_or_path
                                    is None
                                    else None
                                ),
                                revision=self.config.revision,
                                variant=self.config.variant,
                                force_upcast=False,
                            )
                        ),
                        transformer=self.transformer,
                    )
                    if (
                        self.config.flow_matching
                        and self.config.flow_matching_loss == "diffusion"
                    ):
                        # Diffusion-based SD3 is currently fixed to a Euler v-prediction schedule.
                        self.pipeline.scheduler = SCHEDULER_NAME_MAP[
                            "euler"
                        ].from_pretrained(
                            self.config.pretrained_model_name_or_path,
                            revision=self.config.revision,
                            subfolder="scheduler",
                            prediction_type="v_prediction",
                            timestep_spacing=self.config.training_scheduler_timestep_spacing,
                            rescale_betas_zero_snr=self.config.rescale_betas_zero_snr,
                        )
                        logger.debug(
                            f"Setting scheduler to Euler for SD3. Config: {self.pipeline.scheduler.config}"
                        )
                elif self.config.model_family == "flux":
                    from diffusers.pipelines import FluxPipeline

                    self.pipeline = FluxPipeline.from_pretrained(
                        self.config.pretrained_model_name_or_path,
                        transformer=self.transformer,
                        text_encoder=self.text_encoder_1
                        or (
                            self.text_encoder_cls_1.from_pretrained(
                                self.config.pretrained_model_name_or_path,
                                subfolder="text_encoder",
                                revision=self.config.revision,
                                variant=self.config.variant,
                            )
                            if self.config.save_text_encoder
                            else None
                        ),
                        tokenizer=self.tokenizer_1,
                        vae=self.vae,
                    )
                elif self.config.model_family == "legacy":
                    from diffusers import StableDiffusionPipeline

                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        self.config.pretrained_model_name_or_path,
                        text_encoder=self.text_encoder_1
                        or (
                            self.text_encoder_cls_1.from_pretrained(
                                self.config.pretrained_model_name_or_path,
                                subfolder="text_encoder",
                                revision=self.config.revision,
                                variant=self.config.variant,
                            )
                            if self.config.save_text_encoder
                            else None
                        ),
                        tokenizer=self.tokenizer_1,
                        vae=self.vae
                        or (
                            self.vae_cls.from_pretrained(
                                self.config.vae_path,
                                subfolder=(
                                    "vae"
                                    if self.config.pretrained_vae_model_name_or_path
                                    is None
                                    else None
                                ),
                                revision=self.config.revision,
                                variant=self.config.variant,
                                force_upcast=False,
                            )
                        ),
                        unet=self.unet,
                        torch_dtype=self.config.weight_dtype,
                    )
                elif self.config.model_family == "smoldit":
                    from helpers.models.smoldit import SmolDiTPipeline

                    self.pipeline = SmolDiTPipeline(
                        text_encoder=self.text_encoder_1
                        or (
                            self.text_encoder_cls_1.from_pretrained(
                                self.config.pretrained_model_name_or_path,
                                subfolder="text_encoder",
                                revision=self.config.revision,
                                variant=self.config.variant,
                            )
                            if self.config.save_text_encoder
                            else None
                        ),
                        tokenizer=self.tokenizer_1,
                        vae=self.vae
                        or (
                            self.vae_cls.from_pretrained(
                                self.config.vae_path,
                                subfolder=(
                                    "vae"
                                    if self.config.pretrained_vae_model_name_or_path
                                    is None
                                    else None
                                ),
                                revision=self.config.revision,
                                variant=self.config.variant,
                                force_upcast=False,
                            )
                        ),
                        transformer=self.transformer,
                        scheduler=None,
                    )

                elif self.config.model_family == "sana":
                    from diffusers import SanaPipeline

                    self.pipeline = SanaPipeline.from_pretrained(
                        self.config.pretrained_model_name_or_path,
                        text_encoder=self.text_encoder_1
                        or (
                            self.text_encoder_cls_1.from_pretrained(
                                self.config.pretrained_model_name_or_path,
                                subfolder="text_encoder",
                                revision=self.config.revision,
                                variant=self.config.variant,
                            )
                            if self.config.save_text_encoder
                            else None
                        ),
                        tokenizer=self.tokenizer_1,
                        vae=self.vae,
                        transformer=self.transformer,
                    )

                else:
                    sdxl_pipeline_cls = StableDiffusionXLPipeline
                    if self.config.model_family == "kolors":
                        from helpers.kolors.pipeline import KolorsPipeline

                        sdxl_pipeline_cls = KolorsPipeline
                    self.pipeline = sdxl_pipeline_cls.from_pretrained(
                        self.config.pretrained_model_name_or_path,
                        text_encoder=(
                            self.text_encoder_cls_1.from_pretrained(
                                self.config.pretrained_model_name_or_path,
                                subfolder="text_encoder",
                                revision=self.config.revision,
                                variant=self.config.variant,
                            )
                            if self.config.save_text_encoder
                            else None
                        ),
                        text_encoder_2=(
                            self.text_encoder_cls_2.from_pretrained(
                                self.config.pretrained_model_name_or_path,
                                subfolder="text_encoder_2",
                                revision=self.config.revision,
                                variant=self.config.variant,
                            )
                            if self.config.save_text_encoder
                            else None
                        ),
                        tokenizer=self.tokenizer_1,
                        tokenizer_2=self.tokenizer_2,
                        vae=StateTracker.get_vae()
                        or self.vae_cls.from_pretrained(
                            self.config.vae_path,
                            subfolder=(
                                "vae"
                                if self.config.pretrained_vae_model_name_or_path is None
                                else None
                            ),
                            revision=self.config.revision,
                            variant=self.config.variant,
                            force_upcast=False,
                        ),
                        unet=self.unet,
                        revision=self.config.revision,
                        add_watermarker=self.config.enable_watermark,
                        torch_dtype=self.config.weight_dtype,
                    )
                if (
                    not self.config.flow_matching
                    and self.config.validation_noise_scheduler is not None
                ):
                    self.pipeline.scheduler = SCHEDULER_NAME_MAP[
                        self.config.validation_noise_scheduler
                    ].from_pretrained(
                        self.config.pretrained_model_name_or_path,
                        revision=self.config.revision,
                        subfolder="scheduler",
                        prediction_type=self.config.prediction_type,
                        timestep_spacing=self.config.training_scheduler_timestep_spacing,
                        rescale_betas_zero_snr=self.config.rescale_betas_zero_snr,
                    )
                self.pipeline.save_pretrained(
                    os.path.join(self.config.output_dir, "pipeline"),
                    safe_serialization=True,
                )

            if self.config.push_to_hub and self.accelerator.is_main_process:
                self.hub_manager.upload_model(validation_images, self.webhook_handler)
        self.accelerator.end_training()

            

    def log_metrics(self, logs: dict):
        """
        Log metrics to wandb and console.
        This method can be extended as needed.
        """
        # If the superclass has a logging method, you can delegate to it.
        # For now, we simply print and assume wandb is set up elsewhere.
        logger.info(f"Metrics: {logs}")
        # TODO: Add wandb logging if wandb is configured.
        # e.g., wandb.log(logs)

    def save_checkpoint(self, filepath: str):
        """
        Save checkpoint that preserves both generator (transformer) and discriminator states.
        TODO: Extend with any additional state that needs to be saved.
        """
        logger.info(f"Saving checkpoint to {filepath}")
        checkpoint = {
            "global_step": self.state["global_step"],
            "generator_state": self.transformer.state_dict(),
            "discriminator_state": self.discriminator.state_dict() if self.discriminator is not None else None,
            "optimizer_state": self.optimizer.state_dict(),
            "discriminator_optimizer_state": self.discriminator_optimizer.state_dict()
            if hasattr(self, "discriminator_optimizer")
            else None,
            "lr_scheduler_state": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            # Save additional state if needed.
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """
        Load checkpoint and restore states for both generator and discriminator.
        TODO: Make sure this integrates with the resume checkpoint logic you use.
        """
        logger.info(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=self.accelerator.device)
        self.state["global_step"] = checkpoint.get("global_step", 0)

        self.transformer.load_state_dict(checkpoint["generator_state"])
        if self.discriminator and checkpoint.get("discriminator_state", None):
            self.discriminator.load_state_dict(checkpoint["discriminator_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if hasattr(self, "discriminator_optimizer") and checkpoint.get("discriminator_optimizer_state", None):
            self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state"])
        if self.lr_scheduler and checkpoint.get("lr_scheduler_state", None):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

        logger.info("Checkpoint loaded successfully.")