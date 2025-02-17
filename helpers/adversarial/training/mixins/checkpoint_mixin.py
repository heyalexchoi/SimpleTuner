import os
import shutil
from accelerate.logging import get_logger

logger = get_logger(__name__)

class CheckpointMixin:
    """Mixin class that handles checkpoint saving and loading"""

    def _handle_checkpointing(self, step, webhook_pending_msg):
        """Handle checkpoint saving logic"""
        if self.state["global_step"] % self.config.checkpointing_steps == 0:
            self._send_webhook_msg(
                message=f"Checkpoint: `{webhook_pending_msg}`",
                message_level="info",
            )
            
            if self.accelerator.is_main_process:
                self._cleanup_old_checkpoints()
                
            if (
                self.accelerator.is_main_process
                or self.config.use_deepspeed_optimizer
            ):
                save_path = os.path.join(
                    self.config.output_dir,
                    f"checkpoint-{self.state['global_step']}",
                )
                self._save_checkpoint(save_path)

    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints based on total limit"""
        if self.config.checkpoints_total_limit is not None:
            checkpoints = os.listdir(self.config.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) >= self.config.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - self.config.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]
                
                logger.debug(
                    f"{len(checkpoints)} checkpoints exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.debug(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(
                        self.config.output_dir, removing_checkpoint
                    )
                    try:
                        shutil.rmtree(removing_checkpoint, ignore_errors=True)
                    except Exception as e:
                        logger.error(f"Failed to remove directory: {removing_checkpoint}")
                        print(e)

    def _save_checkpoint(self, save_path):
        """Save checkpoint state"""
        print("\n")
        self.mark_optimizer_eval()
        self.accelerator.save_state(save_path)
        self.mark_optimizer_train()
        
        # Save sampler states
        for _, backend in StateTracker.get_data_backends().items():
            if "sampler" in backend:
                logger.debug(f"Backend: {backend}")
                backend["sampler"].save_state(
                    state_path=os.path.join(
                        save_path,
                        self.model_hooks.training_state_path,
                    ),
                ) 