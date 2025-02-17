import logging
from accelerate.logging import get_logger
import wandb

logger = get_logger(__name__)

class LoggingMixin:
    """Mixin class that handles logging functionality"""

    def _log_training_step(self, loss, step, epoch, wandb_logs=None):
        """Log metrics for a training step"""
        if wandb_logs is None:
            wandb_logs = {}
            
        wandb_logs.update({
            "train_loss": self.train_loss,
            "optimization_loss": loss,
            "learning_rate": self.lr,
            "epoch": epoch,
        })

        if self.config.model_family == "flux" and self.guidance_values_list:
            guidance_values = torch.tensor(self.guidance_values_list).mean()
            wandb_logs["mean_cfg"] = guidance_values.item()
            self.guidance_values_list = []

        if self.grad_norm is not None:
            if self.config.grad_clip_method == "norm":
                wandb_logs["grad_norm"] = self.grad_norm
            else:
                wandb_logs["grad_absmax"] = self.grad_norm

        # Log timestep distribution
        if self.config.report_to == "wandb" and self.accelerator.is_main_process:
            self._log_timestep_distribution(wandb_logs)

        self.accelerator.log(wandb_logs, step=self.state["global_step"])

        logs = {
            "step_loss": loss.detach().item(),
            "lr": float(self.lr)
        }
        if self.grad_norm is not None:
            if self.config.grad_clip_method == "norm":
                logs["grad_norm"] = float(self.grad_norm)
            elif self.config.grad_clip_method == "value":
                logs["grad_absmax"] = self.grad_norm

        return logs

    def _log_timestep_distribution(self, wandb_logs):
        """Log timestep distribution to wandb"""
        data = [
            [iteration, timestep]
            for iteration, timestep in self.timesteps_buffer
        ]
        table = wandb.Table(data=data, columns=["global_step", "timestep"])
        wandb_logs["timesteps_scatter"] = wandb.plot.scatter(
            table,
            "global_step",
            "timestep",
            title="Timestep distribution by step"
        )
        self.timesteps_buffer = [] 