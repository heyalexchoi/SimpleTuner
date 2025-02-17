from accelerate.logging import get_logger

logger = get_logger(__name__)

class ValidationMixin:
    """Mixin class that handles validation logic"""

    def _handle_validation(self, step, wandb_logs):
        """Handle validation steps"""
        if self.validation is not None:
            if self.validation.would_validate():
                self.mark_optimizer_eval()
                self.enable_sageattention_inference()
                self.disable_gradient_checkpointing()
                
            self.validation.run_validations(
                validation_type="intermediary", 
                step=step
            )
            
            if self.validation.would_validate():
                self.disable_sageattention_inference()
                self.enable_gradient_checkpointing()
                self.mark_optimizer_train()

        if self.evaluation is not None and self.evaluation.would_evaluate(self.state):
            self._handle_evaluation(wandb_logs)

    def _handle_evaluation(self, wandb_logs):
        """Handle evaluation steps"""
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