import os
from accelerate.logging import get_logger

def get_adversarial_logger():
    return get_logger(
        "SimpleTuner.AdversarialTrainer",
        log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
    ) 