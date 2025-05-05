import os
import random
import logging
import torch
import numpy as np
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

class Config:
    model_id: str = "facebook/bart-large"
    
    # Dataset paths
    pegasus_output_path: str = "./output_PEGASUS/generated_train_summaries.jsonl"
    mimic_test_path: str = "./mimic_test.jsonl"
    
    # PEFT adapter paths
    peft_output_dir: str = "./peft_adapter"              # base path → checkpoints per epoch
    peft_final_dir: str = "./peft_adapter/final"         # final model for inference/evaluation
    output_dir: str = "./bart_eval_output"
    # Training hyperparameters
    num_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 4e-5
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 200
    logging_steps: int = 50

    # Input/output max lengths
    max_input_length: int = 512
    max_target_length: int = 80

    # Generation parameters
    max_new_tokens: int = 80
    num_beams: int = 6
    repetition_penalty: float = 2.5
    length_penalty: float = 1.2
    early_stopping: bool = True
    no_repeat_ngram_size: int = 5

    # Other options
    skip_quantization: bool = False
    label_smoothing_factor: float = 0.1

config = Config()
