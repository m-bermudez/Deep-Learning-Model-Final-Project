import os
import random
import logging
import torch
import numpy as np
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

@dataclass
class Config:
    model_id: str = "facebook/bart-large"
    peft_model_path: str = "./peft_adapter/final" 
    anomalies_path: str = "./saved_models/anomalies_json"
    output_path: str = "./BART_RAE_output"

    max_input_length: int = 512
    max_target_length: int = 40
    #min_target_length: int = 20  # ✅ NEW (you can adjust!)

    max_new_tokens: int = 40
    num_beams: int = 6
    repetition_penalty: float = 2.4
    length_penalty: float = 1.2
    early_stopping: bool = True
    no_repeat_ngram_size: int = 5

config = Config()