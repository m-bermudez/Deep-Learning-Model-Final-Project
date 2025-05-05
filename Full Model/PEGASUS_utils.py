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

@dataclass
class Config:
    model_id: str = "google/pegasus-large"

    dataset_path: str = "./mimic_iv_clinic_filtered.jsonl"
    train_split_path: str = "./mimic_train.jsonl"
    test_split_path: str = "./mimic_test.jsonl"
    dataset_text_field: str = "text"
    output_dir: str = "./output_PEGASUS"

    max_input_length: int = 512
    max_target_length: int = 80
    min_target_length: int = 50

    max_new_tokens: int = 80
    num_beams: int = 6
    repetition_penalty: float = 2.0
    length_penalty: float = 1.2
    early_stopping = True
    no_repeat_ngram_size: int = 5

config = Config()

