import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from BART_utils import config, logger

logger.info("Loading BART model...")
model = AutoModelForSeq2SeqLM.from_pretrained(config.model_id)
tokenizer = AutoTokenizer.from_pretrained(config.model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
logger.info("Model ready with LoRA applied.")