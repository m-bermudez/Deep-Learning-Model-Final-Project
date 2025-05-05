import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from BART_RAE_utils import config, logger

def load_model():
    logger.info(f"Loading BART + LoRA model from {config.peft_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_id)
    model = PeftModel.from_pretrained(base_model, config.peft_model_path)
    model = model.merge_and_unload().to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, device
