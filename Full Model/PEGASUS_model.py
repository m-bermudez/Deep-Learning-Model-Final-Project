import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PEGASUS_utils import config, logger

def load_model():
    logger.info(f"Loading Pegasus model: {config.model_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    logger.info("Pegasus model loaded.")
    return model, tokenizer
