import pandas as pd
from transformers import AutoTokenizer
from BART_utils import config, logger
from torch.utils.data import Dataset

tokenizer = AutoTokenizer.from_pretrained(config.model_id)

def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()

def load_train_data():
    logger.info(f"Loading training data from {config.pegasus_output_path}")

    # Load and clean
    df = pd.read_json(config.pegasus_output_path, lines=True)

    df["text"] = df["text"].apply(clean_text)
    df["Summary"] = df["Summary"].apply(clean_text)

    df = df.reset_index(drop=True)

    logger.info(f"Loaded {len(df)} training samples.")
    return df

def load_eval_data():
    logger.info(f"Loading evaluation data from {config.mimic_test_path}")

    # Load and clean
    df = pd.read_json(config.mimic_test_path, lines=True)

    df["text"] = df["text"].apply(clean_text)

    df = df.reset_index(drop=True)

    logger.info(f"Loaded {len(df)} evaluation samples.")
    return df

# ✅ Custom Dataset for PyTorch DataLoader
class SummarizationDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "text": row["text"],
            "Summary": row["Summary"]
        }
