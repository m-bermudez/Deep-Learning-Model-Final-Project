import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PEGASUS_utils import config, logger

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()

def create_and_save_splits(df):
    df[config.dataset_text_field] = df[config.dataset_text_field].apply(clean_text)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_json(config.train_split_path, orient="records", lines=True)
    logger.info(f"Saved {len(train_df)} train records to {config.train_split_path}")
    test_df.to_json(config.test_split_path, orient="records", lines=True)
    logger.info(f"Saved {len(test_df)} test records to {config.test_split_path}")
    return train_df, test_df

def split_and_save_dataset():
    if os.path.exists(config.train_split_path) and os.path.exists(config.test_split_path):
        logger.info("Train/Test split files found. Loading them directly.")
        train_df = pd.read_json(config.train_split_path, lines=True)
        test_df = pd.read_json(config.test_split_path, lines=True)
        return train_df, test_df

    logger.info("Train/Test split files not found. Creating new split...")
    df = pd.read_json(config.dataset_path, lines=True)
    logger.info(f"Loaded {len(df)} total records from original dataset.")
    return create_and_save_splits(df)

