import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PEGASUS_utils import config, logger

# --- Helper to clean generated summaries ---
def clean_summary(summary):
    patterns_to_remove = [
        r"Copyright.*",
        r"E-mail this Article.*",
        r"Print this Article.*",
        r"Share this Article.*",
        r"If you would like to receive this article.*"
    ]
    for pattern in patterns_to_remove:
        summary = re.sub(pattern, "", summary, flags=re.IGNORECASE)

    return summary.strip()

def run_evaluation(train_df):
    logger.info("Starting Pegasus evaluation (summary generation)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    summaries = []

    # --- Start summarization ---
    for idx, row in train_df.iterrows():
        text = row[config.dataset_text_field]

        # Prepare inputs
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=config.max_input_length).to(device)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=config.max_target_length,
                num_beams=config.num_beams,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty,
                early_stopping=config.early_stopping,
                no_repeat_ngram_size=config.no_repeat_ngram_size
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        summaries.append({
            "text": text,
            "Summary": summary
        })

        print(f"Summary number {idx}")
        if idx < 3:
            logger.info(f"Example {idx+1}:\n{text[:300]}...\nSummary: {summary}")

    # --- Final cleaning and filtering ---
    cleaned_summaries = []
    for item in summaries:
        cleaned_summary = clean_summary(item["Summary"])

        # Skip too short cleaned summaries
        if len(cleaned_summary.split()) <= 7:
            continue

        cleaned_summaries.append({
            "text": item["text"],
            "Summary": cleaned_summary
        })

    # --- Save results ---
    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, "generated_train_summaries.jsonl")

    pd.DataFrame(cleaned_summaries).to_json(output_path, orient="records", lines=True)
    logger.info(f"Summaries saved to {output_path}")



