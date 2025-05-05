import torch
import re
import pandas as pd
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
from BART_utils import config, logger
from BART_data import load_eval_data

def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()

def run_evaluation():
    logger.info("Starting evaluation on mimic_test.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base model
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_id).to(device)

    # Load PEFT model
    model = PeftModel.from_pretrained(model, config.peft_final_dir)
    model = model.merge_and_unload().to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    df = load_eval_data()

    results = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_list, rouge2_list, rougeL_list = [], [], []

    for idx, row in df.iterrows():
        text = row["text"]
        ground_truth = row.get("Summary", "")

        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=config.max_input_length).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty,
                early_stopping=config.early_stopping
            )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate ROUGE
        scores = scorer.score(summary, ground_truth)

        rouge1_list.append(scores["rouge1"].fmeasure)
        rouge2_list.append(scores["rouge2"].fmeasure)
        rougeL_list.append(scores["rougeL"].fmeasure)

        results.append({
            "text": text,
            "Ground Truth": ground_truth,
            "Generated Summary": summary,
            "ROUGE-1": scores["rouge1"].fmeasure,
            "ROUGE-2": scores["rouge2"].fmeasure,
            "ROUGE-L": scores["rougeL"].fmeasure,
        })

        print(f"Number of summaries {idx}")
        if idx < 3:
            logger.info(f"\nInput: {text[:300]}...")
            logger.info(f"Generated Summary: {summary}")
            logger.info(f"Ground Truth: {ground_truth}")
            logger.info(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}, ROUGE-2: {scores['rouge2'].fmeasure:.4f}, ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, "bart_eval_results.jsonl")
    pd.DataFrame(results).to_json(output_path, orient="records", lines=True)
    logger.info("Saved evaluation results.")

    # Calculate and log average ROUGE
    avg_rouge1 = sum(rouge1_list) / len(rouge1_list)
    avg_rouge2 = sum(rouge2_list) / len(rouge2_list)
    avg_rougeL = sum(rougeL_list) / len(rougeL_list)

    logger.info("\n=== Average ROUGE Scores ===")
    logger.info(f"ROUGE-1: {avg_rouge1:.4f}")
    logger.info(f"ROUGE-2: {avg_rouge2:.4f}")
    logger.info(f"ROUGE-L: {avg_rougeL:.4f}")
