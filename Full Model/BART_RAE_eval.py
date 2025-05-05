import os
import json
import torch
from BART_RAE_utils import config, logger
from BART_RAE_model import load_model

def load_anomalies():
    anomalies = []
    for filename in os.listdir(config.anomalies_path):
        if filename.endswith(".json"):
            with open(os.path.join(config.anomalies_path, filename)) as f:
                anomaly = json.load(f)
                glc = anomaly.get("GLC")

                # Make sure GLC is numeric and apply filter
                if glc is not None:
                    try:
                        glc_value = int(glc)
                        if glc_value < 80 or glc_value > 170:
                            anomalies.append(anomaly)
                    except ValueError:
                        pass  # Ignore non-numeric GLC
                        
    logger.info(f"Loaded {len(anomalies)} anomalies after GLC filtering")
    return anomalies

def summarize_notes(model, tokenizer, device, notes):
    inputs = tokenizer(notes, return_tensors="pt", padding=True, truncation=True, max_length=config.max_input_length).to(device)
    
    with torch.no_grad():
        ids = model.generate(
            inputs["input_ids"],
            max_length=config.max_target_length,
            num_beams=config.num_beams,
            repetition_penalty=config.repetition_penalty,
            length_penalty=config.length_penalty,
            early_stopping=config.early_stopping,
            no_repeat_ngram_size=config.no_repeat_ngram_size
        )

    summary = tokenizer.decode(ids[0], skip_special_tokens=True)
    return summary

def run_inference():
    logger.info("Starting BART inference on RAE anomalies...")

    # ✅ Make sure output directory exists
    os.makedirs(config.output_path, exist_ok=True)

    model, tokenizer, device = load_model()
    anomalies = load_anomalies()

    results = []
    i = 1

    for anomaly in anomalies:
        notes = anomaly.get("TEXT", "")
        summary = summarize_notes(model, tokenizer, device, notes)

        # ✅ Check if summary has more than 7 words
        if len(summary.strip().split()) <= 7:
            logger.info(f"Skipping short summary for anomaly {i} (summary too short)")
            i += 1
            continue

        logger.info(f"Summary number {i} completed")
        result = {
            "SUBJECT_ID": anomaly.get("SUBJECT_ID"),
            "HADM_ID": anomaly.get("HADM_ID"),
            "GLC": anomaly.get("GLC"),
            "TEXT": notes,
            "BART_SUMMARY": summary
        }
        results.append(result)
        i += 1

    # ✅ Create output file path
    output_file = os.path.join(config.output_path, "bart_summaries.jsonl")

    # ✅ Save jsonl results
    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    logger.info(f"Inference complete! Saved {len(results)} summaries to {output_file}")



