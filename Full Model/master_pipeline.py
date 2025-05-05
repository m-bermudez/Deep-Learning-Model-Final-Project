import subprocess

def run_step(description, command):
    print(f"\n🚀 Starting: {description}")
    result = subprocess.run(command, shell=True)

    if result.returncode == 0:
        print(f"✅ Finished: {description}\n")
    else:
        print(f"❌ Failed: {description}\n")
        exit(1)

if __name__ == "__main__":
    
    # === Step 1: RAE Anomaly Detection ===
    run_step("Running RAE anomaly detection (RAE_main.py) to generate anomalies with SUBJECT_ID, HADM_ID, GLC, TEXT", "python RAE_main.py")
    
    # === Step 2: BioBERT Summarization (Ground Truth Generation) ===
    run_step("Running PEGASUS inference (PEGASUS_main.py) to generate summaries dataset", "python PEGASUS_main.py")
    
    # === Step 3: BART Fine-tuning + Evaluation (ROUGE metrics) ===
    run_step("Running BART fine-tuning and evaluation (BART_main.py)", "python BART_main.py --mode both")
    
    # === Step 4: BART Inference on RAE anomalies ===
    run_step("Running BART inference on RAE anomalies (BART_RAE_main.py) to summarize TEXT fields", "python BART_RAE_main.py")
    
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")