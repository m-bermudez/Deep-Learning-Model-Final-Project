import argparse
from BART_utils import logger
from BART_model import model, tokenizer
from BART_data import load_train_data
from BART_train import run_training
from BART_eval import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BART with LoRA")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "both"], default="both")
    args = parser.parse_args()

    logger.info(f"Running BART pipeline in {args.mode} mode")

    if args.mode in ["train", "both"]:
        dataset = load_train_data()
        run_training(model, tokenizer, dataset)

    if args.mode in ["eval", "both"]:
        run_evaluation()

if __name__ == "__main__":
    main()
