from BART_RAE_utils import logger
from BART_RAE_eval import run_inference

def main():
    logger.info("Running BART_RAE inference pipeline...")
    run_inference()

if __name__ == "__main__":
    main()
