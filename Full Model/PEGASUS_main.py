from PEGASUS_data import split_and_save_dataset
from PEGASUS_eval import run_evaluation

def main():
    train_df, test_df = split_and_save_dataset()
    train_df = train_df.reset_index(drop=True)
    run_evaluation(train_df)

if __name__ == "__main__":
    main()
