import pandas as pd
import os

def split_data_in_memory(csv_path="glucose_insulin_ICU.csv", jsonl_path="diabetes_text_with_glucose.jsonl"):
    

    # Load datasets
    print("Loading CSV (training set)...")
    df_train = pd.read_csv(csv_path)

    print("Loading JSONL (test set)...")
    df_test = pd.read_json(jsonl_path, lines=True)

    # Clean training dataframe
    print("Cleaning training set...")
    df_train = df_train.dropna(subset=["GLC"]).copy()

    # Convert GLC to integer
    df_train["GLC"] = pd.to_numeric(df_train["GLC"], errors="coerce")
    df_train = df_train.dropna(subset=["GLC"])
    df_train["GLC"] = df_train["GLC"].astype(int)

    # Remove training SUBJECT_IDs that are present in test set
    test_subject_ids = df_test["SUBJECT_ID"].unique()
    df_train = df_train[~df_train["SUBJECT_ID"].isin(test_subject_ids)]

    # Keep only GLC between 80 and 170 inclusive
    df_train = df_train[df_train["GLC"].between(80, 170)]

    # Analyze training set
    train_unique_subjects = df_train["SUBJECT_ID"].nunique()
    train_total_rows = len(df_train)

    # Clean test dataframe
    print("Cleaning test set...")
    df_test["GLC"] = pd.to_numeric(df_test["GLC"], errors="coerce")
    df_test = df_test.dropna(subset=["GLC"])
    df_test["GLC"] = df_test["GLC"].astype(int)

    # Analyze test set
    test_normal_count = df_test["GLC"].between(80, 170).sum()
    test_anomaly_count = (~df_test["GLC"].between(50, 200)).sum()

    # Display summary
    print("\n=== Training Set Summary ===")
    print(f"Unique SUBJECT_IDs: {train_unique_subjects}")
    print(f"Total Rows: {train_total_rows}")

    print("\n=== Test Set Summary ===")
    print(f"Normal GLC rows (80-170): {test_normal_count}")
    print(f"Anomalies (GLC < 50 or > 200): {test_anomaly_count}")

    return df_train, df_test
