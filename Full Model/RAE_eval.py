import numpy as np
import torch
import wandb
import seaborn as sns
import os
import matplotlib.pyplot as plt
import json
from RAE_utils import reconstruction_errors

def evaluate_model_mix(model, test_loader, device, save_dir, test_meta, sequence_length=7):
    """Evaluate the model on the test set and save plots + detected anomalies."""
    
    model.to(device)
    model.eval()

    all_sequences_np = []
    all_reconstructions_np = []

    print("Gathering reconstructions for test set...")
    with torch.no_grad():
        for sequences in test_loader:
            sequences = sequences.to(device)
            reconstructions = model(sequences)

            all_sequences_np.append(sequences.cpu().numpy())
            all_reconstructions_np.append(reconstructions.cpu().numpy())

    all_sequences_np = np.concatenate(all_sequences_np, axis=0)
    all_reconstructions_np = np.concatenate(all_reconstructions_np, axis=0)

    # Calculate reconstruction errors
    reconstruction_errors_arr = np.mean((all_sequences_np - all_reconstructions_np) ** 2, axis=(1, 2))

    # Calculate threshold using mean + 2.5 * std (dynamic threshold)
    threshold = np.percentile(reconstruction_errors_arr, 90)
    print(f"Anomaly Threshold (90 percentile): {threshold:.6f}")

    # Separate normal and anomaly errors
    normal_errors = reconstruction_errors_arr[reconstruction_errors_arr <= threshold]
    anomaly_errors = reconstruction_errors_arr[reconstruction_errors_arr > threshold]

    # === Save reconstruction error plots ===
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(normal_errors, bins=50, color="blue", alpha=0.6, label="Normal", kde=True)
    sns.histplot(anomaly_errors, bins=50, color="red", alpha=0.6, label="Anomaly", kde=True)
    plt.axvline(threshold, color='black', linestyle='--', label="Threshold")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title("Histogram of Reconstruction Errors")
    plt.legend()

    # Density plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(normal_errors, color="blue", fill=True, label="Normal")
    sns.kdeplot(anomaly_errors, color="red", fill=True, label="Anomaly")
    plt.axvline(threshold, color='black', linestyle='--', label="Threshold")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.title("Density Plot of Reconstruction Errors")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "reconstruction_error_plots.png")
    plt.savefig(plot_path)
    plt.close()
    wandb.log({"Reconstruction Error Plots": wandb.Image(plot_path)})

    # Calculate and log average loss
    avg_loss = np.mean(reconstruction_errors_arr)
    print(f"Average Reconstruction MSE: {avg_loss:.6f}")
    wandb.log({"avg_test_reconstruction_loss": avg_loss})

    # === Save anomalies (HYBRID LOGIC: only save if reconstruction + GLC abnormal) ===
    anomalies_dir = os.path.join(save_dir, "anomalies_json")
    os.makedirs(anomalies_dir, exist_ok=True)

    anomaly_count = 0
    anomaly_real = 0

    if test_meta is not None:
        for idx, error in enumerate(reconstruction_errors_arr):
            meta = test_meta[idx]
            glc_value = int(meta["GLC"])

            is_glc_anomalous = glc_value < 80 or glc_value > 170
            is_reconstruction_anomalous = error > threshold
            
            if is_reconstruction_anomalous:
                anomaly_real = anomaly_real + 1        
            if is_glc_anomalous and is_reconstruction_anomalous:
                anomaly_data = {
                    "SUBJECT_ID": str(meta.get("SUBJECT_ID", "")),
                    "HADM_ID": str(meta.get("HADM_ID", "")),
                    "GLC": glc_value,
                    "TEXT": meta.get("TEXT", "")
                }
                with open(os.path.join(anomalies_dir, f"anomaly_{idx}.json"), "w") as f:
                    json.dump(anomaly_data, f)

                anomaly_count += 1

    print(f"Total anomalies saved: {anomaly_count}")
    print(f"Real anomalies in the RAE: {anomaly_real}")
    wandb.log({"saved_anomalies_count": anomaly_count})

    return reconstruction_errors_arr