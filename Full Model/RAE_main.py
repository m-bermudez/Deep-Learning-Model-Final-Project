import torch
import os
import wandb
from RAE_data import preprocess_data
from RAE_model import RecurrentAutoencoder
from RAE_train import train_model
from RAE_eval import evaluate_model_mix
from RAE_glucose import split_data_in_memory
from RAE_utils import RAE_CONFIG

# --- Load WandB API key ---
with open('API_KEY.txt', 'r') as file:
    wandb_api_key = file.read().strip()

if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    raise ValueError("WANDB_API_KEY not found in API_KEY.txt!")

# --- Initialize WandB ---
wandb.init(project="Final-Project", entity="usf-magma", config=RAE_CONFIG)
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# --- Load and preprocess data ---
df_train, df_numeric_test = split_data_in_memory()
train_loader, test_loader, num_features, test_meta = preprocess_data(df_train, df_numeric_test, config)

# --- Initialize model ---
model = RecurrentAutoencoder(
    seq_len=config["sequence_length"],
    num_features=num_features,
    hidden_size=config["hidden_size"],
    num_layers=config["num_layers"],
    dropout=config["dropout"]
).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

# --- Train model ---
model = train_model(model, train_loader, criterion, optimizer, config["epochs"], save_dir, device)

# --- Evaluate model + save anomalies ---
evaluate_model_mix(model, test_loader, device, save_dir, test_meta, config["sequence_length"])

wandb.finish()
