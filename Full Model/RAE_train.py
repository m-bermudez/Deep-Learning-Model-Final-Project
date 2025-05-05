import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from RAE_utils import reconstruction_errors

def train_model(model, train_loader, criterion, optimizer, num_epochs, save_dir, device):
    """Train the model on GLC sequences."""
    
    model.to(device)
    train_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for sequences in train_loader:
            sequences = sequences.to(device)  # sequences = (batch_size, seq_len, 1)

            optimizer.zero_grad()
            reconstructions = model(sequences)
            loss = criterion(reconstructions, sequences)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * sequences.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f"rae_epoch_{epoch+1}.pth"))
        wandb.log({"Train Loss": train_losses})

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
    plt.title("Recurrent Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)

    plot_filename = os.path.join(save_dir, "train_loss_plot.png")
    plt.savefig(plot_filename)
    plt.close()

    wandb.log({"Train Loss Plot": wandb.Image(plot_filename)})

    print(f"Training loss plot saved to {plot_filename}")

    return model  # ✅✅✅ MUST RETURN THE MODEL so it can be used in RAE_main.py
