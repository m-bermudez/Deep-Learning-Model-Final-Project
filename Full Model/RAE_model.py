import torch
import torch.nn as nn

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, num_features, hidden_size, num_layers, dropout=0.2):
        super(RecurrentAutoencoder, self).__init__()

        self.seq_len = seq_len
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder
        self.lstm_encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Learned initial input for decoder
        self.decoder_init = nn.Parameter(torch.randn(1, 1, hidden_size))  # Trainable start vector

        # Decoder
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, num_features)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Encoder pass
        _, (hidden_state, cell_state) = self.lstm_encoder(x)

        # Prepare decoder input (learned vector repeated for seq_len steps)
        decoder_input = self.decoder_init.repeat(batch_size, self.seq_len, 1)

        # Decoder pass
        decoder_output, _ = self.lstm_decoder(decoder_input, (hidden_state, cell_state))

        # Project decoder output back to input dimension
        reconstructions = self.output_layer(decoder_output)

        return reconstructions
