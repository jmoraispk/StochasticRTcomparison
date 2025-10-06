import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim=4, seq_len=20, model_dim=32, num_heads=1, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.model_dim = model_dim

        # Project input features to model dimension
        self.input_proj = nn.Linear(input_dim, model_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final linear layer takes the concatenated transformer outputs
        self.output_proj = nn.Linear(seq_len * model_dim, input_dim)

        # If output_proj gets too big, we can use this:
        # # Reduce sequence to a single representation (mean pooling)
        # self.pool = nn.AdaptiveAvgPool1d(1)
        # # Project back to 4-dim output
        # self.output_proj = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        returns: (batch_size, input_dim)
        """
        # Project to model dimension
        x = self.input_proj(x)  # (batch_size, seq_len, model_dim)

        # Pass through transformer
        x = self.encoder(x)  # (batch_size, seq_len, model_dim)

        # Flatten all sequence outputs
        x = x.reshape(x.size(0), -1)  # (batch_size, seq_len * model_dim)

        # Project to final 4-dim output
        x = self.output_proj(x)  # (batch_size, input_dim)

        # If output_proj gets too big, we can use this:
        # # Pool over sequence dimension
        # x = x.transpose(1, 2)  # (batch_size, model_dim, seq_len)
        # x = self.pool(x).squeeze(-1)  # (batch_size, model_dim)

        # # Project to final 4-dim output
        # x = self.output_proj(x)  # (batch_size, input_dim)

        return x


# Example usage
if __name__ == "__main__":
    batch_size = 8
    seq_len = 20
    input_dim = 4

    model = TransformerModel(input_dim=input_dim, seq_len=seq_len)
    x = torch.randn(batch_size, seq_len, input_dim)
    y = model(x)

    print("Input shape:", x.shape)   # (8, 20, 4)
    print("Output shape:", y.shape) # (8, 4)