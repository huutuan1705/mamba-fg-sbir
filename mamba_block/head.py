import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaHead(nn.Module):
    def __init__(self, d_model: int, output_size: int, dropout: float = 0.1):
        """
        Simple head that takes sequence output.

        Args:
            d_model: Hidden dimension size from the Mamba model
            output_size: output size of block
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, output_size)
        """
        # Average pooling over sequence length
        x = x.mean(dim=1)  # (batch_size, d_model)

        # Apply layer norm and dropout
        x = self.norm(x)
        # x = self.dropout(x)

        # Final layer
        return F.normalize(self.fc(x))

# model = MambaHead(d_model=2048, output_size=64)
# dummy_input = torch.randn(48, 25, 2048)
# output = model(dummy_input)
# print("Output shape:", output.shape) # (48, 64)