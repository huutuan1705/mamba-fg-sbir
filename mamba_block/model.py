from typing import Optional

import torch
import torch.nn as nn

from backbone import Mamba, MambaConfig
from head import MambaHead

class MambaModule(nn.Module):
    def __init__(
        self,
        d_model: int = 16,
        n_layers: int = 2,
        output_size: int = 64,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Hidden dimension size
            n_layers: Number of Mamba layers
            output_size: dimension of embedding space
            dropout: Dropout rate for head
        """
        super().__init__()
        config = MambaConfig(d_model=d_model, n_layers=n_layers)
        self.backbone = Mamba(config)
        self.head = MambaHead(d_model=d_model, output_size=output_size, dropout=dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, channels) for embeddings
        """
        sequence_output = self.backbone(x)
        output = self.head(sequence_output)
        
        return output
    
# model = MambaModule(d_model=2048, output_size=64)
# dummy_input = torch.randn(48, 25, 2048)
# output = model(dummy_input)
# print("Output shape:", output.shape)