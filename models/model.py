"""
FedRep Model Components
-----------------------
SharedEncoder  : global encoder trained collaboratively
LocalHead      : client-private classification head
FedRepModel    : convenience wrapper combining both
"""

import torch
import torch.nn as nn
from typing import Optional


class SharedEncoder(nn.Module):
    """
    Global encoder shared across all clients.
    Learns general transaction representations.
    Only this module's weights are sent to / received from the server.
    """

    def __init__(self, input_dim: int, encoding_dim: int = 64, hidden_dims: Optional[list] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 96]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
            prev_dim = h

        layers += [nn.Linear(prev_dim, encoding_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.encoding_dim = encoding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalHead(nn.Module):
    """
    Client-private classification head.
    Never leaves the client. Enables personalisation.
    """

    def __init__(self, encoding_dim: int = 64, num_classes: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        return self.net(encodings)


class FedRepModel(nn.Module):
    """Full model = encoder + head. Used locally on each client."""

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 64,
        hidden_dims: Optional[list] = None,
        num_classes: int = 2,
        head_hidden_dim: int = 32,
    ):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, encoding_dim, hidden_dims)
        self.head = LocalHead(encoding_dim, num_classes, head_hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, encodings)."""
        encodings = self.encoder(x)
        logits = self.head(encodings)
        return logits, encodings

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
