"""
FedProx Server
--------------
Maintains the global encoder. Each round it:
  1. Broadcasts current global encoder weights to all clients.
  2. Receives privatised encodings (already DP-noised) from clients.
  3. Aggregates encoder weight updates with FedProx-style weighted averaging.

FedProx difference from FedAvg
--------------------------------
In standard FedAvg the server simply averages client encoder updates.
FedProx adds a proximal penalty  μ/2 * ||w - w_global||^2  on the CLIENT
side during local optimisation to keep local models close to the global one.
The server aggregation itself is still a weighted average — the proximal
effect comes from the constrained local training, not from a special
server-side step.

This module handles the server-side weight aggregation and global state.
The proximal penalty μ is passed to each client for their local training.
"""

import copy
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.model import SharedEncoder


class FedProxServer:
    """
    Central server that owns and aggregates the global encoder.

    Parameters
    ----------
    global_encoder : SharedEncoder
        The initial global encoder (all clients get a copy each round).
    mu : float
        FedProx proximal penalty coefficient sent to clients.
    device : torch.device
    """

    def __init__(
        self,
        global_encoder: SharedEncoder,
        mu: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.global_encoder = global_encoder.to(self.device)
        self.mu = mu
        self.round = 0
        self._history: List[dict] = []

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    def get_global_weights(self) -> OrderedDict:
        """Return a deep copy of the global encoder's state_dict."""
        return copy.deepcopy(self.global_encoder.state_dict())

    # ------------------------------------------------------------------
    # Aggregation (FedProx = weighted FedAvg on the client-constrained updates)
    # ------------------------------------------------------------------

    def aggregate(
        self,
        client_weights: List[OrderedDict],
        client_sizes: List[int],
    ) -> None:
        """
        Federated weighted average of client encoder state_dicts.

        client_weights : list of encoder state_dicts from participating clients
        client_sizes   : number of local samples per client (used for weighting)
        """
        if not client_weights:
            raise ValueError("No client weights received for aggregation.")

        total = sum(client_sizes)
        weights_norm = [s / total for s in client_sizes]

        new_state = OrderedDict()
        for key in client_weights[0]:
            # Weighted sum of each parameter tensor
            new_state[key] = sum(
                w * client_weights[i][key].float()
                for i, w in enumerate(weights_norm)
            )

        self.global_encoder.load_state_dict(new_state)
        self.round += 1

        self._history.append({
            "round": self.round,
            "n_clients": len(client_weights),
            "total_samples": total,
        })

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "round": self.round,
                "encoder_state": self.global_encoder.state_dict(),
                "mu": self.mu,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.global_encoder.load_state_dict(ckpt["encoder_state"])
        self.round = ckpt.get("round", 0)
        self.mu = ckpt.get("mu", self.mu)

    def history(self) -> List[dict]:
        return self._history

    def __repr__(self):
        return (
            f"FedProxServer(round={self.round}, "
            f"mu={self.mu}, encoder={self.global_encoder})"
        )
