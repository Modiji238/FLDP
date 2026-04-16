"""
FedRep Client
-------------
Each client:
  1. Receives global encoder weights from server.
  2. Runs FedRep local training:
       Phase A — freeze encoder, train local head for `head_steps` steps.
       Phase B — freeze head, train encoder for `encoder_steps` steps
                 with FedProx proximal penalty ||w_enc - w_global||^2.
  3. Applies DP (clip + noise) to the mean encoding of its local data.
  4. Sends updated encoder weights + DP encoding to server.

The local head NEVER leaves the client.
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from models.model import FedRepModel
from models.dp_layer import DPEncodingLayer


class FedRepClient:
    """
    Simulated federated learning client.

    Parameters
    ----------
    client_id   : int
    X, y        : local dataset tensors (float32, long)
    input_dim   : feature dimension
    encoding_dim: encoder output dimension
    hidden_dims : encoder hidden layer sizes
    dp_layer    : DPEncodingLayer instance (shared config, independent noise)
    mu          : FedProx proximal coefficient
    lr          : learning rate for both phases
    batch_size  : local mini-batch size
    head_steps  : number of gradient steps for local head per round
    encoder_steps: number of gradient steps for encoder per round
    device      : torch device
    """

    def __init__(
        self,
        client_id: int,
        X: torch.Tensor,
        y: torch.Tensor,
        input_dim: int,
        encoding_dim: int = 64,
        hidden_dims: Optional[list] = None,
        dp_layer: Optional[DPEncodingLayer] = None,
        mu: float = 0.01,
        lr: float = 1e-3,
        batch_size: int = 256,
        head_steps: int = 10,
        encoder_steps: int = 10,
        device: Optional[torch.device] = None,
    ):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.n_samples = len(y)
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims or [128, 96]
        self.dp_layer = dp_layer
        self.mu = mu
        self.lr = lr
        self.batch_size = batch_size
        self.head_steps = head_steps
        self.encoder_steps = encoder_steps
        self.device = device or torch.device("cpu")

        # Build local model (encoder + head)
        self.model = FedRepModel(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        # Weighted sampler to handle class imbalance
        self.dataloader = self._make_dataloader()

        # Track metrics
        self.train_history: list = []

    # ------------------------------------------------------------------
    # Data utilities
    # ------------------------------------------------------------------

    def _make_dataloader(self) -> DataLoader:
        """
        Build a DataLoader with WeightedRandomSampler for class imbalance.
        Oversamples the minority class (fraud) in each mini-batch.
        """
        y_np = self.y.numpy()
        class_counts = torch.bincount(self.y).float()
        # Guard: if a class is missing on this client, assign weight 0
        sample_weights = torch.zeros(self.n_samples)
        for i, label in enumerate(y_np):
            if class_counts[label] > 0:
                sample_weights[i] = 1.0 / class_counts[label]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=self.n_samples,
            replacement=True,
        )
        dataset = TensorDataset(self.X.to(self.device), self.y.to(self.device))
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, drop_last=False)

    # ------------------------------------------------------------------
    # FedRep local training
    # ------------------------------------------------------------------

    def _loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy to account for imbalance."""
        # Compute per-batch class weights
        counts = torch.bincount(labels, minlength=2).float().clamp(min=1)
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return nn.functional.cross_entropy(logits, labels, weight=weights)

    def _proximal_term(self, global_enc_state: dict) -> torch.Tensor:
        """FedProx proximal penalty: μ/2 * ||w_enc - w_global||^2."""
        penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.model.encoder.named_parameters():
            global_param = global_enc_state[name].to(self.device)
            penalty = penalty + ((param - global_param) ** 2).sum()
        return (self.mu / 2.0) * penalty

    def _phase_a_head(self, n_steps: int) -> float:
        """Phase A: freeze encoder, update local head only."""
        for p in self.model.encoder.parameters():
            p.requires_grad_(False)
        for p in self.model.head.parameters():
            p.requires_grad_(True)

        opt = optim.Adam(self.model.head.parameters(), lr=self.lr)
        loader_iter = iter(self.dataloader)
        total_loss = 0.0

        for _ in range(n_steps):
            try:
                xb, yb = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.dataloader)
                xb, yb = next(loader_iter)

            opt.zero_grad()
            logits, _ = self.model(xb)
            loss = self._loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        return total_loss / max(n_steps, 1)

    def _phase_b_encoder(self, n_steps: int, global_enc_state: dict) -> float:
        """Phase B: freeze head, update encoder with FedProx penalty."""
        for p in self.model.encoder.parameters():
            p.requires_grad_(True)
        for p in self.model.head.parameters():
            p.requires_grad_(False)

        opt = optim.Adam(self.model.encoder.parameters(), lr=self.lr)
        loader_iter = iter(self.dataloader)
        total_loss = 0.0

        for _ in range(n_steps):
            try:
                xb, yb = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.dataloader)
                xb, yb = next(loader_iter)

            opt.zero_grad()
            logits, _ = self.model(xb)
            task_loss = self._loss_fn(logits, yb)
            prox_loss = self._proximal_term(global_enc_state)
            loss = task_loss + prox_loss
            loss.backward()
            opt.step()
            total_loss += loss.item()

        return total_loss / max(n_steps, 1)

    # ------------------------------------------------------------------
    # Main round interface
    # ------------------------------------------------------------------

    def receive_global_encoder(self, global_state: dict) -> None:
        """Load global encoder weights into local model."""
        self.model.encoder.load_state_dict(copy.deepcopy(global_state))

    def local_train(self) -> Tuple[dict, torch.Tensor, dict]:
        """
        Run one full FedRep round of local training.

        Returns
        -------
        encoder_state : updated encoder state_dict (to send to server)
        dp_encoding   : DP-privatised mean encoding (shape: [encoding_dim])
        metrics       : dict with loss info and class distribution
        """
        global_enc_state = copy.deepcopy(self.model.encoder.state_dict())

        # Phase A — head update
        head_loss = _safe_train(self._phase_a_head, self.head_steps)

        # Phase B — encoder update with FedProx
        enc_loss = _safe_train(self._phase_b_encoder, self.encoder_steps, global_enc_state)

        # Compute DP encodings from the full local dataset
        dp_encoding = self._compute_dp_encoding()

        # Metrics
        metrics = {
            "client_id": self.client_id,
            "n_samples": self.n_samples,
            "fraud_ratio": (self.y == 1).float().mean().item(),
            "head_loss": head_loss,
            "encoder_loss": enc_loss,
        }
        self.train_history.append(metrics)

        return copy.deepcopy(self.model.encoder.state_dict()), dp_encoding, metrics

    def _compute_dp_encoding(self) -> torch.Tensor:
        """
        Encode all local samples and apply DP privatisation.
        Returns a single (encoding_dim,) tensor.
        """
        self.model.eval()
        with torch.no_grad():
            # Process in chunks to avoid OOM on large local datasets
            all_encs = []
            chunk_size = 1024
            for start in range(0, self.n_samples, chunk_size):
                xb = self.X[start: start + chunk_size].to(self.device)
                enc = self.model.encode(xb)
                all_encs.append(enc)
            encodings = torch.cat(all_encs, dim=0)   # (N, D)

        self.model.train()

        if self.dp_layer is not None:
            return self.dp_layer(encodings)           # (D,) privatised mean
        else:
            return encodings.mean(dim=0)              # no DP

    def evaluate(self) -> dict:
        """Evaluate local model on its own data. Returns accuracy and F1."""
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(self.X.to(self.device))
            preds = logits.argmax(dim=1).cpu()
        self.model.train()

        y_np = self.y.numpy()
        p_np = preds.numpy()

        tp = ((p_np == 1) & (y_np == 1)).sum()
        fp = ((p_np == 1) & (y_np == 0)).sum()
        fn = ((p_np == 0) & (y_np == 1)).sum()
        tn = ((p_np == 0) & (y_np == 0)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(len(y_np), 1)

        return {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_train(fn, *args, **kwargs) -> float:
    """Call a training phase function, returning 0.0 on error."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[WARNING] Training phase failed: {e}")
        return 0.0
