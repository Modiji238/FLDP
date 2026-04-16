"""
================================================================================
  FedAvg + Differential Privacy (DP) Baseline
  Dataset : creditcard.csv  (Kaggle Credit Card Fraud Detection)
  Supports: IID  and  Non-IID  data splits
  Compare  : use logged metrics against your FedRep+FedProx+DP results
================================================================================
"""

import copy
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score
)

# ── reproducibility ─────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ================================================================================
#  CONFIGURATION  –  tweak here to match your FedRep+FedProx+DP run
# ================================================================================
CONFIG = {
    # --- federated ---
    "num_clients"       : 5,
    "num_rounds"        : 20,
    "local_epochs"      : 5,
    "local_lr"          : 1e-3,
    "local_batch_size"  : 256,
    "fraction_fit"      : 1.0,        # fraction of clients sampled each round

    # --- data ---
    "data_path"         : "creditcard.csv",
    "test_size"         : 0.2,
    "val_size"          : 0.1,        # fraction of train used for server validation
    "iid"               : True,       # set False for Non-IID (Dirichlet) split

    # Non-IID Dirichlet concentration (lower → more heterogeneous)
    "dirichlet_alpha"   : 0.5,

    # --- differential privacy (Gaussian mechanism, per-round) ---
    "dp_enabled"        : True,
    "dp_clip_norm"      : 1.0,        # per-sample gradient clip (L2)
    "dp_noise_multiplier": 1.1,       # σ = noise_multiplier * clip_norm / batch_size
    "dp_delta"          : 1e-5,

    # --- model ---
    "hidden_dims"       : [128, 64, 32],
    "dropout"           : 0.3,

    # --- misc ---
    "device"            : "cuda" if torch.cuda.is_available() else "cpu",
}
# ================================================================================


# ────────────────────────────────────────────────────────────────────────────────
#  1.  MODEL
# ────────────────────────────────────────────────────────────────────────────────
class FraudDetectionNet(nn.Module):
    """Simple MLP for binary fraud classification."""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            # LayerNorm instead of BatchNorm1d: stable with any batch size,
            # and does NOT produce NaN when DP noise perturbs weights early on.
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))   # single logit → BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────────────
#  2.  DATA LOADING & SPLIT
# ────────────────────────────────────────────────────────────────────────────────
def load_creditcard(path: str, test_size: float, val_size: float):
    df = pd.read_csv(path)
    print(f"[Data] Loaded {len(df):,} rows | Fraud rate: {df['Class'].mean()*100:.3f}%")

    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = df["Class"].values.astype(np.float32)

    # scale Amount and Time (V1-V28 already PCA-scaled)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=SEED
    )

    def to_ds(Xa, ya):
        return TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya))

    return to_ds(X_train, y_train), to_ds(X_val, y_val), to_ds(X_test, y_test)


def iid_split(dataset: TensorDataset, num_clients: int):
    """Randomly shuffle and partition equally."""
    n = len(dataset)
    idx = np.random.permutation(n)
    splits = np.array_split(idx, num_clients)
    return [Subset(dataset, s.tolist()) for s in splits]


def non_iid_split(dataset: TensorDataset, num_clients: int, alpha: float):
    """Dirichlet-based label-heterogeneous split."""
    labels = dataset.tensors[1].numpy().astype(int)
    class_indices = defaultdict(list)
    for i, l in enumerate(labels):
        class_indices[l].append(i)

    client_indices = [[] for _ in range(num_clients)]
    for cls, idxs in class_indices.items():
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * len(idxs)).astype(int)
        proportions[-1] = len(idxs) - proportions[:-1].sum()   # fix rounding
        start = 0
        for c, cnt in enumerate(proportions):
            client_indices[c].extend(idxs[start:start + cnt])
            start += cnt

    return [Subset(dataset, idxs) for idxs in client_indices]


# ────────────────────────────────────────────────────────────────────────────────
#  3.  DIFFERENTIAL PRIVACY UTILITIES
# ────────────────────────────────────────────────────────────────────────────────
def clip_gradients(model: nn.Module, clip_norm: float):
    """Clip each parameter's gradient to L2 norm ≤ clip_norm."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)


def add_gaussian_noise_to_model(model: nn.Module, noise_std: float, device: str):
    """Add i.i.d. Gaussian noise to every parameter (simulates DP aggregation noise)."""
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * noise_std)


def compute_noise_std(noise_multiplier: float, clip_norm: float, num_clients: int):
    """
    σ = noise_multiplier × clip_norm  (applied once at the server after averaging).
    Scaled by 1/num_clients because we average – equivalent to Gaussian mechanism.
    """
    return noise_multiplier * clip_norm / num_clients


# ────────────────────────────────────────────────────────────────────────────────
#  4.  CLIENT
# ────────────────────────────────────────────────────────────────────────────────
class FedAvgClient:
    def __init__(self, client_id: int, dataset: Subset, cfg: dict):
        self.id  = client_id
        self.cfg = cfg
        self.device = cfg["device"]

        # pos_weight for class imbalance
        labels = dataset.dataset.tensors[1][dataset.indices].numpy()
        neg, pos = (labels == 0).sum(), (labels == 1).sum()
        pw = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.loader = DataLoader(
            dataset,
            batch_size=cfg["local_batch_size"],
            shuffle=True,
            drop_last=False,
        )

    def train(self, global_weights: dict) -> dict:
        """Run local SGD and return updated weights."""
        model = FraudDetectionNet(
            self.cfg["input_dim"],
            self.cfg["hidden_dims"],
            self.cfg["dropout"],
        ).to(self.device)
        model.load_state_dict(global_weights)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=self.cfg["local_lr"])

        for _ in range(self.cfg["local_epochs"]):
            for X_batch, y_batch in self.loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss   = self.criterion(logits, y_batch)
                loss.backward()

                # ── DP: clip gradients per-batch ───────────────────────────
                if self.cfg["dp_enabled"]:
                    clip_gradients(model, self.cfg["dp_clip_norm"])

                optimizer.step()

        return model.state_dict()


# ────────────────────────────────────────────────────────────────────────────────
#  5.  SERVER  –  FedAvg aggregation + DP noise injection
# ────────────────────────────────────────────────────────────────────────────────
class FedAvgServer:
    def __init__(self, global_model: nn.Module, cfg: dict):
        self.model  = global_model
        self.cfg    = cfg
        self.device = cfg["device"]
        self.history = []        # list of per-round metric dicts

    def aggregate(self, client_weights: list, client_sizes: list) -> None:
        """Weighted average of client model weights."""
        total = sum(client_sizes)
        avg_weights = copy.deepcopy(client_weights[0])

        for key in avg_weights:
            avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)
            for w, sz in zip(client_weights, client_sizes):
                avg_weights[key] += w[key].float() * (sz / total)

        # ── DP: add calibrated Gaussian noise to aggregated model ──────────
        if self.cfg["dp_enabled"]:
            noise_std = compute_noise_std(
                self.cfg["dp_noise_multiplier"],
                self.cfg["dp_clip_norm"],
                len(client_weights),
            )
            for key in avg_weights:
                avg_weights[key] += torch.randn_like(avg_weights[key]) * noise_std

        self.model.load_state_dict(avg_weights)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "val") -> dict:
        self.model.eval()
        all_logits, all_labels = [], []
        for X, y in loader:
            X = X.to(self.device)
            logits = self.model(X).cpu()
            all_logits.append(logits)
            all_labels.append(y)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels).numpy()

        # Guard: clamp logits to a finite range before sigmoid so that
        # early-round DP noise (which can make weights very large) never
        # produces NaN / Inf probabilities that crash sklearn metrics.
        logits = torch.clamp(logits, min=-30.0, max=30.0)
        probs  = torch.sigmoid(logits).numpy()

        # Extra safety: replace any residual NaN/Inf with 0.5 (uncertain)
        if not np.isfinite(probs).all():
            n_bad = (~np.isfinite(probs)).sum()
            print(f"  [WARN] {n_bad} non-finite prob(s) in {split} – replacing with 0.5")
            probs = np.where(np.isfinite(probs), probs, 0.5)

        preds = (probs >= 0.5).astype(int)

        metrics = {
            "split"     : split,
            "roc_auc"   : roc_auc_score(labels, probs),
            "pr_auc"    : average_precision_score(labels, probs),
            "f1"        : f1_score(labels, preds, zero_division=0),
            "accuracy"  : (preds == labels).mean(),
        }
        return metrics


# ────────────────────────────────────────────────────────────────────────────────
#  6.  FEDERATED TRAINING LOOP
# ────────────────────────────────────────────────────────────────────────────────
def federated_train(cfg: dict):
    device = cfg["device"]
    print(f"\n{'='*70}")
    print(f"  FedAvg + DP  |  {'IID' if cfg['iid'] else 'Non-IID'}  |  {cfg['num_clients']} clients  |  {cfg['num_rounds']} rounds")
    print(f"  DP enabled={cfg['dp_enabled']}  clip={cfg['dp_clip_norm']}  noise_mult={cfg['dp_noise_multiplier']}")
    print(f"{'='*70}\n")

    # ---------- data ----------
    train_ds, val_ds, test_ds = load_creditcard(
        cfg["data_path"], cfg["test_size"], cfg["val_size"]
    )
    cfg["input_dim"] = train_ds.tensors[0].shape[1]

    if cfg["iid"]:
        client_datasets = iid_split(train_ds, cfg["num_clients"])
    else:
        client_datasets = non_iid_split(train_ds, cfg["num_clients"], cfg["dirichlet_alpha"])

    # print client distribution
    for i, ds in enumerate(client_datasets):
        labels = ds.dataset.tensors[1][ds.indices].numpy()
        print(f"  Client {i}: {len(ds):5d} samples | fraud={labels.mean()*100:.2f}%")

    val_loader  = DataLoader(val_ds,  batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    # ---------- global model ----------
    global_model = FraudDetectionNet(
        cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"]
    ).to(device)

    server  = FedAvgServer(global_model, cfg)
    clients = [FedAvgClient(i, ds, cfg) for i, ds in enumerate(client_datasets)]

    history = []

    # ---------- rounds ----------
    for rnd in range(1, cfg["num_rounds"] + 1):
        # sample fraction of clients
        m = max(1, int(cfg["fraction_fit"] * cfg["num_clients"]))
        selected = np.random.choice(cfg["num_clients"], m, replace=False)

        global_weights = copy.deepcopy(global_model.state_dict())

        round_weights = []
        round_sizes   = []
        for cid in selected:
            w = clients[cid].train(global_weights)
            round_weights.append(w)
            round_sizes.append(len(clients[cid].loader.dataset))

        # aggregate + inject DP noise
        server.aggregate(round_weights, round_sizes)

        # evaluate on validation set every round
        val_metrics = server.evaluate(val_loader, split="val")
        val_metrics["round"] = rnd
        history.append(val_metrics)

        if rnd % 5 == 0 or rnd == 1:
            print(
                f"  Round {rnd:3d}/{cfg['num_rounds']} | "
                f"ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                f"PR-AUC:  {val_metrics['pr_auc']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f}"
            )

    # ---------- final test evaluation ----------
    print(f"\n{'─'*70}")
    test_metrics = server.evaluate(test_loader, split="test")
    print(f"  FINAL TEST RESULTS")
    print(f"  ROC-AUC : {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC  : {test_metrics['pr_auc']:.4f}")
    print(f"  F1      : {test_metrics['f1']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    # full classification report
    global_model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            logits = global_model(X.to(device)).cpu()
            all_probs.append(torch.sigmoid(logits).numpy())
            all_labels.append(y.numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs >= 0.5).astype(int)

    print(f"\n  Classification Report:\n")
    print(classification_report(labels, preds, target_names=["Legit", "Fraud"], digits=4))
    print(f"  Confusion Matrix:\n{confusion_matrix(labels, preds)}\n")

    return history, test_metrics, global_model


# ────────────────────────────────────────────────────────────────────────────────
#  7.  MAIN  –  run IID then Non-IID and print comparison table
# ────────────────────────────────────────────────────────────────────────────────
def run_comparison():
    results = {}

    for split_type in ["IID", "Non-IID"]:
        cfg = copy.deepcopy(CONFIG)
        cfg["iid"] = (split_type == "IID")
        history, test_metrics, _ = federated_train(cfg)
        results[split_type] = test_metrics

    print("\n" + "="*70)
    print("  COMPARISON TABLE  –  FedAvg + DP")
    print("="*70)
    print(f"  {'Metric':<15} {'IID':>12} {'Non-IID':>12}")
    print("  " + "-"*40)
    for metric in ["roc_auc", "pr_auc", "f1", "accuracy"]:
        iid_val    = results["IID"][metric]
        noniid_val = results["Non-IID"][metric]
        print(f"  {metric:<15} {iid_val:>12.4f} {noniid_val:>12.4f}")
    print("="*70)
    print("\n  TIP: Compare these numbers with your FedRep+FedProx+DP results.")
    print("  Metrics: ROC-AUC / PR-AUC (best for imbalanced data), F1, Accuracy.\n")


# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # --- Run both splits and compare ---
    run_comparison()

    # --- Or run a single split ---
    # history, test_metrics, model = federated_train(CONFIG)