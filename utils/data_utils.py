"""
Data Utilities
--------------
load_creditcard      : loads and preprocesses creditcard.csv
split_iid            : equal-block IID split across clients
split_non_iid        : Dirichlet-based non-IID split (optional)
get_client_stats     : prints per-client class distribution
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# Loading and preprocessing
# ------------------------------------------------------------------

def load_creditcard(
    csv_path: str,
    scale_amount_time: bool = True,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load creditcard.csv, standardise features, and return train/test splits.

    The dataset has columns:
        Time, V1..V28 (PCA'd), Amount, Class (0=legit, 1=fraud)

    Parameters
    ----------
    csv_path          : path to creditcard.csv
    scale_amount_time : whether to standardise 'Time' and 'Amount'
                        (V1–V28 are already PCA'd so they're OK as-is)
    test_ratio        : fraction held out as a global test set
    seed              : random seed

    Returns
    -------
    X_train, y_train, X_test, y_test  — all torch tensors (float32 / long)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it in the project root or pass the correct path via --data_path."
        )

    df = pd.read_csv(csv_path)

    # Basic validation
    required_cols = {"Time", "Amount", "Class"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing columns. Expected: {required_cols}")

    print(f"[Data] Loaded {len(df):,} rows. "
          f"Fraud rate: {df['Class'].mean() * 100:.3f}%")

    # Scale 'Time' and 'Amount'
    if scale_amount_time:
        scaler = StandardScaler()
        df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])

    # Features and labels
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Class"].values.astype(np.int64)

    # Stratified train/test split to preserve fraud ratio
    rng = np.random.default_rng(seed)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]

    n_fraud_test = int(len(fraud_idx) * test_ratio)
    n_legit_test = int(len(legit_idx) * test_ratio)

    rng.shuffle(fraud_idx)
    rng.shuffle(legit_idx)

    test_idx = np.concatenate([fraud_idx[:n_fraud_test], legit_idx[:n_legit_test]])
    train_idx = np.concatenate([fraud_idx[n_fraud_test:], legit_idx[n_legit_test:]])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = torch.tensor(X[train_idx])
    y_train = torch.tensor(y[train_idx])
    X_test = torch.tensor(X[test_idx])
    y_test = torch.tensor(y[test_idx])

    print(f"[Data] Train: {len(y_train):,} | Test: {len(y_test):,}")
    print(f"[Data] Train fraud: {y_train.sum().item()} "
          f"({y_train.float().mean().item() * 100:.3f}%)")

    return X_train, y_train, X_test, y_test


# ------------------------------------------------------------------
# IID equal-block split
# ------------------------------------------------------------------

def split_iid(
    X: torch.Tensor,
    y: torch.Tensor,
    num_clients: int,
    seed: int = 42,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split data into `num_clients` equal contiguous blocks.
    Data is shuffled first so each client gets a random mix.

    The dataset (284k rows) divides evenly for most num_clients values;
    remainder rows are dropped so all blocks are exactly equal.

    Returns list of (X_i, y_i) tuples.
    """
    n = len(y)
    rng = torch.Generator()
    rng.manual_seed(seed)
    idx = torch.randperm(n, generator=rng)

    # Truncate to make equal blocks
    block_size = n // num_clients
    idx = idx[:block_size * num_clients]

    splits = []
    for i in range(num_clients):
        start = i * block_size
        end = start + block_size
        client_idx = idx[start:end]
        splits.append((X[client_idx], y[client_idx]))

    return splits


def split_non_iid(
    X: torch.Tensor,
    y: torch.Tensor,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Dirichlet-based non-IID split.
    alpha controls heterogeneity: small alpha -> more skewed per client.

    Returns list of (X_i, y_i) tuples.
    """
    rng = np.random.default_rng(seed)
    y_np = y.numpy()
    classes = np.unique(y_np)
    client_indices = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(y_np == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        # Ensure each client gets at least 1 sample
        proportions = (proportions * len(cls_idx)).astype(int)
        diff = len(cls_idx) - proportions.sum()
        proportions[0] += diff   # fix rounding

        cumsum = np.cumsum([0] + proportions.tolist())
        for i in range(num_clients):
            client_indices[i].extend(cls_idx[cumsum[i]:cumsum[i + 1]].tolist())

    splits = []
    for idxs in client_indices:
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        splits.append((X[idxs], y[idxs]))

    return splits


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

def get_client_stats(splits: List[Tuple[torch.Tensor, torch.Tensor]]) -> pd.DataFrame:
    """Return a DataFrame with per-client sample counts and fraud rates."""
    rows = []
    for i, (X, y) in enumerate(splits):
        n_fraud = y.sum().item()
        rows.append({
            "client": i,
            "n_samples": len(y),
            "n_fraud": int(n_fraud),
            "n_legit": int(len(y) - n_fraud),
            "fraud_pct": round(float(y.float().mean()) * 100, 3),
        })
    df = pd.DataFrame(rows)
    return df


def global_test_loader(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 1024,
) -> torch.utils.data.DataLoader:
    """Simple DataLoader for the held-out global test set."""
    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(X_test, y_test)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
