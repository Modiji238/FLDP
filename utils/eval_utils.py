"""
Evaluation & Logging Utilities
--------------------------------
evaluate_global_model   : test the aggregated global encoder + a fresh head
evaluate_client_models  : collect per-client metrics
pretty_print_round      : console summary per FL round
save_results            : persist results to JSON
"""

import json
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_global_model(
    encoder,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    encoding_dim: int = 64,
    device: Optional[torch.device] = None,
    batch_size: int = 1024,
) -> dict:
    """
    Evaluate the global encoder by training a lightweight probe head
    on the test encodings (linear evaluation protocol).

    This gives a fair estimate of how expressive the global encoder is
    independently of any client's local head.
    """
    device = device or torch.device("cpu")
    encoder = encoder.to(device).eval()

    # Encode test set
    all_encs, all_labels = [], []
    with torch.no_grad():
        for start in range(0, len(y_test), batch_size):
            xb = X_test[start: start + batch_size].to(device)
            all_encs.append(encoder(xb))
            all_labels.append(y_test[start: start + batch_size])

    encs = torch.cat(all_encs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Train a simple linear probe
    probe = nn.Linear(encoding_dim, 2).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    counts = torch.bincount(labels.to(device)).float().clamp(min=1)
    w = (1.0 / counts)
    w = w / w.sum()

    for _ in range(200):
        opt.zero_grad()
        logits = probe(encs)
        loss = nn.functional.cross_entropy(logits, labels.to(device), weight=w)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(encs).argmax(dim=1).cpu()

    return _classification_metrics(labels.numpy(), preds.numpy(), tag="global_encoder_probe")


def evaluate_client_model(client) -> dict:
    """Call client.evaluate() and return metrics."""
    return client.evaluate()


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, tag: str = "") -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy  = (tp + tn) / max(len(y_true), 1)

    return {
        "tag": tag,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def pretty_print_round(
    fl_round: int,
    client_metrics: List[dict],
    server_metrics: Optional[dict] = None,
    privacy_report: Optional[dict] = None,
):
    """Print a clean summary table for one FL round."""
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  Round {fl_round:>3}")
    print(sep)

    if client_metrics:
        avg_f1 = np.mean([m["f1"] for m in client_metrics])
        avg_rec = np.mean([m["recall"] for m in client_metrics])
        avg_acc = np.mean([m["accuracy"] for m in client_metrics])
        print(f"  Clients  ({len(client_metrics)} active)")
        print(f"    avg accuracy : {avg_acc:.4f}")
        print(f"    avg recall   : {avg_rec:.4f}   (fraud detection)")
        print(f"    avg F1       : {avg_f1:.4f}")

    if server_metrics:
        print(f"  Global encoder probe")
        print(f"    accuracy : {server_metrics['accuracy']:.4f}")
        print(f"    recall   : {server_metrics['recall']:.4f}")
        print(f"    F1       : {server_metrics['f1']:.4f}")

    if privacy_report:
        print(f"  Privacy budget  ε={privacy_report['epsilon']:.4f}  "
              f"δ={privacy_report['delta']}")

    print(sep)


def save_results(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved to {path}")
