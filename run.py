"""
run.py  —  FedRep + Differential Privacy + FedProx  (Credit Card Fraud)
=========================================================================

Single-file entry point that runs both the server and all clients
in a simulated federated loop on a single machine.

Usage
-----
    python run.py --data_path creditcard.csv

Key flags (all have sensible defaults — just change --data_path):
    --data_path      Path to creditcard.csv
    --num_clients    Number of simulated clients         [default: 5]
    --num_rounds     FL communication rounds             [default: 20]
    --split          iid | non_iid                       [default: iid]
    --encoding_dim   Encoder output dimension            [default: 64]
    --head_steps     Local head gradient steps/round     [default: 10]
    --encoder_steps  Local encoder gradient steps/round  [default: 10]
    --mu             FedProx proximal coefficient        [default: 0.01]
    --lr             Local learning rate                 [default: 1e-3]
    --batch_size     Local mini-batch size               [default: 256]
    --clip_norm      DP clip norm (C)                    [default: 1.0]
    --noise_mult     DP noise multiplier (σ/C)           [default: 1.1]
    --delta          DP delta                            [default: 1e-5]
    --no_dp          Disable differential privacy
    --eval_every     Evaluate every N rounds             [default: 1]
    --save_dir       Directory for checkpoints/results   [default: output]
    --seed           Global random seed                  [default: 42]
    --device         cpu | cuda | mps                    [default: auto]

Architecture summary
--------------------
  Each client owns:
    SharedEncoder (global)  — weights broadcast/collected each round
    LocalHead    (private)  — never leaves client
    DPEncodingLayer         — clips + noises encoder outputs before upload

  Server owns:
    FedProxServer           — aggregates encoder weights via weighted avg
                              (FedProx proximal penalty applied client-side)

  Per round:
    1. Server broadcasts global encoder weights.
    2. Each client runs FedRep:
         Phase A: train local head (encoder frozen)
         Phase B: train encoder with FedProx penalty (head frozen)
    3. Each client computes DP mean encoding of local data.
    4. Server collects updated encoder weights + (optionally) DP encodings.
    5. Server runs FedProx aggregation → new global encoder.
    6. Optionally evaluate on held-out global test set.
"""

import argparse
import os
import sys
import time
from typing import List, Optional

import torch
import numpy as np

# Make sure project root is on path when run from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import SharedEncoder, FedRepModel
from models.dp_layer import DPEncodingLayer
from models.server import FedProxServer
from models.client import FedRepClient
from utils.data_utils import (
    load_creditcard,
    split_iid,
    split_non_iid,
    get_client_stats,
)
from utils.eval_utils import (
    evaluate_global_model,
    evaluate_client_model,
    pretty_print_round,
    save_results,
)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FedRep + DP + FedProx — Credit Card Fraud Detection"
    )

    # Data
    p.add_argument("--data_path", type=str, default="creditcard.csv",
                   help="Path to creditcard.csv")
    p.add_argument("--split", type=str, default="iid", choices=["iid", "non_iid"],
                   help="Data split strategy across clients")
    p.add_argument("--dirichlet_alpha", type=float, default=0.5,
                   help="Dirichlet alpha for non-IID split (lower = more heterogeneous)")
    p.add_argument("--test_ratio", type=float, default=0.2,
                   help="Global test set fraction")

    # Federation
    p.add_argument("--num_clients", type=int, default=5,
                   help="Number of federated clients")
    p.add_argument("--num_rounds", type=int, default=20,
                   help="Number of FL communication rounds")
    p.add_argument("--fraction_fit", type=float, default=1.0,
                   help="Fraction of clients that participate per round (1.0 = all)")

    # Model
    p.add_argument("--encoding_dim", type=int, default=64,
                   help="Encoder output dimension")
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 96],
                   help="Encoder hidden layer sizes")

    # Training
    p.add_argument("--head_steps", type=int, default=10,
                   help="Local head gradient steps per round (Phase A)")
    p.add_argument("--encoder_steps", type=int, default=10,
                   help="Encoder gradient steps per round (Phase B)")
    p.add_argument("--mu", type=float, default=0.01,
                   help="FedProx proximal coefficient")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Local learning rate")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Local mini-batch size")

    # Differential Privacy
    p.add_argument("--no_dp", action="store_true",
                   help="Disable differential privacy (baseline)")
    p.add_argument("--clip_norm", type=float, default=1.0,
                   help="DP L2 clip norm (C)")
    p.add_argument("--noise_mult", type=float, default=1.1,
                   help="DP noise multiplier (σ = noise_mult * clip_norm)")
    p.add_argument("--delta", type=float, default=1e-5,
                   help="DP target delta")

    # Misc
    p.add_argument("--eval_every", type=int, default=1,
                   help="Evaluate global encoder every N rounds")
    p.add_argument("--save_dir", type=str, default="output",
                   help="Directory for checkpoints and results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto | cpu | cuda | mps")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_clients(clients: List[FedRepClient], fraction: float, seed: int) -> List[FedRepClient]:
    """Randomly select a fraction of clients to participate this round."""
    k = max(1, int(len(clients) * fraction))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(clients), size=k, replace=False)
    return [clients[i] for i in sorted(indices)]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "═" * 70)
    print("  FedRep + Differential Privacy + FedProx")
    print("  Credit Card Fraud Detection")
    print("═" * 70)
    print(f"  Device      : {device}")
    print(f"  Clients     : {args.num_clients}")
    print(f"  Rounds      : {args.num_rounds}")
    print(f"  DP          : {'OFF (baseline)' if args.no_dp else 'ON'}")
    if not args.no_dp:
        print(f"  clip_norm   : {args.clip_norm}  |  noise_mult : {args.noise_mult}  |  δ={args.delta}")
    print(f"  FedProx μ   : {args.mu}")
    print(f"  Split       : {args.split}")
    print()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test = load_creditcard(
        args.data_path,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    input_dim = X_train.shape[1]

    # ── 2. Split across clients ───────────────────────────────────────────────
    if args.split == "iid":
        splits = split_iid(X_train, y_train, args.num_clients, seed=args.seed)
    else:
        splits = split_non_iid(
            X_train, y_train, args.num_clients,
            alpha=args.dirichlet_alpha, seed=args.seed
        )

    stats_df = get_client_stats(splits)
    print("[Data] Per-client distribution:")
    print(stats_df.to_string(index=False))
    print()

    # ── 3. DP layer ───────────────────────────────────────────────────────────
    dp_layer = None if args.no_dp else DPEncodingLayer(
        clip_norm=args.clip_norm,
        noise_mult=args.noise_mult,
        delta=args.delta,
        device=device,
    )

    # ── 4. Global server ──────────────────────────────────────────────────────
    global_encoder = SharedEncoder(
        input_dim=input_dim,
        encoding_dim=args.encoding_dim,
        hidden_dims=args.hidden_dims,
    )
    server = FedProxServer(global_encoder=global_encoder, mu=args.mu, device=device)

    # ── 5. Clients ────────────────────────────────────────────────────────────
    clients: List[FedRepClient] = []
    for i, (Xi, yi) in enumerate(splits):
        client = FedRepClient(
            client_id=i,
            X=Xi,
            y=yi,
            input_dim=input_dim,
            encoding_dim=args.encoding_dim,
            hidden_dims=args.hidden_dims,
            dp_layer=dp_layer,
            mu=args.mu,
            lr=args.lr,
            batch_size=args.batch_size,
            head_steps=args.head_steps,
            encoder_steps=args.encoder_steps,
            device=device,
        )
        clients.append(client)

    print(f"[Init] {len(clients)} clients created.\n")

    # ── 6. FL training loop ───────────────────────────────────────────────────
    all_results = {
        "config": vars(args),
        "rounds": [],
    }
    round_seed_base = args.seed * 1000

    for fl_round in range(1, args.num_rounds + 1):
        t0 = time.time()

        # Select participating clients
        active_clients = select_clients(clients, args.fraction_fit, round_seed_base + fl_round)

        # Broadcast global encoder weights
        global_weights = server.get_global_weights()
        for client in active_clients:
            client.receive_global_encoder(global_weights)

        # Local training
        updated_weights = []
        client_sizes = []
        round_metrics = []

        for client in active_clients:
            enc_state, _dp_enc, metrics = client.local_train()
            updated_weights.append(enc_state)
            client_sizes.append(client.n_samples)
            round_metrics.append(metrics)

        # FedProx aggregation
        server.aggregate(updated_weights, client_sizes)

        # Evaluation
        server_metrics = None
        if fl_round % args.eval_every == 0:
            server_metrics = evaluate_global_model(
                encoder=server.global_encoder,
                X_test=X_test,
                y_test=y_test,
                encoding_dim=args.encoding_dim,
                device=device,
            )
            client_eval_metrics = [evaluate_client_model(c) for c in active_clients]
        else:
            client_eval_metrics = []

        # Privacy accounting
        privacy = None
        if dp_layer is not None:
            privacy = dp_layer.privacy_report(
                num_rounds=fl_round,
                num_samples=len(y_train) // args.num_clients,
                batch_size=args.batch_size,
            )

        elapsed = time.time() - t0
        pretty_print_round(fl_round, client_eval_metrics, server_metrics, privacy)
        print(f"  Round time: {elapsed:.1f}s")

        # Record
        all_results["rounds"].append({
            "round": fl_round,
            "train_metrics": round_metrics,
            "eval_metrics": client_eval_metrics,
            "global_metrics": server_metrics,
            "privacy": privacy,
            "elapsed_s": round(elapsed, 2),
        })

    # ── 7. Final evaluation & save ────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  Final Evaluation")
    print("═" * 70)

    final_global = evaluate_global_model(
        encoder=server.global_encoder,
        X_test=X_test,
        y_test=y_test,
        encoding_dim=args.encoding_dim,
        device=device,
    )
    print(f"  Global encoder  F1={final_global['f1']:.4f}  "
          f"Recall={final_global['recall']:.4f}  "
          f"Precision={final_global['precision']:.4f}")

    all_client_final = []
    for client in clients:
        # Give each client final global encoder, then re-evaluate
        client.receive_global_encoder(server.get_global_weights())
        m = evaluate_client_model(client)
        all_client_final.append(m)
        print(f"  Client {client.client_id:>2}  "
              f"F1={m['f1']:.4f}  Recall={m['recall']:.4f}  "
              f"n_fraud={client.y.sum().item()}")

    if dp_layer is not None:
        final_privacy = dp_layer.privacy_report(
            num_rounds=args.num_rounds,
            num_samples=len(y_train) // args.num_clients,
            batch_size=args.batch_size,
        )
        print(f"\n  Privacy budget: ε={final_privacy['epsilon']:.4f}  δ={final_privacy['delta']}")

    # Save checkpoint
    ckpt_path = os.path.join(args.save_dir, "global_encoder.pt")
    server.save(ckpt_path)
    print(f"\n[Checkpoint] Global encoder saved to {ckpt_path}")

    # Save results JSON
    all_results["final"] = {
        "global": final_global,
        "clients": all_client_final,
    }
    save_results(all_results, os.path.join(args.save_dir, "results.json"))
    print("Done.")


if __name__ == "__main__":
    main()
