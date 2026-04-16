"""
Differential Privacy for Encodings
------------------------------------
Applies output perturbation (clip + Gaussian noise) to encoder outputs
before they are shared with the server.

This is encoding-level DP (not DPSGD / gradient-level DP).
Privacy guarantee: (ε, δ)-DP on the encoding space.

Key parameters
--------------
clip_norm  : L2 sensitivity bound (C in the DP literature)
noise_mult : σ = noise_mult * clip_norm  ->  Gaussian noise std
delta      : target δ for privacy accounting
"""

import math
import torch
import numpy as np
from typing import Optional


class DPEncodingLayer:
    """
    Stateless helper that clips and noises a batch of encodings.

    Usage
    -----
        dp = DPEncodingLayer(clip_norm=1.0, noise_mult=1.1)
        noised = dp(encodings)          # during upload
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_mult: float = 1.1,
        delta: float = 1e-5,
        device: Optional[torch.device] = None,
    ):
        if clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        if noise_mult < 0:
            raise ValueError("noise_mult must be non-negative (0 = no noise)")

        self.clip_norm = clip_norm
        self.noise_mult = noise_mult
        self.delta = delta
        self.device = device or torch.device("cpu")
        self._rounds_run = 0

    # ------------------------------------------------------------------
    # Core DP operations
    # ------------------------------------------------------------------

    def clip(self, encodings: torch.Tensor) -> torch.Tensor:
        """
        Per-sample L2 clipping so that ||e_i||_2 <= clip_norm.
        encodings : (N, D)
        """
        norms = encodings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        scale = (self.clip_norm / norms).clamp(max=1.0)
        return encodings * scale

    def add_noise(self, encodings: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise N(0, (noise_mult * clip_norm)^2) to each element.
        Works on aggregated (mean) encodings, shape (D,) or (N, D).
        """
        if self.noise_mult == 0.0:
            return encodings
        sigma = self.noise_mult * self.clip_norm
        noise = torch.randn_like(encodings) * sigma
        return encodings + noise

    def privatise(self, encodings: torch.Tensor) -> torch.Tensor:
        """
        Full DP pipeline: clip individual encodings, average, then noise.
        Suitable for aggregating a client's local encoding batch before upload.

        encodings : (N, D)  — N samples, D encoding dims
        returns   : (D,)    — privatised mean encoding to send to server
        """
        with torch.no_grad():
            clipped = self.clip(encodings)
            mean_enc = clipped.mean(dim=0)       # aggregate across samples
            noised = self.add_noise(mean_enc)
        self._rounds_run += 1
        return noised

    # ------------------------------------------------------------------
    # Privacy accounting (moments accountant approximation)
    # ------------------------------------------------------------------

    def compute_epsilon(self, num_rounds: int, num_samples: int, batch_size: int) -> float:
        """
        Approximate ε using the moments accountant / RDP conversion.
        Uses the simplified Gaussian mechanism formula:
            ε ≈ noise_mult * sqrt(2 * ln(1/δ)) / (q * sqrt(T))
        where q = sampling rate = batch_size / num_samples, T = rounds.

        This is an approximation; for exact accounting use Google's
        dp_accounting library.
        """
        if self.noise_mult == 0:
            return float("inf")
        q = batch_size / max(num_samples, 1)
        T = max(num_rounds, 1)
        # Gaussian mechanism ε via moments accountant approximation
        # Reference: Abadi et al. 2016, "Deep Learning with Differential Privacy"
        eps = (
            math.sqrt(2 * math.log(1.25 / self.delta))
            * self.clip_norm
            / (self.noise_mult * q * math.sqrt(T))
        )
        return eps

    def privacy_report(self, num_rounds: int, num_samples: int, batch_size: int) -> dict:
        eps = self.compute_epsilon(num_rounds, num_samples, batch_size)
        return {
            "epsilon": round(eps, 4),
            "delta": self.delta,
            "clip_norm": self.clip_norm,
            "noise_multiplier": self.noise_mult,
            "rounds": num_rounds,
        }

    def __call__(self, encodings: torch.Tensor) -> torch.Tensor:
        return self.privatise(encodings)

    def __repr__(self):
        return (
            f"DPEncodingLayer(clip_norm={self.clip_norm}, "
            f"noise_mult={self.noise_mult}, delta={self.delta})"
        )
