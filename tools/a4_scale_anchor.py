"""Anchor comparison: per-token absmax scale vs a calibrated per-channel scale.

Pure arithmetic on the same int4 grid, no model: for each seed draw fresh post-FWHT
activations, quantise with (a) absmax/7 per token and (b) one least-squares multiplier
per channel fitted on a calibration batch, then compare the MSE of the dequantised
tensor. The variant with the lower error/variance is the scale the A4 arm uses.
"""

from __future__ import annotations

import statistics

import numpy as np


def main() -> int:
    """Print the scale anchor table over seeds."""
    rng = np.random.default_rng(7)
    absmax_errors: list[float] = []
    channel_errors: list[float] = []
    for _seed in range(12):
        z = rng.standard_normal((64, 128))
        base = np.maximum(np.abs(z).max(axis=-1, keepdims=True), 1e-8) / 7
        q_absmax = np.clip(np.round(z / base), -7, 7)
        err_absmax = float(np.mean((q_absmax * base - z) ** 2))
        # Least-squares multiplier per channel for the SAME codes: a learnable scale
        # moves their magnitude, not the already-chosen code boundaries.
        codes = q_absmax * base
        denom = np.maximum((codes**2).sum(axis=0), 1e-8)
        scale_c = np.clip((codes * z).sum(axis=0) / denom, 0.5, 2.0)
        err_channel = float(np.mean((codes * scale_c - z) ** 2))
        absmax_errors.append(err_absmax)
        channel_errors.append(err_channel)
    print(
        f"absmax:  mean MSE={statistics.fmean(absmax_errors):.6f} "
        f"sd={statistics.stdev(absmax_errors):.6f}"
    )
    print(
        f"channel: mean MSE={statistics.fmean(channel_errors):.6f} "
        f"sd={statistics.stdev(channel_errors):.6f}"
    )
    print(
        "pick:",
        "pc" if statistics.fmean(channel_errors) < statistics.fmean(absmax_errors) else "absmax",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
