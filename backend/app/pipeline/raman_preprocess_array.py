# backend/app/pipeline/raman_preprocess_array.py

from __future__ import annotations
import numpy as np


def _moving_average(y: np.ndarray, window: int = 9) -> np.ndarray:
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(ypad, kernel, mode="valid")


def _poly_baseline(x: np.ndarray, y: np.ndarray, deg: int = 3) -> np.ndarray:
    deg = int(np.clip(deg, 1, 8))
    coeff = np.polyfit(x, y, deg=deg)
    baseline = np.polyval(coeff, x)
    return baseline


def preprocess_spectrum_array(
    x: np.ndarray,
    y: np.ndarray,
    *,
    crop: tuple[float, float] | None = (700.0, 1800.0),
    smooth_window: int = 9,
    baseline_deg: int = 3,
    normalize: str = "minmax",  # "minmax" | "zscore" | "none"
) -> dict:
    """
    Basic, dependency-free preprocessing for Raman spectra arrays.
    Returns a dict similar to your CSV pipeline output.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y length mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] < 20:
        raise ValueError("Spectrum must contain at least 20 points")

    # sort by x
    idx = np.argsort(x)
    x = x[idx]
    y_raw = y[idx]

    # crop region if requested
    if crop is not None:
        lo, hi = crop
        mask = (x >= lo) & (x <= hi)
        if mask.sum() >= 20:
            x = x[mask]
            y_raw = y_raw[mask]

    # denoise/smooth
    y_smooth = _moving_average(y_raw, window=smooth_window)

    # baseline correction
    baseline = _poly_baseline(x, y_smooth, deg=baseline_deg)
    y_bc = y_smooth - baseline

    # normalize
    y_processed = y_bc.copy()
    if normalize == "minmax":
        mn, mx = float(np.min(y_processed)), float(np.max(y_processed))
        denom = (mx - mn) if (mx - mn) != 0 else 1.0
        y_processed = (y_processed - mn) / denom
    elif normalize == "zscore":
        mu, sd = float(np.mean(y_processed)), float(np.std(y_processed))
        y_processed = (y_processed - mu) / (sd if sd != 0 else 1.0)
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be one of: minmax, zscore, none")

    return {
        "x": x,
        "y_raw": y_raw,
        "y_processed": y_processed,
    }
