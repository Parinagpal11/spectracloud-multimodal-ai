# ml/features.py
from __future__ import annotations

import numpy as np

try:
    from scipy.signal import find_peaks  # optional but nice
except Exception:
    find_peaks = None


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def extract_raman_features(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_peaks: int = 10,
    peak_prominence: float = 0.02,
) -> np.ndarray:
    """
    x: wavenumbers (1D)
    y: processed intensity (1D)  -> after baseline/smooth/normalize
    returns: 1D feature vector (float32)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if x.size != y.size or x.size < 10:
        raise ValueError("x and y must be same length and reasonably sized.")

    # Basic stats
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    y_median = float(np.median(y))
    y_p25 = float(np.percentile(y, 25))
    y_p75 = float(np.percentile(y, 75))
    y_iqr = float(y_p75 - y_p25)

    # Shape-ish stats
    dy = np.diff(y)
    abs_dy_mean = float(np.mean(np.abs(dy)))
    abs_dy_std = float(np.std(np.abs(dy)))
    sign_changes = float(np.sum(np.diff(np.sign(dy)) != 0))  # rough “wiggliness”

    # Area / energy
    area = float(np.trapezoid(y, x))
    energy = float(np.sum(y * y))
    l1 = float(np.sum(np.abs(y)))
    l2 = float(np.sqrt(energy))
    l1_over_l2 = _safe_div(l1, l2)

    # Centroid + spread (treat y as weights; shift if negative)
    y_shift = y - y_min
    w_sum = float(np.sum(y_shift))
    centroid = float(np.sum(x * y_shift) / w_sum) if w_sum > 0 else float(np.mean(x))
    spread = float(np.sqrt(np.sum(((x - centroid) ** 2) * y_shift) / w_sum)) if w_sum > 0 else float(np.std(x))

    # Peak features (optional if scipy exists)
    peak_count = 0.0
    peak_positions = np.zeros(max_peaks, dtype=np.float64)
    peak_heights = np.zeros(max_peaks, dtype=np.float64)
    peak_proms = np.zeros(max_peaks, dtype=np.float64)

    if find_peaks is not None:
        # prominence scaled to signal range
        prom = peak_prominence * (y_max - y_min if y_max != y_min else 1.0)
        peaks, props = find_peaks(y, prominence=prom)

        peak_count = float(len(peaks))

        if len(peaks) > 0:
            heights = y[peaks]
            order = np.argsort(heights)[::-1]  # top peaks by height
            peaks = peaks[order]

            k = min(max_peaks, len(peaks))
            idxs = peaks[:k]

            peak_positions[:k] = x[idxs]
            peak_heights[:k] = y[idxs]
            # props['prominences'] aligns with original peaks; reorder similarly
            prominences = props.get("prominences", np.zeros(len(peaks)))
            peak_proms[:k] = prominences[:k]

    # Final vector
    feats = np.concatenate(
        [
            np.array(
                [
                    y_min, y_max, y_mean, y_std, y_median, y_iqr,
                    abs_dy_mean, abs_dy_std, sign_changes,
                    area, energy, l1, l2, l1_over_l2,
                    centroid, spread,
                    peak_count,
                ],
                dtype=np.float64,
            ),
            peak_positions,
            peak_heights,
            peak_proms,
        ]
    )

    return feats.astype(np.float32)
