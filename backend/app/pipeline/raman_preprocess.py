import io
import numpy as np
import pandas as pd

# ---------------- CSV loader ----------------
def load_spectrum_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))

    cols = {c.lower().strip(): c for c in df.columns}
    x_candidates = ["wavelength", "wavenumber", "ramanshift", "raman_shift", "shift", "x"]
    y_candidates = ["intensity", "y", "signal"]

    x_col = next((cols[k] for k in x_candidates if k in cols), None)
    y_col = next((cols[k] for k in y_candidates if k in cols), None)

    if x_col is None or y_col is None:
        raise ValueError(
            f"CSV must include wavelength/wavenumber + intensity columns. "
            f"Found columns: {list(df.columns)}"
        )

    out = df[[x_col, y_col]].copy()
    out.columns = ["x", "y"]
    out = out.dropna()
    return out

# ---------------- preprocessing helpers ----------------
def moving_average(y: np.ndarray, window: int = 11) -> np.ndarray:
    if window < 3:
        return y
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")

def baseline_subtract(y: np.ndarray, poly_deg: int = 3) -> np.ndarray:
    x = np.arange(len(y))
    coeff = np.polyfit(x, y, deg=poly_deg)
    baseline = np.polyval(coeff, x)
    return y - baseline

def normalize_minmax(y: np.ndarray) -> np.ndarray:
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if ymax - ymin < 1e-12:
        return y
    return (y - ymin) / (ymax - ymin)

# ---------------- main preprocess ----------------
def preprocess_spectrum(df: pd.DataFrame) -> dict:
    x = df["x"].to_numpy(dtype=float)
    y_raw = df["y"].to_numpy(dtype=float)

    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y_raw)):
        raise ValueError("Spectrum contains NaN or Inf values")

    if np.allclose(y_raw, y_raw[0]):
        raise ValueError("Spectrum intensity is constant (no signal)")

    if np.any(np.diff(x) == 0):
        raise ValueError("Spectrum x-axis contains duplicate values")

    y_smooth = moving_average(y_raw, window=11)
    y_base = baseline_subtract(y_smooth, poly_deg=3)
    y_norm = normalize_minmax(y_base)

    return {
        "x": x,
        "y_raw": y_raw,
        "y_processed": y_norm,
    }
