import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def spectrum_plot_base64(
    x: np.ndarray,
    y_raw: np.ndarray,
    y_processed: np.ndarray,
    max_points: int = 4000
) -> str:
    if not (len(x) == len(y_raw) == len(y_processed)):
        raise ValueError("Spectrum arrays must have the same length")

    # Downsample large spectra
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        x = x[idx]
        y_raw = y_raw[idx]
        y_processed = y_processed[idx]

    fig = plt.figure(figsize=(7, 4))
    plt.plot(x, y_raw, label="raw", linewidth=1)
    plt.plot(x, y_processed, label="processed", linewidth=1)
    plt.xlabel("Wavelength / Raman shift")
    plt.ylabel("Intensity")
    plt.title("Spectrum: Raw vs Processed")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")
