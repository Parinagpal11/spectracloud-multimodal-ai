from backend.app.ml.prototypes_store import PROTOTYPES

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import numpy as np

from app.pipeline.raman_preprocess import load_spectrum_csv, preprocess_spectrum
from app.utils.plots import spectrum_plot_base64
from app.ml.features import extract_raman_features
from backend.app.ml.prototypes import predict_with_prototypes  # ✅ correct import


# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spectracloud")

# ---------------- app ----------------
app = FastAPI(title="SpectraCloud API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok", "message": "SpectraCloud backend running"}

# ---------------- helpers ----------------
def _ensure_csv(file: UploadFile):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Spectrum must be a .csv file")

def _ensure_image(file: UploadFile):
    if not (file.filename or "").lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Image must be PNG/JPG")

# ---------------- analyze ----------------
@app.post("/analyze")
async def analyze(
    spectrum: UploadFile = File(...),
    image: UploadFile = File(...)
):
    try:
        # ---- validate inputs ----
        _ensure_csv(spectrum)
        _ensure_image(image)

        spectrum_bytes = await spectrum.read()
        if not spectrum_bytes:
            raise HTTPException(status_code=400, detail="Spectrum CSV is empty")

        logger.info(f"Received spectrum={spectrum.filename}, image={image.filename}")

        # ---- load + preprocess spectrum ----
        df = load_spectrum_csv(spectrum_bytes)

        if len(df) < 20:
            raise HTTPException(
                status_code=400,
                detail="Spectrum CSV must contain at least 20 rows"
            )

        processed = preprocess_spectrum(df)

        # ---- Step 4.2: extract features ----
        features = extract_raman_features(
            processed["x"],
            processed["y_processed"]
        )

        # ---- Step 4.3: prototypes scoring ----
        label = None
        prototype_score = None
        all_scores = None

        if PROTOTYPES:
            label, prototype_score, all_scores = predict_with_prototypes(features, PROTOTYPES)

        # ---- generate plot ----
        plot_b64 = spectrum_plot_base64(
            processed["x"],
            processed["y_raw"],
            processed["y_processed"]
        )

        # ---- deterministic confidence stub (NOT ML yet) ----
        y = processed["y_processed"]
        snr_like = float(np.std(y) / (np.mean(np.abs(np.diff(y))) + 1e-9))
        confidence = max(0.50, min(0.95, 0.50 + 0.10 * (snr_like / 10.0)))

        # ✅ Response: includes 4.3 fields + keeps your old keys
        return {
            "status": "success",
            "prediction": "normal",     # placeholder until ML is added
            "confidence": round(confidence, 3),

            "spectrum_plot_png_base64": plot_b64,

            # ✅ Step 4.3 debug + wiring
            "feature_dim": int(features.shape[0]),
            "label": label,  # None if prototypes not loaded yet
            "prototype_score": prototype_score,  # cosine similarity [-1, 1]
            "all_scores": all_scores,  # remove later if you want

            "received": {
                "spectrum_filename": spectrum.filename,
                "image_filename": image.filename,
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected server error")
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
