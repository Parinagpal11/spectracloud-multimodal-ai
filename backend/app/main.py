# backend/app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import numpy as np
import json
from pathlib import Path

from app.pipeline.raman_preprocess import load_spectrum_csv, preprocess_spectrum
from app.utils.plots import spectrum_plot_base64
from app.ml.features import extract_raman_features
from app.ml.prototypes_store import PROTOTYPES
from app.ml.prototypes import predict_with_prototypes
from typing import Optional
from app.ml.prototypes_store import PROTOTYPES
from app.ml.prototypes import predict_with_prototypes

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spectracloud")

# ---------------- app ----------------
app = FastAPI(title="SpectraCloud API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
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

def _ensure_image(file: Optional[UploadFile]):
    if file is None:
        return
    if not (file.filename or "").lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Image must be PNG/JPG")

# ---------------- analyze ----------------
@app.post("/analyze")
async def analyze(
    spectrum: UploadFile = File(...),
    image: Optional[UploadFile] = File(...)
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

        # ---- features ----
        features = extract_raman_features(
            processed["x"],
            processed["y_processed"]
        )

        # ---- prototypes scoring ----
        label = None
        prototype_score = None
        all_scores = None

        if PROTOTYPES:
            label, prototype_score, all_scores = predict_with_prototypes(features, PROTOTYPES)
            logger.info(f"Loaded prototypes in API: {len(PROTOTYPES)}")

        # ---- plot ----
        plot_b64 = spectrum_plot_base64(
            processed["x"],
            processed["y_raw"],
            processed["y_processed"]
        )

        # ---- confidence ----
        # If prototype_score exists (cosine sim ~ [-1, 1]), map it to [0.5, 0.95]
        if prototype_score is not None:
            # map [-1, 1] -> [0, 1]
            conf01 = (float(prototype_score) + 1.0) / 2.0
            confidence = max(0.50, min(0.95, conf01))
        else:
            # fallback if prototypes not loaded
            y = processed["y_processed"]
            snr_like = float(np.std(y) / (np.mean(np.abs(np.diff(y))) + 1e-9))
            confidence = max(0.50, min(0.95, 0.50 + 0.10 * (snr_like / 10.0)))

        return {
            "status": "success",

            # keep this as a placeholder until you map labels -> real classes
            "prediction": "normal",

            "confidence": round(float(confidence), 3),
            "spectrum_plot_png_base64": plot_b64,

            # prototype-based output
            "feature_dim": int(features.shape[0]),
            "label": label,  # None if prototypes not loaded
            "label_display": LABEL_MAP.get(str(label), str(label)) if label is not None else None,
            "prototype_score": None if prototype_score is None else round(float(prototype_score), 6),

            # debug (remove later if you want)
            "all_scores": all_scores,

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
    LABEL_MAP_PATH = Path(__file__).parent / "ml" / "label_map.json"

LABEL_MAP = {}

def _load_label_map() -> dict:
    try:
        label_map_path = Path(__file__).parent / "ml" / "label_map.json"
        if label_map_path.exists():
            return json.loads(label_map_path.read_text())
    except Exception as e:
        logger.warning(f"label_map.json not loaded: {e}")
    return {}

LABEL_MAP = _load_label_map()


