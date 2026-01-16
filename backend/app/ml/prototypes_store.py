# backend/app/ml/prototypes_store.py
import json
from pathlib import Path
import numpy as np

PROTOTYPES = {}

def load_prototypes() -> dict:
    path = Path(__file__).parent / "prototypes.json"
    if not path.exists():
        return {}

    with open(path, "r") as f:
        raw = json.load(f)

    # expected structure: {"labels": [...], "feature_dim": 47, "N": ..., "prototypes": {...}}
    protos_raw = raw.get("prototypes", {})

    out = {}
    for k, v in protos_raw.items():
        out[str(k)] = np.array(v, dtype=float)

    return out

# load at import time
PROTOTYPES = load_prototypes()
