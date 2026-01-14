# ml/prototypes_store.py
from __future__ import annotations
from typing import Dict
import numpy as np

# Simple global store (hackathon style)
PROTOTYPES: Dict[str, np.ndarray] = {}
