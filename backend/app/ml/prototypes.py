# ml/prototypes.py
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / (n + eps)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))

def build_class_prototypes(
    features_by_label: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    features_by_label: label -> array of shape (num_samples, feature_dim)
    returns: label -> prototype vector (feature_dim,)
    """
    prototypes: Dict[str, np.ndarray] = {}
    for label, feats in features_by_label.items():
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] == 0:
            raise ValueError(f"Bad feats for label={label}, shape={feats.shape}")
        proto = feats.mean(axis=0)
        prototypes[label] = l2_normalize(proto)
    return prototypes

def predict_with_prototypes(
    feature_vec: np.ndarray,
    prototypes: Dict[str, np.ndarray],
) -> Tuple[str, float, Dict[str, float]]:
    """
    returns:
      best_label, best_score, all_scores (cosine similarity)
    """
    if not prototypes:
        raise ValueError("No prototypes loaded/built.")

    scores: Dict[str, float] = {}
    for label, proto in prototypes.items():
        scores[label] = cosine_sim(feature_vec, proto)

    best_label = max(scores, key=scores.get)
    best_score = float(scores[best_label])
    return best_label, best_score, scores
