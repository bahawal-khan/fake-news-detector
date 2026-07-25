"""
Inference logic: loads the saved TF-IDF vectorizer, scaler, and classifier,
and exposes a single `predict(title, text)` function used by the API layer.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from .features import build_feature_frame
from .preprocess import clean_text

MODEL_DIR = Path(__file__).resolve().parent / "model"


class FakeNewsModel:
    def __init__(self):
        self.vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        self.model = joblib.load(MODEL_DIR / "model.pkl")
        metadata = joblib.load(MODEL_DIR / "metadata.pkl")
        self.model_name = metadata.get("model_name", "unknown")

    def predict(self, title: str, text: str) -> dict:
        clean_title = clean_text(title)
        clean_body = clean_text(text)
        combined_clean = f"{clean_title} {clean_body}".strip()

        tfidf_vec = self.vectorizer.transform([combined_clean])

        feat_frame = build_feature_frame(pd.Series([title]), pd.Series([text]))
        feat_scaled = self.scaler.transform(feat_frame.values)

        X = hstack([tfidf_vec, csr_matrix(feat_scaled)]).tocsr()

        proba = self.model.predict_proba(X)[0]
        fake_prob, real_prob = float(proba[0]), float(proba[1])
        is_fake = fake_prob > real_prob

        return {
            "label": "Fake" if is_fake else "Real",
            "is_fake": bool(is_fake),
            "confidence": max(fake_prob, real_prob),
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "model_name": self.model_name,
        }


# Loaded once at process startup and reused across requests
_model_instance: FakeNewsModel | None = None


def get_model() -> FakeNewsModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = FakeNewsModel()
    return _model_instance
