"""
query_classifier.py
-------------------
Drop this file next to your RAG code.
Requires: scikit-learn, joblib  (pip install scikit-learn joblib)

Usage:
    from queryProcess.query_classifier import QueryClassifier

    clf = QueryClassifier()          # loads model from disk
    result = clf.classify("מה הציון הממוצע בקורס מבני נתונים?")
    # result → {"type": "sql", "confidence": 0.85}

    result = clf.classify("חוות דעת על דני קרן")
    # result → {"type": "semantic", "confidence": 0.91}
"""

import os
import joblib
from pathlib import Path

# Path to the saved model — same directory as this file
_MODEL_PATH = Path(__file__).parent / "query_classifier.joblib"


class QueryClassifier:
    """Classifies a user query as 'sql' or 'semantic'."""

    def __init__(self, model_path: str | Path = _MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run train_classifier.py first to generate it."
            )
        self._pipeline = joblib.load(model_path)
        self._label_map = {1: "sql", 0: "semantic"}

    def classify(self, query: str) -> dict:
        """
        Returns:
            {"type": "sql" | "semantic", "confidence": float}
        """
        pred  = self._pipeline.predict([query])[0]
        proba = self._pipeline.predict_proba([query])[0]
        return {
            "type":       self._label_map[pred],
            "confidence": round(float(max(proba)), 4),
        }

    def classify_batch(self, queries: list[str]) -> list[dict]:
        """Classify multiple queries at once (more efficient)."""
        preds  = self._pipeline.predict(queries)
        probas = self._pipeline.predict_proba(queries)
        return [
            {"type": self._label_map[p], "confidence": round(float(max(pr)), 4)}
            for p, pr in zip(preds, probas)
        ]
