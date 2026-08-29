"""
AdtU Campus Copilot — Classifier Training Script.

Architecture (locked):
    FeatureUnion(
        TF-IDF word-level  (1-3 grams, sublinear TF)
        TF-IDF char-level  (2-4 grams, sublinear TF)
    )
    -> LinearSVC

Training dataset (v3, 228 rows):
    data/classifier/training_queries_v3.csv

Usage:
    python -m app.classifier.train

Outputs:
    app/classifier/model/intent_classifier.pkl
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from app.classifier import LOCKED_LABELS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[2]
TRAIN_CSV: Path = ROOT / "data" / "classifier" / "training_queries_v3.csv"
MODEL_DIR: Path = Path(__file__).resolve().parent / "model"
MODEL_PATH: Path = MODEL_DIR / "intent_classifier.pkl"

# Expected distribution for training_queries_v3.csv
_EXPECTED_DIST: dict[str, int] = {
    "admissions": 62,
    "fees": 46,
    "facilities": 52,
    "out_of_scope": 68,
}


def load_and_validate(csv_path: Path) -> tuple[list[str], list[str]]:
    """Load the frozen CSV and validate labels.

    Returns:
        (queries, labels) — both as plain Python lists.

    Raises:
        SystemExit: if the file is missing, malformed, or contains unknown labels.
    """
    if not csv_path.exists():
        print(f"[ERROR] Training CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df: pd.DataFrame = pd.read_csv(csv_path)

    if "query" not in df.columns or "label" not in df.columns:
        print(
            "[ERROR] CSV must have exactly two columns: query, label",
            file=sys.stderr,
        )
        sys.exit(1)

    # Drop any accidental blank trailing rows
    df = df.dropna(subset=["query", "label"])

    unknown: set[str] = set(df["label"].unique()) - set(LOCKED_LABELS)
    if unknown:
        print(f"[ERROR] Unknown labels found: {unknown}", file=sys.stderr)
        sys.exit(1)

    print(f"[train] Dataset loaded  : {len(df)} rows from {csv_path}")
    dist: dict[str, int] = df["label"].value_counts().to_dict()
    for lbl in LOCKED_LABELS:
        print(f"[train]   {lbl:<18} = {dist.get(lbl, 0)}")

    # Verify row count and distribution match the v2 expectation
    all_ok = True
    if len(df) != sum(_EXPECTED_DIST.values()):
        print(
            f"[ERROR] Row count {len(df)} != expected {sum(_EXPECTED_DIST.values())}",
            file=sys.stderr,
        )
        all_ok = False
    for lbl, expected in _EXPECTED_DIST.items():
        actual = dist.get(lbl, 0)
        if actual != expected:
            print(
                f"[ERROR] {lbl}: got {actual}, expected {expected}",
                file=sys.stderr,
            )
            all_ok = False
    dups = df.duplicated(subset=["query"]).sum()
    if dups:
        print(f"[ERROR] {dups} duplicate queries found", file=sys.stderr)
        all_ok = False
    if not all_ok:
        sys.exit(1)
    print("[train] Pre-train validation passed.")

    return df["query"].tolist(), df["label"].tolist()


def build_pipeline() -> Pipeline:
    """Construct the locked TF-IDF + LinearSVC pipeline.

    Feature union:
        word_tfidf  — word 1–3 grams, sublinear TF, min_df=1
        char_tfidf  — char_wb 2–4 grams, sublinear TF, min_df=1

    Classifier:
        LinearSVC   — C=1.0, max_iter=2000, random_state=42
    """
    word_tfidf: TfidfVectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
        lowercase=True,
    )

    char_tfidf: TfidfVectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
        lowercase=True,
    )

    features: FeatureUnion = FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_tfidf),
            ("char_tfidf", char_tfidf),
        ]
    )

    pipeline: Pipeline = Pipeline(
        steps=[
            ("features", features),
            ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42)),
        ]
    )
    return pipeline


def train() -> None:
    """Entry point: load data → build pipeline → train → save."""
    print("[train] Starting classifier training ...")

    queries, labels = load_and_validate(TRAIN_CSV)

    pipeline: Pipeline = build_pipeline()
    pipeline.fit(queries, labels)
    print("[train] Pipeline fitted  : TF-IDF(word+char) -> LinearSVC")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[train] Model saved      : {MODEL_PATH}")
    print("[train] Done.")


if __name__ == "__main__":
    train()
