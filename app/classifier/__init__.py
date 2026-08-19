"""
AdtU Campus Copilot — Intent Classifier package.

Locked runtime labels:
    admissions | fees | facilities | out_of_scope

Architecture: TF-IDF (word + char) → LinearSVC pipeline (scikit-learn).
"""

LOCKED_LABELS: tuple[str, ...] = (
    "admissions",
    "fees",
    "facilities",
    "out_of_scope",
)
