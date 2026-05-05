"""Train a gradient boosting classifier to predict project withdrawal.

The label is `withdrawn` (1) vs. all else. Operational projects are explicitly
labeled 0; in-progress projects are excluded from training so the model learns
the historical separation rather than treating undecided projects as negatives.

The historical baseline withdrawal rate is ~86%, so ROC-AUC and PR-AUC are more
informative than accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.load_data import load_queued_up

FEATURE_NUMERIC = ["mw", "queue_age_years"]
FEATURE_CATEGORICAL = ["rto", "resource_type"]


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with a resolved outcome (withdrawn OR operational)."""
    resolved = df[(df["withdrawn"] == 1) | (df["operational"] == 1)].copy()
    resolved = resolved.dropna(subset=[c for c in FEATURE_NUMERIC if c in resolved.columns])
    return resolved


def encode_features(df: pd.DataFrame, encoder: OneHotEncoder | None = None):
    numeric_cols = [c for c in FEATURE_NUMERIC if c in df.columns]
    cat_cols = [c for c in FEATURE_CATEGORICAL if c in df.columns]
    cat = df[cat_cols].fillna("UNKNOWN").astype(str)
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_encoded = encoder.fit_transform(cat)
    else:
        cat_encoded = encoder.transform(cat)
    cat_names = encoder.get_feature_names_out(cat_cols)
    cat_df = pd.DataFrame(cat_encoded, columns=cat_names, index=df.index)
    numeric_df = df[numeric_cols].copy()
    X = pd.concat([numeric_df, cat_df], axis=1)
    return X, encoder


def train(df: pd.DataFrame, random_state: int = 42):
    train_df = build_training_frame(df)
    if len(train_df) < 100:
        raise ValueError(
            f"Only {len(train_df)} resolved rows — not enough to train. "
            "Check that operational and withdrawn outcomes are populated."
        )

    y = train_df["withdrawn"].values
    X, encoder = encode_features(train_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(f"Training rows: {len(train_df):,}")
    print(f"Withdrawal base rate: {y.mean():.1%}\n")
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(f"PR-AUC : {average_precision_score(y_test, proba):.3f}\n")
    print(classification_report(y_test, pred, digits=3))

    importances = pd.Series(clf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False).head(15)
    print("\nTop feature importances:")
    print(importances.to_string())

    return clf, encoder, importances


def score_open_queue(
    df: pd.DataFrame, clf: GradientBoostingClassifier, encoder: OneHotEncoder
) -> pd.DataFrame:
    """Score in-progress (unresolved) projects with their predicted P(withdraw)."""
    open_df = df[(df["withdrawn"] == 0) & (df["operational"] == 0)].copy()
    if open_df.empty:
        return open_df
    open_df = open_df.dropna(subset=[c for c in FEATURE_NUMERIC if c in open_df.columns])
    X, _ = encode_features(open_df, encoder=encoder)
    open_df["p_withdraw"] = clf.predict_proba(X)[:, 1]
    return open_df


if __name__ == "__main__":
    df = load_queued_up()
    clf, encoder, importances = train(df)
    scored = score_open_queue(df, clf, encoder)
    if not scored.empty and "rto" in scored.columns:
        print("\nMean P(withdraw) for open queue, by RTO:")
        print(
            scored.groupby("rto")["p_withdraw"]
            .agg(["count", "mean"])
            .sort_values("count", ascending=False)
            .head(15)
            .round(3)
            .to_string()
        )
