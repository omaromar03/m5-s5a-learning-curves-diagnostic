"""
Module 5 Week A — Stretch: Learning Curves Diagnostic

This script:
1. Loads the telecom churn dataset
2. Preprocesses numeric and categorical features
3. Trains a logistic regression model in a pipeline
4. Uses sklearn.learning_curve with stratified CV
5. Plots training and validation F1 scores with confidence bands
6. Saves the figure as learning_curve.png

Run:
    python learning_curves_diagnostic.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


DATA_PATH = "data/telecom_churn.csv"
OUTPUT_PLOT = "learning_curve.png"
RANDOM_STATE = 42


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    df = pd.read_csv(filepath)
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe before modeling.
    - Convert total_charges to numeric
    - Drop rows with missing target
    """
    df = df.copy()

    # Convert total_charges safely in case it is read as object/string
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    # Drop rows with missing target if any
    df = df.dropna(subset=["churned"])

    return df


def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into X and y.
    customer_id is dropped because it is an identifier, not a useful feature.
    """
    X = df.drop(columns=["churned", "customer_id"])
    y = df["churned"].astype(int)
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Build preprocessing + logistic regression pipeline.
    """
    numeric_features = [
        "senior_citizen",
        "tenure",
        "monthly_charges",
        "total_charges",
        "num_support_calls",
    ]

    categorical_features = [
        "gender",
        "contract_type",
        "internet_service",
        "payment_method",
        "has_partner",
        "has_dependents",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def compute_learning_curves(X: pd.DataFrame, y: pd.Series, pipeline: Pipeline):
    """
    Compute learning curves using stratified cross-validation and F1-score.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    train_sizes = np.linspace(0.1, 1.0, 8)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator=pipeline,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return train_sizes_abs, train_scores, val_scores


def plot_learning_curves(train_sizes, train_scores, val_scores, output_path: str):
    """
    Plot training and validation learning curves with ±1 std bands.
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))

    plt.plot(train_sizes, train_mean, marker="o", label="Training F1 Score")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
    )

    plt.plot(train_sizes, val_mean, marker="s", label="Validation F1 Score")
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2,
    )

    plt.title("Learning Curve: Logistic Regression on Telecom Churn")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def print_summary(train_sizes, train_scores, val_scores):
    """
    Print summary statistics to help interpretation.
    """
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    print("\nLearning Curve Summary")
    print("-" * 60)
    for size, tr, va in zip(train_sizes, train_mean, val_mean):
        print(f"Train size: {size:4d} | Train F1: {tr:.4f} | Validation F1: {va:.4f}")

    final_gap = train_mean[-1] - val_mean[-1]
    print("-" * 60)
    print(f"Final training score:   {train_mean[-1]:.4f}")
    print(f"Final validation score: {val_mean[-1]:.4f}")
    print(f"Final gap:              {final_gap:.4f}")


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing dataframe...")
    df = preprocess_dataframe(df)

    print("Preparing features and target...")
    X, y = split_features_target(df)

    print(f"Dataset shape after cleaning: {df.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print("Target distribution:")
    print(y.value_counts(normalize=True).sort_index())

    print("Building pipeline...")
    pipeline = build_pipeline(X)

    print("Computing learning curves...")
    train_sizes, train_scores, val_scores = compute_learning_curves(X, y, pipeline)

    print("Saving plot...")
    plot_learning_curves(train_sizes, train_scores, val_scores, OUTPUT_PLOT)

    print_summary(train_sizes, train_scores, val_scores)

    print(f"\nDone. Plot saved as: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()