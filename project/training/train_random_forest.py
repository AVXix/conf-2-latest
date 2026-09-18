"""Train the Random Forest baseline on a supplied feature CSV.

The checked-in Data CSV contains genre labels, so this trains genre
classification. It is not an era-classification experiment.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from ..classifiers import RandomForestEraClassifier


def load_feature_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Feature CSV is empty: {path}")
    feature_names = [name for name in rows[0] if name not in {"filename", "length", "label"}]
    if not feature_names:
        raise ValueError("Feature CSV does not contain numeric feature columns")
    try:
        features = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    except (KeyError, ValueError) as exc:
        raise ValueError("Feature CSV contains missing or non-numeric feature values") from exc
    labels = np.asarray([row["label"] for row in rows])
    return features, labels


def train_random_forest(
    csv_path: str | Path,
    output_path: str | Path,
    n_estimators: int = 300,
    test_size: float = 0.2,
    random_state: int = 42,
) -> float:
    features, labels = load_feature_csv(csv_path)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    classifier = RandomForestEraClassifier(n_estimators=n_estimators, random_state=random_state)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    accuracy = float(accuracy_score(test_labels, predictions))
    print(f"samples={len(labels)} features={features.shape[1]} classes={len(np.unique(labels))}")
    print(f"test_accuracy={accuracy:.4f}")
    print(classification_report(test_labels, predictions, zero_division=0))
    classifier.save(output_path)
    print(f"saved={output_path}")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Random Forest on a feature CSV")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("genre_random_forest.pkl"))
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_random_forest(args.csv_path, args.output, args.estimators, args.test_size, args.seed)


if __name__ == "__main__":
    main()