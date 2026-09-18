from pathlib import Path

from project.training.train_random_forest import load_feature_csv


def test_load_feature_csv():
    features, labels = load_feature_csv(Path("Data/features_3_sec.csv"))
    assert features.shape[0] == labels.shape[0]
    assert features.shape[1] > 2