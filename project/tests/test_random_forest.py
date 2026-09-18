import torch

from project.classifiers import RandomForestEraClassifier


def test_random_forest_era_classifier():
    features = torch.tensor([
        [0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0],
        [2.0, 2.0], [2.1, 2.0],
    ])
    labels = [0, 0, 1, 1, 2, 2]
    classifier = RandomForestEraClassifier(n_estimators=10).fit(features, labels)
    assert classifier.predict(features).shape == (6,)
    assert classifier.predict_proba(features).shape == (6, 3)