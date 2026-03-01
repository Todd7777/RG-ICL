import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics.classification import ClassificationMetrics
from metrics.vqa import VQAMetrics
from metrics.calibration import CalibrationMetrics


def test_binary_classification_perfect():
    m = ClassificationMetrics()
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.1, 0.9, 0.95, 0.85])

    result = m.compute_binary(y_true, y_pred, y_prob)
    assert result.auc == 1.0, f"Expected AUC=1.0, got {result.auc}"
    assert result.accuracy == 1.0
    assert result.n_samples == 6
    assert result.brier < 0.05
    print("PASS: test_binary_classification_perfect")


def test_binary_classification_random():
    m = ClassificationMetrics()
    rng = np.random.RandomState(42)
    n = 200
    y_true = rng.randint(0, 2, n)
    y_prob = rng.uniform(0, 1, n)
    y_pred = (y_prob >= 0.5).astype(int)

    result = m.compute_binary(y_true, y_pred, y_prob)
    assert 0.3 < result.auc < 0.7
    assert 0.3 < result.accuracy < 0.7
    assert result.n_samples == n
    assert 0.0 <= result.brier <= 1.0
    assert 0.0 <= result.ece <= 1.0
    assert result.nll > 0
    print("PASS: test_binary_classification_random")


def test_multiclass_classification():
    m = ClassificationMetrics()
    rng = np.random.RandomState(42)
    n = 100
    n_classes = 4
    y_true = rng.randint(0, n_classes, n)
    y_prob = rng.dirichlet(np.ones(n_classes), n)
    y_pred = np.argmax(y_prob, axis=1)

    result = m.compute_multiclass(y_true, y_pred, y_prob, n_classes)
    assert 0.0 <= result.auc <= 1.0
    assert len(result.auc_per_label) == n_classes
    assert len(result.sens_at_spec90_per_label) == n_classes
    assert result.confusion is not None
    assert result.confusion.shape == (n_classes, n_classes)
    assert result.n_samples == n
    print("PASS: test_multiclass_classification")


def test_multilabel_classification():
    m = ClassificationMetrics()
    rng = np.random.RandomState(42)
    n = 100
    n_labels = 5
    y_true = rng.randint(0, 2, (n, n_labels))
    y_prob = rng.uniform(0, 1, (n, n_labels))
    y_pred = (y_prob >= 0.5).astype(int)

    result = m.compute_multilabel(y_true, y_pred, y_prob, n_labels)
    assert len(result.auc_per_label) == n_labels
    assert 0.0 <= result.auc <= 1.0
    assert result.n_samples == n
    assert 0.0 <= result.brier
    assert 0.0 <= result.ece
    print("PASS: test_multilabel_classification")


def test_sens_at_spec90():
    m = ClassificationMetrics()
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.concatenate([
        np.random.RandomState(42).uniform(0, 0.4, 50),
        np.random.RandomState(42).uniform(0.6, 1.0, 50),
    ])
    y_pred = (y_prob >= 0.5).astype(int)

    result = m.compute_binary(y_true, y_pred, y_prob)
    assert result.sens_at_spec90 > 0.5
    assert result.spec_at_sens90 > 0.5
    print("PASS: test_sens_at_spec90")


def test_vqa_metrics_perfect():
    m = VQAMetrics()
    references = ["The heart is enlarged", "No fracture detected", "Bilateral pleural effusion"]
    hypotheses = ["The heart is enlarged", "No fracture detected", "Bilateral pleural effusion"]

    result = m.compute(references, hypotheses)
    assert result.bleu4 > 0.9
    assert result.rouge_l > 0.99
    assert result.meteor > 0.9
    assert result.exact_match == 1.0
    assert result.n_samples == 3
    print("PASS: test_vqa_metrics_perfect")


def test_vqa_metrics_partial():
    m = VQAMetrics()
    references = ["The heart is enlarged", "No fracture detected"]
    hypotheses = ["The heart appears enlarged and dilated", "A fracture was found"]

    result = m.compute(references, hypotheses)
    assert 0.0 < result.bleu4 < 1.0
    assert 0.0 < result.rouge_l < 1.0
    assert result.exact_match == 0.0
    assert result.n_samples == 2
    assert len(result.per_sample_bleu4) == 2
    assert len(result.per_sample_rouge_l) == 2
    print("PASS: test_vqa_metrics_partial")


def test_vqa_metrics_empty():
    m = VQAMetrics()
    references = ["answer"]
    hypotheses = [""]

    result = m.compute(references, hypotheses)
    assert result.bleu4 == 0.0
    assert result.rouge_l == 0.0
    assert result.meteor == 0.0
    print("PASS: test_vqa_metrics_empty")


def test_calibration_perfect():
    cm = CalibrationMetrics(n_bins=10)
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.05, 0.1, 0.95, 0.9, 0.15, 0.85])
    y_pred = (y_prob >= 0.5).astype(int)

    result = cm.compute(y_true, y_prob, y_pred)
    assert result.ece < 0.2
    assert result.brier < 0.05
    assert len(result.bin_accuracies) == 10
    assert len(result.bin_counts) == 10
    assert sum(result.bin_counts) == len(y_true)
    print("PASS: test_calibration_perfect")


def test_calibration_overconfident():
    cm = CalibrationMetrics(n_bins=10)
    rng = np.random.RandomState(42)
    n = 200
    y_true = rng.randint(0, 2, n)
    y_prob = np.clip(rng.uniform(0.8, 1.0, n), 0, 1)
    y_pred = np.ones(n, dtype=int)

    result = cm.compute(y_true, y_prob, y_pred)
    assert result.ece > 0.2
    assert result.mce > 0.2
    print("PASS: test_calibration_overconfident")


def test_calibration_multilabel():
    cm = CalibrationMetrics(n_bins=10)
    rng = np.random.RandomState(42)
    n = 100
    n_labels = 5
    y_true = rng.randint(0, 2, (n, n_labels))
    y_prob = rng.uniform(0, 1, (n, n_labels))

    results = cm.compute_multilabel(y_true, y_prob, n_labels)
    assert len(results) == n_labels
    for r in results:
        assert 0.0 <= r.ece <= 1.0
        assert r.brier >= 0.0
    print("PASS: test_calibration_multilabel")


if __name__ == "__main__":
    test_binary_classification_perfect()
    test_binary_classification_random()
    test_multiclass_classification()
    test_multilabel_classification()
    test_sens_at_spec90()
    test_vqa_metrics_perfect()
    test_vqa_metrics_partial()
    test_vqa_metrics_empty()
    test_calibration_perfect()
    test_calibration_overconfident()
    test_calibration_multilabel()
    print("\nAll metrics tests passed.")
