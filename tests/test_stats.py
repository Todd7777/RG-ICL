import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stats.bootstrap import PairedBootstrap
from stats.delong import DeLongTest


def test_bootstrap_identical_methods():
    bootstrap = PairedBootstrap(n_resamples=500, seed=42, ci_level=0.95)
    rng = np.random.RandomState(42)
    n = 100
    y_true = rng.randint(0, 2, n)
    scores = rng.uniform(0, 1, n)

    from sklearn.metrics import roc_auc_score
    result = bootstrap.test_auc(y_true, scores, scores, "method", "method")

    assert abs(result.observed_diff) < 1e-10
    assert result.p_value > 0.5
    assert result.n_resamples == 500
    print("PASS: test_bootstrap_identical_methods")


def test_bootstrap_clearly_better():
    bootstrap = PairedBootstrap(n_resamples=1000, seed=42, ci_level=0.95)
    rng = np.random.RandomState(42)
    n = 200
    y_true = np.array([0] * 100 + [1] * 100)

    scores_a = np.concatenate([
        rng.uniform(0.3, 0.7, 100),
        rng.uniform(0.3, 0.7, 100),
    ])
    scores_b = np.concatenate([
        rng.uniform(0.0, 0.3, 100),
        rng.uniform(0.7, 1.0, 100),
    ])

    result = bootstrap.test_auc(y_true, scores_a, scores_b, "weak", "strong")

    assert result.observed_b > result.observed_a
    assert result.observed_diff > 0
    assert result.p_value < 0.05
    assert result.ci_lower_diff > 0
    print("PASS: test_bootstrap_clearly_better")


def test_bootstrap_stores_indices():
    bootstrap = PairedBootstrap(n_resamples=100, seed=42, store_indices=True)
    rng = np.random.RandomState(42)
    n = 50
    y_true = rng.randint(0, 2, n)
    scores_a = rng.uniform(0, 1, n)
    scores_b = rng.uniform(0, 1, n)

    def mean_fn(y, s):
        return float(np.mean(s))

    result = bootstrap.test(y_true, scores_a, scores_b, mean_fn, "mean")
    assert result.resample_indices is not None
    assert len(result.resample_indices) == 100
    assert all(len(idx) == n for idx in result.resample_indices)
    print("PASS: test_bootstrap_stores_indices")


def test_bootstrap_reproducibility():
    rng = np.random.RandomState(42)
    n = 100
    y_true = rng.randint(0, 2, n)
    scores_a = rng.uniform(0, 1, n)
    scores_b = rng.uniform(0, 1, n)

    b1 = PairedBootstrap(n_resamples=500, seed=123)
    b2 = PairedBootstrap(n_resamples=500, seed=123)

    r1 = b1.test_auc(y_true, scores_a, scores_b)
    r2 = b2.test_auc(y_true, scores_a, scores_b)

    assert r1.p_value == r2.p_value
    assert r1.ci_lower_diff == r2.ci_lower_diff
    assert r1.ci_upper_diff == r2.ci_upper_diff
    print("PASS: test_bootstrap_reproducibility")


def test_bootstrap_accuracy():
    bootstrap = PairedBootstrap(n_resamples=500, seed=42)
    rng = np.random.RandomState(42)
    n = 100
    y_true = rng.randint(0, 2, n)
    preds_a = y_true.copy()
    preds_b = rng.randint(0, 2, n)

    result = bootstrap.test_accuracy(y_true, preds_a, preds_b, "perfect", "random")
    assert result.observed_a == 1.0
    assert result.observed_b < 1.0
    assert result.observed_diff < 0
    print("PASS: test_bootstrap_accuracy")


def test_delong_identical():
    delong = DeLongTest()
    rng = np.random.RandomState(42)
    n = 100
    y_true = np.array([0] * 50 + [1] * 50)
    scores = rng.uniform(0, 1, n)

    result = delong.test(y_true, scores, scores, "A", "A")
    assert abs(result.auc_diff) < 1e-10
    assert result.p_value > 0.9
    print("PASS: test_delong_identical")


def test_delong_significant():
    delong = DeLongTest()
    y_true = np.array([0] * 100 + [1] * 100)

    rng = np.random.RandomState(42)
    scores_a = np.concatenate([
        rng.uniform(0.3, 0.7, 100),
        rng.uniform(0.3, 0.7, 100),
    ])
    scores_b = np.concatenate([
        rng.uniform(0.0, 0.2, 100),
        rng.uniform(0.8, 1.0, 100),
    ])

    result = delong.test(y_true, scores_a, scores_b, "weak", "strong")
    assert result.auc_b > result.auc_a
    assert result.p_value < 0.05
    assert result.se_diff > 0
    assert result.n_positive == 100
    assert result.n_negative == 100
    print("PASS: test_delong_significant")


def test_delong_auc_values():
    delong = DeLongTest()
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(42)
    y_true = np.array([0] * 50 + [1] * 50)
    scores = rng.uniform(0, 1, 100)

    result = delong.test(y_true, scores, scores)
    sklearn_auc = roc_auc_score(y_true, scores)

    assert abs(result.auc_a - sklearn_auc) < 0.01, f"DeLong AUC {result.auc_a} != sklearn {sklearn_auc}"
    print("PASS: test_delong_auc_values")


def test_delong_multilabel():
    delong = DeLongTest()
    rng = np.random.RandomState(42)
    n = 100
    n_labels = 3
    y_true = rng.randint(0, 2, (n, n_labels))
    scores_a = rng.uniform(0, 1, (n, n_labels))
    scores_b = rng.uniform(0, 1, (n, n_labels))

    results = delong.test_multilabel(y_true, scores_a, scores_b, n_labels)
    assert len(results) == n_labels
    for r in results:
        assert 0.0 <= r.auc_a <= 1.0
        assert 0.0 <= r.auc_b <= 1.0
        assert 0.0 <= r.p_value <= 1.0
    print("PASS: test_delong_multilabel")


if __name__ == "__main__":
    test_bootstrap_identical_methods()
    test_bootstrap_clearly_better()
    test_bootstrap_stores_indices()
    test_bootstrap_reproducibility()
    test_bootstrap_accuracy()
    test_delong_identical()
    test_delong_significant()
    test_delong_auc_values()
    test_delong_multilabel()
    print("\nAll stats tests passed.")
