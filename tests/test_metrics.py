from pipeline.metrics import compute_metrics


def test_compute_metrics_perfect_predictions():
    y_true = ["A", "B", "A", "C"]
    y_pred = ["A", "B", "A", "C"]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0