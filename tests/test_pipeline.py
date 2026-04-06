import json
import os
from unittest.mock import patch

from pipeline.evaluator import evaluate_models


SAMPLE_TEXTS = ["resume text one", "resume text two"]
SAMPLE_LABELS = ["HR", "ENGINEERING"]


def mock_load_data(_):
    return SAMPLE_TEXTS, SAMPLE_LABELS


def mock_run_model_in_docker(docker_image, texts, timeout_seconds=60):
    return SAMPLE_LABELS, 0.1234


def mock_load_config():
    return {
        "dataset": {
            "path": "data/processed/processed_resumes.csv"
        },
        "models": [
            {"name": "logistic_regression", "docker_image": "logistic_regression"},
            {"name": "random_forest", "docker_image": "random_forest"},
        ],
        "evaluation": {
            "save_results_path": "outputs/results.json",
            "save_leaderboard_path": "outputs/leaderboard.csv"
        },
        "runtime": {
            "batch_size": 32,
            "timeout_seconds": 30
        },
        "logging": {
            "log_file": "outputs/logs/run.log",
            "level": "INFO"
        }
    }


@patch("pipeline.evaluator.load_config", side_effect=mock_load_config)
@patch("pipeline.evaluator.run_model_in_docker", side_effect=mock_run_model_in_docker)
@patch("pipeline.evaluator.load_data", side_effect=mock_load_data)
def test_evaluate_models(mock_load, mock_run, mock_config):
    os.makedirs("outputs/logs", exist_ok=True)

    results = evaluate_models(
        model_name="all",
        dataset_path="dummy.csv",
        use_docker=True
    )

    assert os.path.exists("outputs/results.json")
    assert os.path.exists("outputs/leaderboard.csv")

    with open("outputs/results.json", "r", encoding="utf-8") as f:
        saved_results = json.load(f)

    assert len(results) == 2
    assert len(saved_results) == 2
    assert saved_results[0]["accuracy"] == 1.0
    assert "model" in saved_results[0]
    assert "f1" in saved_results[0]