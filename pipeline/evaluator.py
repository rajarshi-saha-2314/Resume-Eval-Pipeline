import csv
import json
import pickle
import subprocess
import time
from pathlib import Path

from pipeline.data_loader import load_data
from pipeline.metrics import compute_metrics
from pipeline.utils import load_config, ensure_directory


def ensure_output_dirs():
    ensure_directory("outputs")
    ensure_directory("outputs/logs")


def log_message(message: str, log_file: str):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def run_model_in_docker(docker_image: str, texts, timeout_seconds: int = 60):
    start_time = time.time()

    process = subprocess.Popen(
        ["docker", "run", "-i", "--rm", docker_image],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    input_data = "\n".join(texts)

    try:
        stdout, stderr = process.communicate(input=input_data, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        raise TimeoutError(
            f"Docker container '{docker_image}' timed out after {timeout_seconds} seconds."
        )

    latency = time.time() - start_time

    if process.returncode != 0:
        raise RuntimeError(
            f"Docker container '{docker_image}' failed.\nError: {stderr.strip()}"
        )

    predictions = [line.strip() for line in stdout.splitlines() if line.strip()]
    return predictions, latency


def run_model_locally(model_name: str, texts):
    model_path = Path("models") / model_name / "model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    start_time = time.time()

    with open(model_path, "rb") as f:
        model, vectorizer = pickle.load(f)

    text_vectors = vectorizer.transform(texts)
    predictions = model.predict(text_vectors)

    latency = time.time() - start_time
    return [str(pred) for pred in predictions], latency


def save_results_json(results, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def save_leaderboard_csv(results, path: str):
    fieldnames = ["model", "accuracy", "precision", "recall", "f1", "latency_seconds"]
    sorted_results = sorted(results, key=lambda x: x.get("f1", 0), reverse=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_results)


def evaluate_models(model_name="all", dataset_path=None, use_docker=True):
    ensure_output_dirs()

    config = load_config()
    log_file = config["logging"]["log_file"]
    timeout_seconds = config["runtime"]["timeout_seconds"]

    if dataset_path is None:
        dataset_path = config["dataset"]["path"]

    models = config["models"]

    if model_name == "all":
        models_to_run = models
    else:
        models_to_run = [m for m in models if m["name"] == model_name]
        if not models_to_run:
            raise ValueError(f"Model '{model_name}' not found in config.")

    texts, labels = load_data(dataset_path)

    log_message("=== Evaluation Started ===", log_file)
    log_message(f"Dataset: {dataset_path}", log_file)
    log_message(f"Total samples: {len(texts)}", log_file)
    log_message(f"Models selected: {[m['name'] for m in models_to_run]}", log_file)
    log_message(f"Execution mode: {'Docker' if use_docker else 'Local'}", log_file)

    results = []

    for model_cfg in models_to_run:
        current_model_name = model_cfg["name"]
        docker_image = model_cfg["docker_image"]

        log_message(f"Running model: {current_model_name}", log_file)

        try:
            if use_docker:
                predictions, latency = run_model_in_docker(
                    docker_image=docker_image,
                    texts=texts,
                    timeout_seconds=timeout_seconds
                )
            else:
                predictions, latency = run_model_locally(
                    model_name=current_model_name,
                    texts=texts
                )

            if len(predictions) != len(labels):
                raise ValueError(
                    f"Prediction count mismatch for '{current_model_name}': "
                    f"expected {len(labels)}, got {len(predictions)}"
                )

            metrics = compute_metrics(labels, predictions)

            result = {
                "model": current_model_name,
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "latency_seconds": round(latency, 4)
            }

            results.append(result)
            log_message(f"Completed: {result}", log_file)

        except Exception as e:
            error_message = f"Error while evaluating '{current_model_name}': {str(e)}"
            log_message(error_message, log_file)
            print(error_message)

    results_json_path = config["evaluation"]["save_results_path"]
    leaderboard_csv_path = config["evaluation"]["save_leaderboard_path"]

    save_results_json(results, results_json_path)
    save_leaderboard_csv(results, leaderboard_csv_path)

    log_message("=== Evaluation Finished ===", log_file)

    return results