import os
import yaml


def load_config(path="configs/config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)