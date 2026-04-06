import csv


def load_data(path: str):
    """
    Loads processed dataset and returns texts and labels as lists.
    Expected columns: clean_text, label
    """
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = {"clean_text", "label"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")

        for row in reader:
            text = (row.get("clean_text") or "").strip()
            label = (row.get("label") or "").strip()

            if text and label:
                texts.append(text)
                labels.append(label)

    if not texts or not labels:
        raise ValueError("Dataset is empty after dropping missing values.")

    return texts, labels