import sys
import pickle


def load_model():
    with open("model.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer


def read_inputs():
    raw = sys.stdin.read()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return lines


def main():
    model, vectorizer = load_model()
    inputs = read_inputs()

    if not inputs:
        return

    inputs_vec = vectorizer.transform(inputs)
    predictions = model.predict(inputs_vec)

    for pred in predictions:
        print(pred)


if __name__ == "__main__":
    main()