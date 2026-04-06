import json
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.evaluator import evaluate_models

st.set_page_config(page_title="ResumeEval Dashboard", layout="wide")

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = Path("outputs")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASET_PATH = PROCESSED_DIR / "processed_resumes.csv"


def validate_uploaded_dataset(df: pd.DataFrame):
    required_columns = {"clean_text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Uploaded CSV must contain the columns: clean_text, label"
        )


def save_uploaded_dataset(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    validate_uploaded_dataset(df)
    df.to_csv(DEFAULT_DATASET_PATH, index=False)
    return df


def load_results_json():
    results_path = OUTPUTS_DIR / "results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_leaderboard_csv():
    leaderboard_path = OUTPUTS_DIR / "leaderboard.csv"
    if leaderboard_path.exists():
        return pd.read_csv(leaderboard_path)
    return None


st.title("📊 ResumeEval Dashboard")
st.markdown(
    "Upload a processed dataset and compare multiple resume classification models."
)

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "Upload processed CSV",
        type=["csv"],
        help="CSV must contain: clean_text, label"
    )

    selected_model = st.selectbox(
        "Select model",
        options=["all", "logistic_regression", "random_forest"],
        index=0
    )

    run_eval = st.button("🚀 Run Evaluation", use_container_width=True)

if uploaded_file is not None:
    try:
        uploaded_df = save_uploaded_dataset(uploaded_file)
        st.success("Dataset uploaded and saved successfully.")
        with st.expander("Preview uploaded dataset"):
            st.dataframe(uploaded_df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Dataset validation failed: {e}")

if run_eval:
    if not DEFAULT_DATASET_PATH.exists():
        st.error("Please upload a processed dataset first.")
    else:
        try:
            with st.spinner("Running evaluation..."):
                results = evaluate_models(
                    model_name=selected_model,
                    dataset_path=str(DEFAULT_DATASET_PATH),
                    use_docker=False
                )
            if results:
                st.success("Evaluation completed successfully.")
            else:
                st.warning("Evaluation finished, but no results were generated.")
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

results = load_results_json()
leaderboard = load_leaderboard_csv()

if results:
    st.subheader("Raw Results")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

if leaderboard is not None and not leaderboard.empty:
    st.subheader("Leaderboard")
    st.dataframe(leaderboard, use_container_width=True)

    metric = st.selectbox(
        "Select metric to compare",
        ["accuracy", "precision", "recall", "f1", "latency_seconds"],
        index=3
    )

    chart_df = leaderboard.set_index("model")[[metric]]
    st.subheader(f"Model Comparison: {metric}")
    st.bar_chart(chart_df)

    best_model = leaderboard.sort_values(by="f1", ascending=False).iloc[0]
    st.info(
        f"Best model by F1 score: **{best_model['model']}** "
        f"(F1 = {best_model['f1']})"
    )
else:
    st.info("No evaluation results available yet. Upload a dataset and run evaluation.")import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to Python path so `pipeline` can be imported on Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.evaluator import evaluate_models

st.set_page_config(page_title="ResumeEval Dashboard", layout="wide")

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASET_PATH = PROCESSED_DIR / "processed_resumes.csv"


def validate_uploaded_dataset(df: pd.DataFrame):
    required_columns = {"clean_text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Uploaded CSV must contain the columns: clean_text, label")


def save_uploaded_dataset(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    validate_uploaded_dataset(df)
    df.to_csv(DEFAULT_DATASET_PATH, index=False)
    return df


def load_results_json():
    results_path = OUTPUTS_DIR / "results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_leaderboard_csv():
    leaderboard_path = OUTPUTS_DIR / "leaderboard.csv"
    if leaderboard_path.exists():
        return pd.read_csv(leaderboard_path)
    return None


st.title("📊 ResumeEval Dashboard")
st.markdown(
    "Upload a processed dataset and compare multiple resume classification models."
)

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "Upload processed CSV",
        type=["csv"],
        help="CSV must contain: clean_text, label",
    )

    selected_model = st.selectbox(
        "Select model",
        options=["all", "logistic_regression", "random_forest"],
        index=0,
    )

    run_eval = st.button("🚀 Run Evaluation", use_container_width=True)

if uploaded_file is not None:
    try:
        uploaded_df = save_uploaded_dataset(uploaded_file)
        st.success("Dataset uploaded and saved successfully.")
        with st.expander("Preview uploaded dataset"):
            st.dataframe(uploaded_df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Dataset validation failed: {e}")

if run_eval:
    if not DEFAULT_DATASET_PATH.exists():
        st.error("Please upload a processed dataset first.")
    else:
        try:
            with st.spinner("Running evaluation..."):
                results = evaluate_models(
                    model_name=selected_model,
                    dataset_path=str(DEFAULT_DATASET_PATH),
                    use_docker=False,
                )
            if results:
                st.success("Evaluation completed successfully.")
            else:
                st.warning("Evaluation finished, but no results were generated.")
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

results = load_results_json()
leaderboard = load_leaderboard_csv()

if results:
    st.subheader("Raw Results")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

if leaderboard is not None and not leaderboard.empty:
    st.subheader("Leaderboard")
    st.dataframe(leaderboard, use_container_width=True)

    metric = st.selectbox(
        "Select metric to compare",
        ["accuracy", "precision", "recall", "f1", "latency_seconds"],
        index=3,
    )

    chart_df = leaderboard.set_index("model")[[metric]]
    st.subheader(f"Model Comparison: {metric}")
    st.bar_chart(chart_df)

    best_model = leaderboard.sort_values(by="f1", ascending=False).iloc[0]
    st.info(
        f"Best model by F1 score: **{best_model['model']}** "
        f"(F1 = {best_model['f1']})"
    )
else:
    st.info("No evaluation results available yet. Upload a dataset and run evaluation.")