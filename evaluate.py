import argparse

from pipeline.evaluator import evaluate_models


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation pipeline for resume classification models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model name to evaluate or 'all'"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to processed dataset CSV"
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Run models locally instead of Docker"
    )

    args = parser.parse_args()

    results = evaluate_models(
        model_name=args.model,
        dataset_path=args.dataset,
        use_docker=not args.no_docker
    )

    if results:
        print("Evaluation completed successfully.")
    else:
        print("Evaluation completed with no results.")


if __name__ == "__main__":
    main()