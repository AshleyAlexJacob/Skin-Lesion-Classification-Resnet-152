"""Pipeline script for evaluating the ResNet-152 skin lesion classification model."""

from __future__ import annotations

import argparse

from src.model.evaluation import ModelEvaluator


def main() -> None:
    """Entry point for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate ResNet-152 model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="artifacts/models/best_model.pth",
        help="Path to saved model weights (.pth)",
    )
    args = parser.parse_args()

    try:
        evaluator = ModelEvaluator(config_path=args.config, model_path=args.model)
        metrics, y_true, y_pred = evaluator.evaluate()
        evaluator.save_results(metrics, y_true, y_pred)
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()
