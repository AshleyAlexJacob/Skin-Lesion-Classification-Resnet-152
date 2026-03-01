"""Model evaluation logic and metrics computation for skin lesion classification."""

from __future__ import annotations

import csv
import datetime
import pathlib

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.model.resnet import SkinLesionResNet


class ModelEvaluator:
    """Evaluates a trained ResNet-152 model for skin lesion classification."""

    def __init__(
        self, config_path: str | pathlib.Path, model_path: str | pathlib.Path
    ) -> None:
        """Initialize evaluator from config and model checkpoint.

        Args:
            config_path: Path to the YAML configuration file.
            model_path: Path to the saved model state dictionary (.pth).
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self._setup_device()

        # Determine number of classes from class counts if possible
        # Otherwise wait until dataset is loaded
        class_counts = self.config.get("training", {}).get("class_counts", {})
        self.num_classes = len(class_counts) if class_counts else 7
        self.class_names = list(class_counts.keys()) if class_counts else []

        # Assuming standard ResNet transforms used in training
        img_size_cfg = self.config.get("data", {}).get("image_size", [224, 224])
        if isinstance(img_size_cfg, int):
            self.img_size = (img_size_cfg, img_size_cfg)
        else:
            self.img_size = tuple(img_size_cfg)
            if len(self.img_size) == 1:
                self.img_size = (self.img_size[0], self.img_size[0])

        self.batch_size = self.config.get("training", {}).get("batch_size", 32)

        self.dataset_loader = self.load_dataset()
        self.model = self.load_model(model_path)

    def _setup_device(self) -> None:
        """Set up the compute device according to config."""
        device_cfg = self.config.get("training", {}).get("device", "auto")
        if device_cfg == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device_cfg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

    def load_dataset(self) -> DataLoader:
        """Load the evaluation dataset, falling back to validation if test is absent.

        Returns:
            A PyTorch DataLoader containing the evaluation data.

        Raises:
            ValueError: If neither testing nor validation directories are found.
        """
        data_cfg = self.config.get("data", {})
        test_dir = pathlib.Path(data_cfg.get("test_dir", "data/processed/test"))
        val_dir = pathlib.Path(data_cfg.get("val_dir", "data/processed/val"))

        eval_dir = None
        if test_dir.exists() and test_dir.is_dir() and any(test_dir.iterdir()):
            print(f"Found test dataset at {test_dir}")
            eval_dir = test_dir
        elif val_dir.exists() and val_dir.is_dir() and any(val_dir.iterdir()):
            print(
                f"Test dataset not found. Falling back to validation dataset at {val_dir}"
            )
            eval_dir = val_dir

        if eval_dir is not None:
            data_transforms = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            dataset = datasets.ImageFolder(root=str(eval_dir), transform=data_transforms)
            dataset_classes = dataset.classes

            subset_fraction = self.config.get("training", {}).get("subset_fraction", 1.0)
            if subset_fraction < 1.0:
                subset_size = max(1, int(len(dataset) * subset_fraction))
                dataset = torch.utils.data.Subset(dataset, range(subset_size))

            # Update class info from dataset if not already set or if it's more accurate
            if not self.class_names or len(self.class_names) != len(dataset_classes):
                self.class_names = dataset_classes
                self.num_classes = len(self.class_names)

            num_workers = self.config.get("training", {}).get("num_workers", 4)

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            return dataloader
            
        print("Test/val directories not populated. Falling back to src.data.loader.get_data_loaders using base data_dir.")
        from src.data.loader import get_data_loaders
        data_dir = data_cfg.get("data_dir", "data/processed")
        num_workers = self.config.get("training", {}).get("num_workers", 4)
        subset_fraction = self.config.get("training", {}).get("subset_fraction", 1.0)
        
        _, test_loader, class_names = get_data_loaders(
            data_dir=data_dir,
            batch_size=self.batch_size,
            img_size=self.img_size,
            num_workers=num_workers,
            subset_fraction=subset_fraction,
        )
        
        if not self.class_names or len(self.class_names) != len(class_names):
            self.class_names = class_names
            self.num_classes = len(self.class_names)
            
        return test_loader

    def load_model(self, model_path: str | pathlib.Path) -> SkinLesionResNet:
        """Load the pre-trained ResNet model from a saved state dictionary.

        Args:
            model_path: Path to the .pth checkpoint file.

        Returns:
            The loaded SkinLesionResNet model ready for inference.

        Raises:
            FileNotFoundError: If the model file is not found.
        """
        path = pathlib.Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"No saved model found at {path}")

        model = SkinLesionResNet(
            num_classes=self.num_classes,
            pretrained=False,  # We load our own weights
        )
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"Model successfully loaded from {path}")
        return model

    def evaluate(self) -> tuple[dict[str, float], list[int], list[int]]:
        """Run the evaluation loop across the dataset.

        Returns:
            A tuple containing:
                - metrics: Dictionary of calculated metrics (e.g., accuracy).
                - y_true: List of ground truth class indices.
                - y_pred: List of predicted class indices.
        """
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        print("Starting evaluation...")
        with torch.no_grad():
            eval_pbar = tqdm(self.dataset_loader, desc="Evaluating", leave=True)
            for inputs, labels in eval_pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predicted.cpu().tolist())

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        metrics = {"accuracy": accuracy}

        print(f"Evaluation completed. Accuracy: {accuracy:.2f}%")
        return metrics, y_true, y_pred

    def save_results(
        self, metrics: dict[str, float], y_true: list[int], y_pred: list[int]
    ) -> pathlib.Path:
        """Save evaluation metrics and individual predictions to a CSV file.

        Saves to artifacts/results/testing/results_DD_MM_YYYY.csv.

        Args:
            metrics: Dictionary of aggregated metrics.
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            The path to the saved CSV file.
        """
        results_dir = pathlib.Path("artifacts/results/testing")
        results_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.datetime.now().strftime("%d_%m_%Y")
        csv_filename = f"results_{today}.csv"
        csv_path = results_dir / csv_filename

        print(f"Saving results to {csv_path}...")
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)

            # Write metrics header
            writer.writerow(["Metric", "Value"])
            for key, val in metrics.items():
                writer.writerow([key, f"{val:.4f}"])

            writer.writerow([])  # Empty row separator

            # Write predictions header
            writer.writerow(
                [
                    "Sample_Index",
                    "True_Label_Name",
                    "Predicted_Label_Name",
                    "True_Class_Idx",
                    "Predicted_Class_Idx",
                    "Correct",
                ]
            )

            for i, (true_idx, pred_idx) in enumerate(zip(y_true, y_pred)):
                true_name = (
                    self.class_names[true_idx]
                    if true_idx < len(self.class_names)
                    else str(true_idx)
                )
                pred_name = (
                    self.class_names[pred_idx]
                    if pred_idx < len(self.class_names)
                    else str(pred_idx)
                )
                is_correct = "Yes" if true_idx == pred_idx else "No"

                writer.writerow(
                    [i, true_name, pred_name, true_idx, pred_idx, is_correct]
                )

        print("Results saved successfully.")
        return csv_path
