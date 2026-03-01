"""Pipeline script for training the ResNet-152 skin lesion classification model."""

from __future__ import annotations

import argparse
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from src.data.loader import get_data_loaders
from src.model.resnet import SkinLesionResNet


class ModelTrainer:
    """Trains the ResNet-152 model for skin lesion classification."""

    def __init__(self, config_path: str | pathlib.Path) -> None:
        """Initialize trainer from config.

        Args:
            config_path: Path to the YAML configuration file.
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self._setup_device()

        self.epochs = self.config["training"]["epochs"]
        self.learning_rate = self.config["training"]["learning_rate"]

        # Data loaders
        data_dir = self.config.get("data", {}).get("data_dir", "data/processed")
        batch_size = self.config["training"]["batch_size"]
        subset_fraction = self.config.get("training", {}).get("subset_fraction", 1.0)

        self.train_loader, self.val_loader, self.class_names = get_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            subset_fraction=subset_fraction,
        )

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        self._setup_model()
        self._setup_loss_and_optimizer()

        self.best_val_acc = 0.0
        self.artifacts_dir = pathlib.Path("artifacts/models")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _setup_device(self) -> None:
        """Set up the compute device according to config."""
        device_cfg = self.config["training"]["device"]
        if device_cfg == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device_cfg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

    def _setup_model(self) -> None:
        """Initialize the ResNet model and configure frozen layers."""
        model_cfg = self.config["model"]
        self.model = SkinLesionResNet(
            num_classes=self.num_classes,
            pretrained=model_cfg.get("pretrained", True),
        )

        if model_cfg.get("freeze_layers", True):
            self.model.freeze_layers(unfreeze_fc=True)

        unfreeze_blocks = model_cfg.get("unfreeze_blocks", 0)
        if unfreeze_blocks > 0:
            self.model.unfreeze_blocks(unfreeze_blocks)

        self.model = self.model.to(self.device)

    def _setup_loss_and_optimizer(self) -> None:
        """Set up weighted cross entropy loss and optimizer."""
        train_cfg = self.config["training"]

        weights = None
        if train_cfg.get("use_class_weights", False):
            class_counts = train_cfg.get("class_counts", {})
            if class_counts:
                # Calculate weights inversely proportional to target class frequencies
                total_samples = sum(class_counts.values())
                weight_list = []
                for name in self.class_names:
                    # Default heavily if class missing to prevent imbalance dominance
                    count = class_counts.get(name.lower(), 1)
                    weight_list.append(total_samples / count)

                weights_tensor = torch.tensor(weight_list, dtype=torch.float32)
                weights = weights_tensor.to(self.device)
                print(f"Using class weights: {weight_list}")

        self.criterion = nn.CrossEntropyLoss(weight=weights)

        # Only pass parameters that require gradient to the optimizer
        params_to_update = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(params_to_update, lr=self.learning_rate)

    def apply_minority_augmentation(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Apply simple batched augmentations to minority classes.

        Args:
            inputs: Batch of image tensors.
            labels: Batch of labels.

        Returns:
            Augmented image tensors.
        """
        minority_classes = self.config.get("training", {}).get("minority_classes", [])
        if not minority_classes:
            return inputs

        # Create mask for minority samples
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for cls_name in minority_classes:
            if cls_name in self.class_to_idx:
                idx = self.class_to_idx[cls_name]
                mask |= labels == idx

        if mask.any():
            minority_inputs = inputs[mask]

            # 1. Random horizontal flip
            flip_mask = torch.rand(minority_inputs.shape[0], device=self.device) > 0.5
            minority_inputs[flip_mask] = torch.flip(
                minority_inputs[flip_mask], dims=[-1]
            )

            # 2. Random Gaussian noise to simulate variation
            noise = torch.randn_like(minority_inputs) * 0.05
            minority_inputs = minority_inputs + noise

            inputs[mask] = minority_inputs

        return inputs

    def train(self) -> None:
        """Execute the training loop."""
        print(f"Starting training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            train_pbar = tqdm(
                self.train_loader,
                desc=f"Epoch [{epoch + 1}/{self.epochs}] Training",
                leave=False,
            )
            for inputs, labels in train_pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Apply batched augmentation for minority classes
                inputs = self.apply_minority_augmentation(inputs, labels)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss = running_loss / len(self.train_loader)
            train_acc = 100.0 * correct / total

            # Validation step
            val_loss, val_acc = self._validate()

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
            )

            if val_acc > self.best_val_acc:
                print(
                    f"Validation accuracy improved from {self.best_val_acc:.2f}% "
                    f"to {val_acc:.2f}%. Saving model..."
                )
                self.best_val_acc = val_acc
                self.save_model()

        print("Training completed.")

    def _validate(self) -> tuple[float, float]:
        """Run evaluation on the validation set.

        Returns:
            Tuple of (validation_loss, validation_accuracy).
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            val_pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for inputs, labels in val_pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100.0 * correct / total if total > 0 else 0.0
        return val_loss, val_acc

    def save_model(self, filename: str = "best_model.pth") -> None:
        """Save the model's state dictionary to the artifacts directory.

        Args:
            filename: Name of the saved model file.
        """
        save_path = self.artifacts_dir / filename
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, filename: str = "best_model.pth") -> None:
        """Load the model's state dictionary from the artifacts directory.

        Args:
            filename: Name of the saved model file.
        """
        load_path = self.artifacts_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"No saved model found at {load_path}")
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        print(f"Model loaded from {load_path}")


def main() -> None:
    """Entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train ResNet-152 model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    try:
        trainer = ModelTrainer(config_path=args.config)
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    main()
