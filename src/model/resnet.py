from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights


class SkinLesionResNet(nn.Module):
    """ResNet-152 model adapted for skin lesion classification."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True) -> None:
        """Initialize the SkinLesionResNet model.

        Args:
            num_classes: The number of classes to predict.
            pretrained: Whether to load pre-trained ImageNet weights.
        """
        super().__init__()

        if pretrained:
            weights = ResNet152_Weights.DEFAULT
            self.model = resnet152(weights=weights)
        else:
            self.model = resnet152(weights=None)

        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Output logits of shape (batch, num_classes).
        """
        return self.model(x)

    def freeze_layers(self, unfreeze_fc: bool = True) -> None:
        """Freeze all layers in the model.

        Args:
            unfreeze_fc: If True, keeps the final fully connected layer unfrozen.
        """
        for param in self.model.parameters():
            param.requires_grad = False

        if unfreeze_fc:
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def unfreeze_blocks(self, num_blocks: int) -> None:
        """Unfreeze the last `num_blocks` of the ResNet architecture.

        ResNet has 4 main layer blocks (layer1, layer2, layer3, layer4).
        This method will unfreeze from layer4 backwards.

        Args:
            num_blocks: Number of layer blocks to unfreeze (1 to 4).
        """
        layers_to_unfreeze = []
        if num_blocks >= 1:
            layers_to_unfreeze.append(self.model.layer4)
        if num_blocks >= 2:
            layers_to_unfreeze.append(self.model.layer3)
        if num_blocks >= 3:
            layers_to_unfreeze.append(self.model.layer2)
        if num_blocks >= 4:
            layers_to_unfreeze.append(self.model.layer1)

        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

    def finetune(self) -> None:
        """Unfreeze all layers for finetuning."""
        for param in self.parameters():
            param.requires_grad = True
