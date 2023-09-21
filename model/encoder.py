import torch
import torch.nn as nn
from torchvision import models


class CNNModel(nn.Module):
    def __init__(self, embedding_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(CNNModel, self).__init__()
        weights = models.ResNet152_Weights.DEFAULT
        resnet = models.resnet152(weights=weights)
        module_list = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet_module = nn.Sequential(*module_list)

        # Freeze or unfreeze ResNet parameters
        for param in self.resnet_module.parameters():
            param.requires_grad = False

        self.linear_layer = nn.Linear(resnet.fc.in_features, embedding_size)
        self.batch_norm = nn.BatchNorm1d(embedding_size, momentum=0.01)

    def forward(self, input_images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            resnet_features = self.resnet_module(input_images)
        resnet_features = resnet_features.reshape(resnet_features.size(0), -1)
        final_features = self.batch_norm(self.linear_layer(resnet_features))
        return final_features
