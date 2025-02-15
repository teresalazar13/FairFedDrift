import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class NNModel(nn.Module):
    def __init__(self, dataset, seed):
        super(NNModel, self).__init__()
        torch.manual_seed(seed)

        self.dataset = dataset
        self.batch_size = 10 if not dataset.is_image else (64 if dataset.is_large else 32)
        self.n_epochs = 10 if not dataset.is_image else (30 if dataset.is_large else 5)

        if not dataset.is_image:  # Adult dataset (Tabular)
            self.model = nn.Sequential(
                nn.Linear(dataset.input_shape[0], 10),
                nn.Tanh(),
                nn.Linear(10, 1),
                nn.Sigmoid()
            )
        else:
            if dataset.is_large:  # CIFAR-100 (ResNet18)
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                self.model.fc = nn.Sequential(
                    nn.Linear(self.model.fc.in_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, 100)  # CIFAR-100 has 100 classes
                )
            else:  # MNIST / FEMNIST
                self.model = nn.Sequential(
                    nn.Conv2d(in_channels=dataset.input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Flatten(),
                    nn.Linear(32 * (dataset.input_shape[1] // 2) * (dataset.input_shape[2] // 2), 100),
                    nn.ReLU(),
                    nn.Linear(100, 10),
                    nn.Softmax(dim=1)
                )

    def forward(self, x):
        return self.model(x)

    def set_weights(self, weights):
        """Set model weights from a given list of NumPy arrays or tensors."""
        with torch.no_grad():  # Disable gradient tracking
            for param, weight in zip(self.model.parameters(), weights):
                # Convert NumPy array or scalar to a PyTorch tensor
                if isinstance(weight, np.ndarray) or isinstance(weight, (int, float)):
                    weight_tensor = torch.as_tensor(weight, dtype=param.dtype, device=param.device)
                elif isinstance(weight, torch.Tensor):
                    weight_tensor = weight.to(dtype=param.dtype, device=param.device)
                else:
                    raise TypeError(f"Unsupported weight type: {type(weight)}")

                # Ensure weight shape matches model parameter shape
                if weight_tensor.shape != param.shape:
                    raise ValueError(f"Shape mismatch: Expected {param.shape}, got {weight_tensor.shape}")

                param.copy_(weight_tensor)  # Update model weights

    def get_weights(self):
        return [param.detach().cpu().numpy() for param in self.model.parameters()]

    def compile(self):
        if self.dataset.is_large:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        self.criterion = nn.BCELoss() if self.dataset.is_binary_target else nn.CrossEntropyLoss()

    def learn(self, x, y):
        # Convert input and labels to tensors with the correct dtype
        x = torch.tensor(x, dtype=torch.float32)  # Convert input to tensor (B, H, W, C)
        y = torch.tensor(y, dtype=torch.int64)  # Convert labels to tensor (B, ) or (B, num_classes)

        # Handle input tensor dimensions (if 3D, add the channel dimension)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension (B, 1, H, W)

        # Permute the input tensor to match the required shape for CNNs (B, C, H, W)
        x = x.permute(0, 3, 1, 2)  # Convert (B, H, W, C) → (B, C, H, W)

        self.model.train()  # Set the model to training mode
        self.optimizer.zero_grad()  # Clear previous gradients

        # Forward pass through the model
        outputs = self.model(x)  # outputs shape should be (B, num_classes) for classification

        # If the target `y` is one-hot encoded, convert it to class indices
        if y.dim() > 1:  # If y is one-hot encoded (B, num_classes)
            y = y.argmax(dim=1)  # Convert to class indices (B, )

        # Ensure the target tensor is of type Long (torch.int64), which is expected by cross_entropy
        y = y.long()

        # Compute the loss
        loss = self.criterion(outputs, y)  # CrossEntropy expects (B, num_classes) for outputs and (B,) for targets

        # Backward pass
        loss.backward()
        self.optimizer.step()  # Update the model parameters

        return loss.item()  # Return the scalar loss value for logging/monitoring

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension (B, 1, H, W)
        x = x.permute(0, 3, 1, 2)

        self.model.eval()

        with torch.no_grad():
            y_pred_raw = self.model(x)

        return y_pred_raw
