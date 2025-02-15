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
        x = torch.tensor(x, dtype=torch.float32)  # Convert input to tensor
        y = torch.tensor(y, dtype=torch.float32)  # Convert labels to tensor
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension (B, 1, H, W)
        x = x.permute(0, 3, 1, 2)  # Convert (B, H, W, C) → (B, C, H, W)

        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(x)

        if y.dim() > 1:
            y = y.argmax(dim=1)  # Convert from (B, num_classes) to (B,)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension (B, 1, H, W)
        x = x.permute(0, 3, 1, 2)

        self.model.eval()

        with torch.no_grad():
            y_pred_raw = self.model(x)

        return y_pred_raw
