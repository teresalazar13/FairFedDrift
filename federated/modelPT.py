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
        self.batch_size = 64
        self.n_epochs = 30
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(256),
            nn.Linear(256, 100)  # CIFAR-100 has 100 classes
        )

    def forward(self, x):
        return self.model(x)

    def set_weights(self, weights):
        """Set model weights from given list of NumPy arrays."""
        with torch.no_grad():  # Disable gradient tracking
            for param, weight in zip(self.model.parameters(), weights):
                if isinstance(weight, (int, float)):  # If it's a scalar, make it an array
                    weight = np.array([weight])

                weight_tensor = torch.tensor(weight, dtype=param.dtype)  # Convert to PyTorch tensor

                if weight_tensor.shape != param.shape:  # Ensure shape matches
                    raise ValueError(f"Shape mismatch: Expected {param.shape}, got {weight_tensor.shape}")

                param.copy_(weight_tensor)  # Copy into model

    def get_weights(self):
        return [param.detach().cpu().numpy() for param in self.model.parameters()]

    def compile(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, x, y):
        x = torch.tensor(x, dtype=torch.float32)  # Convert input to tensor
        y = torch.tensor(y, dtype=torch.float32)  # Convert labels to tensor
        y = torch.argmax(y, dim=1)
        y = y.to(torch.float)
        x = x.permute(0, 3, 1, 2)  # Convert (B, H, W, C) → (B, C, H, W)


        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(x)

        print(f"outputs shape: {outputs.shape}, y shape: {y.shape}")
        print(f"y dtype: {y.dtype}")  # Check dtype to ensure it's long (integers)
        print(outputs)
        print(y)

        loss = self.criterion(outputs, y)  # CrossEntropyLoss expects y as class indices (long)
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2)
        self.model.eval()

        with torch.no_grad():
            y_pred_raw = self.model(x)

        return y_pred_raw
