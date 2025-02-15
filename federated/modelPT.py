import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms


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
        with torch.no_grad():
            for param, weight in zip(self.model.parameters(), weights):
                param.copy_(torch.tensor(weight, dtype=param.dtype))

    def get_weights(self):
        return [param.detach().cpu().numpy() for param in self.model.parameters()]

    def compile(self):
        if self.dataset.is_large:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        self.criterion = nn.BCELoss() if self.dataset.is_binary_target else nn.CrossEntropyLoss()

    def learn(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)
