import torch
import torchvision.models as models
import numpy as np


class NNPT:
    def __init__(self, dataset, seed):
        torch.manual_seed(seed)
        self.batch_size = 128
        self.n_epochs = 30
        self.model = NNPTLarge(dataset)
        self.model = self.model.to('cuda')

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def get_weights(self):
        return self.model.state_dict()

    def compile(self, _):
        pass

    def learn(self, x_, y_):
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        x_tensor = torch.tensor(x_, dtype=torch.float32)
        x_tensor = x_tensor.permute(0, 3, 1, 2)
        x_tensor = x_tensor.to('cuda')
        y_tensor = torch.tensor(np.argmax(y_, axis=-1), dtype=torch.long)
        y_tensor = y_tensor.to('cuda')
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.n_epochs):
            for batch, (X, y) in enumerate(dataloader):
                self.model.zero_grad()
                pred = self.model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
                optimizer.step()

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            x_tensor = x_tensor.permute(0, 3, 1, 2)
            x_tensor = x_tensor.to('cuda')
            y_pred_raw = self.model(x_tensor)
            y_pred_softmax = torch.nn.functional.softmax(y_pred_raw, dim=1)

            return y_pred_softmax.cpu().detach().numpy()


class NNPTLarge(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.shufflenet = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)

        # Freeze all parameters
        for param in self.shufflenet.parameters():
            param.requires_grad = False

        # Unfreeze BatchNorm layers
        for name, layer in self.shufflenet.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                for param in layer.parameters():
                    param.requires_grad = True

        # Modify the classifier
        self.shufflenet.fc = torch.nn.Sequential(
            torch.nn.Linear(self.shufflenet.fc.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, dataset.n_classes)
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.shufflenet(x)
        return x