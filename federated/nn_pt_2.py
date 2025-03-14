import torch
import numpy as np


class NNPT2:
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
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
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
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 256)
        self.fc2 = torch.nn.Linear(256, 10)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, start_dim=1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
