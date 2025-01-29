import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report


# 简单神经网络定义
class PneumoniaNN(nn.Module):
    def __init__(self, input_size):
        super(PneumoniaNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


def train_neural_network(x_train, x_test, y_train, y_test):
    x_train, x_test = torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(X_test.values,
                                                                                      dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1), torch.tensor(y_test.values,
                                                                                                  dtype=torch.float32).view(
        -1, 1)

    model = PneumoniaNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred = model(x_test).round()
        accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())
        print("神经网络准确率:", accuracy)
        print("分类报告:")
        print(classification_report(y_test.numpy(), y_pred.numpy()))
    return model
