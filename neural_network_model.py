import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


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


def train_neural_network(X_train, X_test, y_train, y_test):
    X_train, X_test = torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(X_test.values,
                                                                                      dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1), torch.tensor(y_test.values,
                                                                                                  dtype=torch.float32).view(
        -1, 1)

    model = PneumoniaNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)  # 确保形状一致
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred = model(X_test).round()
        y_pred_raw = model(X_test).detach().numpy()  # 获取原始预测概率值
        accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())
        print("神经网络准确率:", accuracy)
        print("分类报告:")
        print(classification_report(y_test.numpy(), y_pred.numpy()))

        # 绘制 ROC 曲线
        fpr, tpr, thresholds = roc_curve(y_test.numpy(), y_pred_raw)
        roc_auc = roc_auc_score(y_test.numpy(), y_pred_raw)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    return model
