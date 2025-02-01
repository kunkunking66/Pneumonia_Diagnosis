import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.optim.lr_scheduler import StepLR

# 更复杂的神经网络定义
class PneumoniaNN(nn.Module):
    def __init__(self, input_size):
        super(PneumoniaNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),  # 增加神经元数量
            nn.LeakyReLU(0.1),  # 使用LeakyReLU激活函数
            nn.Dropout(0.3),  # 添加Dropout防止过拟合
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


def train_neural_network(X_train, X_test, y_train, y_test):
    # 将数据转换为PyTorch张量
    X_train, X_test = torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(X_test.values, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1), torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # 初始化模型、损失函数和优化器
    model = PneumoniaNN(X_train.shape[1])
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 添加L2正则化
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率调度器

    # 训练模型
    for epoch in range(100):  # 增加训练轮数
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率

        # 打印每个epoch的损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

    # 评估模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).round()
        accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())
        print("神经网络准确率:", accuracy)
        print("分类报告:")
        print(classification_report(y_test.numpy(), y_pred.numpy()))

    return model


# 示例调用
# 假设你已经加载了数据并进行了预处理
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = train_neural_network(X_train, X_test, y_train, y_test)