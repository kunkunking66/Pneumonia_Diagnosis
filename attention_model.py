import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset


# 注意力机制层
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn_weights = nn.Linear(input_dim, input_dim)  # 输出维度与输入一致

    def forward(self, x):
        attn_scores = torch.sigmoid(self.attn_weights(x))  # [batch_size, input_dim]
        return x * attn_scores  # 特征加权，保持形状 [batch_size, input_dim]


# 基于注意力机制的神经网络
class AttentionModel(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)  # 输入 398 维，输出 32 维
        self.attention = Attention(32)       # 保持 32 维
        self.fc2 = nn.Linear(32, 1)          # 输出 1 维
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # [batch_size, 32]
        x = self.attention(x)        # [batch_size, 32]
        x = self.fc2(x)              # [batch_size, 1]
        x = self.sigmoid(x)          # [batch_size, 1]
        return x


def train_attention_model(X_train, X_test, y_train, y_test, epochs=50, batch_size=32):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    model = AttentionModel(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)  # 输出形状 [batch_size, 1]
            loss = criterion(outputs, batch_y)  # 输入和目标形状均为 [batch_size, 1]
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy().flatten()
        y_pred_labels = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_labels)
    print("注意力机制模型准确率:", accuracy)
    print("分类报告:")
    print(classification_report(y_test, y_pred_labels))

    # 保存模型到上一级目录的 model 文件夹
    import os
    save_path = './model/attention.pth'

    # 确保保存路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")

    return model
