from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
import torch.nn as nn

HIDENUM = 10

# MLP model


class IrisMlp(nn.Module):
    def __init__(self):
        super(IrisMlp, self).__init__()
        self.in2hide = nn.Linear(4, HIDENUM)
        self.hide2out = nn.Linear(HIDENUM, 3)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.in2hide(x)
        x = self.activation(x)
        x = self.hide2out(x)
        return x


model = IrisMlp()


# cost function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# dataset
iris = load_iris()
X = iris.data
y = iris.target
# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
# 划分训练集和测试集 (80%训练, 20%测试)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# train
num_epochs = 100
train_losses = []
test_losses = []

model.train()
for epoch in range(num_epochs):

    # 清零梯度
    optimizer.zero_grad()

    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 记录损失
    train_losses.append(loss.item())

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 测试模式
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())

    # 每10个epoch打印一次损失
    if (epoch + 1) % 10 == 0:
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')


# test
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'测试集准确率: {accuracy*100:.2f}%')



# plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()
