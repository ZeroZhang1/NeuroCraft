import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.optim import SGD


class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(11, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 1)

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        o = self.act(self.hidden(x))
        return self.output(o)


if __name__ == "__main__":
    # 准备数据
    # ---读取csv文件
    data = pd.read_csv("winequality-red.csv", sep=";")
    wine_data = data.iloc[:, :-1].values
    quality = data.iloc[:, -1].values
    quality = quality * 0.1

    # ---数据归一化
    # 创建MinMaxScaler对象
    scaler = MinMaxScaler()
    # 对 wine_data 进行归一化
    wine_data_normalized = scaler.fit_transform(wine_data)

    wine_train_data = torch.tensor(wine_data_normalized[:1400])
    train_data_quality = torch.tensor(quality[:1400])

    # 定义模型
    model = MLP()
    # 将模型的参数和输入数据转换为Float
    model = model.float()

    # 将模型和数据移动到GPU（如果可用）
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01) # 随机梯度下降（Stochastic Gradient Descent，SGD）优化器

    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, labels in zip(wine_train_data, train_data_quality):
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs.float())
            loss = criterion(outputs.float(), labels.float())

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
