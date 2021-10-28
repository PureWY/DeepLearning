import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import nn

data = pd.read_csv('./Income1.csv')

# plt.scatter(data.Education, data.Income)
# plt.xlabel('Education')
# plt.ylabel('Income')
# plt.show()

# 数据预处理
X = torch.from_numpy(data.Education.values.reshape(-1, 1).astype(np.float32))
Y = torch.from_numpy(data.Income.values.reshape(-1, 1).astype(np.float32))
# print(X)
# print(Y)

# 手动初始化w与b
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

learning_rate = 0.0001

# 模型的公式 w@x + b
for epoch in range(5000):
  for x, y in zip(X, Y):
    y_pred = torch.matmul(x, w) + b
    loss = (y - y_pred).pow(2).mean()
    if not w.grad is None:
      w.grad.data.zero_()
    loss.backward()
    with torch.no_grad():
      w.data -= w.grad.data*learning_rate
      b.data -= b.grad.data*learning_rate

plt.scatter(data.Education, data.Income)
plt.plot(X.numpy(), (X*w + b).data.numpy(), c='r')
plt.show()


# model = nn.Linear(1, 1) # w@input + b 等价于model(input)
# loss_fn = nn.MSELoss() # 损失函数
# opt = torch.optim.SGD(model.parameters(), lr=0.0001)  # SGD:随机梯度下降算法 lr：优化参数

# for epoch in range(5000):
#   for x, y in zip(X, Y):
#     y_pred = model(x)         # 使用模型预测
#     loss = loss_fn(y, y_pred)     # 根据预测结果计算损失
#     opt.zero_grad()           # 把变量梯度清零
#     loss.backward()           # 求解梯度
#     opt.step()                # 优化模型参数

# print(model.weight)
# print(model.bias)

# plt.scatter(data.Education, data.Income)
# plt.plot(X.numpy(), model(X).data.numpy())
# plt.show()