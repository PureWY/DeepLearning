import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

data = pd.read_csv('./HR.csv')

print(data.groupby(['salary', 'part']).size())

data = data.join(pd.get_dummies(data.salary))
data = data.join(pd.get_dummies(data.part))

del data['salary']
del data['part']

# print(data.head())

print(data.left.value_counts())

Y_data = data.left.values.reshape(-1, 1)
print(Y_data.shape)

Y = torch.from_numpy(Y_data).type(torch.float32)

print([c for c in data.columns if c != 'left'])

X_data = data[[c for c in data.columns if c != 'left']].values

X = torch.from_numpy(X_data).type(torch.float32)

print(X.size())

#  创建模型-自定义模型
#  nn.Module  继承这个类
#  __init__   初始化所有的层
#  forward    定义模型的运算过程（前向传播的过程）

# class Model(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.liner_1 = nn.Linear(20, 64)
#     self.liner_2 = nn.Linear(64, 64)
#     self.liner_3 = nn.Linear(64, 1)
#     self.relu = nn.ReLU()
#     self.sigmoid = nn.Sigmoid()
  
#   def forward(self, input):
#     x = self.liner_1(input)
#     x = self.relu(x)
#     x = self.liner_2(input)
#     x = self.relu(x)
#     x = self.liner_3(x)
#     x = self.sigmoid(x)
#     return x

#  使用F改写
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.liner_1 = nn.Linear(20, 64)
    self.liner_2 = nn.Linear(64, 64)
    self.liner_3 = nn.Linear(64, 1)
  
  def forward(self, input):
    x = F.relu(self.liner_1(input))
    x = F.relu(self.liner_2(x))
    x = F.sigmoid(self.liner_3(x))
    return x

model = Model()

print(model)

lr = 0.0001

def get_model():
  model = Model()
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  return model, opt

model, optim = get_model()

#  定义损失函数
loss_fn = nn.BCELoss()

batch = 64
no_of_batches = len(data)//batch
epochs = 100

# for epoch in range(epochs):
#   for i in range(no_of_batches):
#     start = i*batch
#     end = start + batch
#     x = X[start: end]
#     y = Y[start: end]
#     y_pred = model(x)
#     loss = loss_fn(y_pred, y)
#     optim.zero_grad()
#     loss.backward()
#     optim.step()

# print(loss_fn(model(X), Y))

HR_ds = TensorDataset(X, Y)
HR_dl = DataLoader(HR_ds, batch_size=batch, shuffle=True)

for epoch in range(epochs):
  for x, y in HR_dl:
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optim.zero_grad()
    loss.backward()
    optim.step()

print(loss_fn(model(X), Y))


