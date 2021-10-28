import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

data = pd.read_csv('./credit-a.csv', header=None)

# print(data.info())

X = data.iloc[:, :-1]   # iloc:按位置取值
Y = data.iloc[:, -1].replace(-1, 0)
# print(Y.unique())

X = torch.from_numpy(X.values).type(torch.float32)
Y = torch.from_numpy(Y.values.reshape(-1, 1)).type(torch.float32)
# print(Y.shape)

model = nn.Sequential(
  nn.Linear(15, 1),
  nn.Sigmoid()
)

print(model)

loss_fn = nn.BCELoss()  # 二院交叉熵损失函数

opt = torch.optim.Adam(model.parameters(), lr=0.0001)

batches = 16
no_of_batch = 653//16

epoches = 1000

for epoch in range(epoches):
  for i in range(no_of_batch):
    start = i*batches
    end = start + batches
    x = X[start:end]
    y = Y[start:end]
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

model.state_dict()    # 相当于sigmoid(w1*x1 + w2*x2 + w3*x3 + ... w15*x15 + b)

print(((model(X).data.numpy() > 0.5).astype('int') == Y.numpy()).mean())


