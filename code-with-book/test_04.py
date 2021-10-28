import torch
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

with open('./data.txt', 'r') as f:
  data_list = f.readlines()
  data_list = [i.split('\n')[0] for i in data_list]
  data_list = [i.split(',') for i in data_list]
  data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

# print(f.readlines())
x0 = list(filter(lambda x: x[-1] == 0.0, data))
x1 = list(filter(lambda x: x[-1] == 1.0, data))
plot_x0_0 = [i[0] for i in x0]
plot_x0_1 = [i[1] for i in x0]
plot_x1_0 = [i[0] for i in x1]
plot_x1_1 = [i[1] for i in x1]

# plt.plot(plot_x0_0, plot_x0_1, 'ro', label='x_0')
# plt.plot(plot_x1_0, plot_x1_1, 'bo', label='x_1')
# plt.legend(loc='best')
# plt.show()

np_data = np.array(data, dtype='float32') # 转换成 numpy array
x_data = torch.from_numpy(np_data[:, 0:2]) # 转换成 Tensor, 大小是 [100, 2]
y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1) # 转换成 Tensor，大小是 [100, 1]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plot_x = np.arange(-10, 10.01, 0.01)
plot_y = sigmoid(plot_x)

x_data = Variable(x_data)
y_data = Variable(y_data)

w = Variable(torch.randn(2, 1), requires_grad=True) 
b = Variable(torch.zeros(1), requires_grad=True)

def logistic_regression(x):
    return F.sigmoid(torch.mm(x, w) + b)

w0 = w[0].data[0]
w1 = w[1].data[0]
b0 = b.data[0]

plot_x = np.arange(0.2, 1, 0.01)
plot_y = (-w0 * plot_x - b0) / w1

plt.plot(plot_x, plot_y, 'g', label='cutting line')
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.legend(loc='best')

class LogisticRegression(nn.Module):
  def __init__(self):
    super(LogisticRegression, self).__init__()
    self.lr = nn.Linear(2, 1)
    self.sm = nn.Sigmoid()

  def forward(self, x):
    x = self.lr(x)
    x = self.sm(x)
    x = x.squeeze(-1)
    return x

y = nn.Linear(2, 1, bias=True)
print(y)

logistic_model = LogisticRegression()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)

for epoch in range(50000):
  x = torch.autograd.variable(x0)
  y = torch.autograd.variable(x1)

  out = logistic_model(x_data)
  loss = criterion(out, y)
  print_loss = loss.data[0]
  mask = out.ge(0.5).float()
  correct = (mask == y_data).sum()
  acc = correct.data[0] / x_data.size(0)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if(epoch + 1) % 1000 == 0:
    print('*'*10)
    print('epoch{}'.format(epoch+1))
    print('loss is {:.4f}'.format(print_loss))
    print('acc is {:.4f}'.format(acc))





