import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn.modules import loss

x_train = np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],[9.779],[6.182],[7.59],[2.167],[7.042],[10.791],[5.313],[7.997],[3.1]], dtype=np.float32)
y_train = np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.572],[3.366],[2.596],[2.53],[1.221],[2.827],[3.465],[1.65],[2.904],[1.3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

class LinearRegression(nn.Module):
  def __init__(self):
    super(LinearRegression, self).__init__()
    self.linear = nn.Linear(1, 1)

  def forward(self, x):
    out = self.linear(x)
    return out

if torch.cuda.is_available():
  model = LinearRegression().cude()
else:
  model = LinearRegression()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

num_epochs = 1000
for epoch in range(num_epochs):
  inputs = torch.autograd.Variable(x_train)
  target = torch.autograd.Variable(y_train)

  out = model(inputs)
  loss = criterion(out, target)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # if (epoch+1) % 20 == 0:
    # print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1, num_epochs, loss.data[0]))

model.eval()
predict = model(torch.autograd.Variable(x_train))
predict = predict.data.numpy()
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Predict line')
plt.show()

