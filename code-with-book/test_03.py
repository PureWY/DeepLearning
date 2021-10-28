import torch
from torch import nn
from torch.nn.modules import loss

def make_features(x):
  x = x.unsqueeze(1)
  return torch.cat([x ** i for i in range(1, 4)], 1)

W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])

def f(x):
  return x.mm(W_target) + b_target[0]

def get_batch(batch_size=32):
  random = torch.randn(batch_size)
  x = make_features(random)
  y = f(x)
  return torch.autograd.Variable(x), torch.autograd.Variable(y)

class poly_model(nn.Module):
  def __init__(self):
    super(poly_model, self).__init__()
    self.poly = nn.Linear(3, 1)

  def forward(self, x):
    out = self.poly(x)
    return out

model = poly_model()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
while True:
  batch_x, batch_y = get_batch()
  output = model(batch_x)
  loss = criterion(output, batch_y)
  print_loss = loss.item()
  print(print_loss)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  epoch += 1
  if print_loss < 1e-3:
    break