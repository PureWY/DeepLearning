import numpy as np
import torch
import ssl
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

train_set = mnist.MNIST('./data', train=True, download=True)
test_set = mnist.MNIST('./data', train=False, download=True)

a_data, a_label = train_set[0]
print(a_data, a_label)

a_data = np.array(a_data, dtype='float32')
print(a_data.shape)

def data_tf(x):
  x = np.array(x, dtype='float32') / 255
  x = (x - 0.5) / 0.5
  x = x.reshape((-1,))
  x = torch.from_numpy(x)
  return x

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)

a, a_label = train_set[0]

train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

a, a_label = next(iter(train_data))

net = nn.Sequential(
  nn.Linear(784, 400),
  nn.ReLU(),
  nn.Linear(400, 200),
  nn.ReLU(),
  nn.Linear(200, 100),
  nn.ReLU(),
  nn.Linear(100, 10)
)

print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1)

losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(20):
  train_loss = 0
  train_acc = 0
  net.train()

  for im, label in test_data:
    im = Variable(im)
    label = Variable(label)
    out = net(im)
    loss = criterion(out, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    _, pred = out.max(1)
    num_correct = (pred == label).sum().item()
    acc = num_correct / im.shape[0]
    train_acc += acc

  losses.append(train_loss / len(train_data))
  acces.append(train_acc / len(train_data))
  eval_loss = 0
  eval_acc = 0
  net.eval()
  for im, label in test_data:
    im = Variable(im)
    label = Variable(label)
    out = net(im)
    loss = criterion(out, label)
    eval_loss += loss.item()
    _, pred = out.max(1)
    num_correct = (pred == label).sum().item()
    acc = num_correct / im.shape[0]
    eval_acc += acc

  eval_losses.append(eval_loss / len(test_data))
  eval_acces.append(eval_acc / len(test_data))

  print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data), 
                     eval_loss / len(test_data), eval_acc / len(test_data)))

# plt.title('train loss')
# plt.plot(np.arange(len(losses)), losses)
# plt.show()


plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title('test acc')
plt.show()










