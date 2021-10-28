import torch

x = torch.randn(3)
x = torch.autograd.Variable(x, requires_grad=True)

y = x*2
print(y)

y.backward(torch.FloatTensor([1, 0.1, 0.01]))
print(x.grad)


A = [1]
print(type(A))

A = torch.tensor(A)
print(type(A))