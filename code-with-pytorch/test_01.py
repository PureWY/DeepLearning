import numpy as np
import torch

torch.tensor([1,2,3,4])

print(torch.tensor([1,2,3,4]).dtype)

t = torch.tensor([1,2,3,4], dtype=torch.float32)

print(t.dtype)

f = torch.tensor(range(10))

print(f)

b = torch.tensor(np.array([1,2,3,4]))

print(b.dtype)

print(torch.tensor([1.0,2.0,3.0,4.0]).dtype)

print(torch.tensor(np.array([1.0,2.0,3.0,4.0])).dtype)

o = torch.tensor([[1,2,3],[4,5,6]])
print(o)

y = torch.randn(3,3).to(torch.int)
print(y)

s = torch.rand(4,4)
print(s)

a = torch.ones(3,3,dtype=torch.int)
print(a)

b = torch.eye(3)
print(b)

c = torch.randint(0,10,(4,4))
print(c)

d = torch.rand(4,4)
e = torch.zeros_like(d)
print(d)
print(e)

f = torch.randn(3,4,5)
print(f.ndimension())
print(f.nelement())
print(f.shape)
print(f.size(0))

g = torch.randn(12)
print(g.view(3,4))