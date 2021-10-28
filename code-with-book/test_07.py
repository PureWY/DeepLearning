import os
import torch
import pandas as pd

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
  f.write('NumRooms,Alley,Price\n')  #列名
  f.write('NA,Pave,127500\n')
  f.write('2,NA,106000\n')
  f.write('4,NA,178100\n')
  f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

# 处理缺失数据-填充
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())  # fillna:填补not a number部分 mean():均值
print(inputs)

# 对于inputs中的类别值或者离散值，我们将NAN视为一个类别
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x, y)
