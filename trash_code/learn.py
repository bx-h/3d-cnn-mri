import torch
from torch.autograd import Variable


# tensor 放在 variable 中
# variable.data 转为 tensor
# variable.data.numpy 转为 numpy
# variable 操作 只能使用 variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])

variable = Variable(tensor, requires_grad=True)

print(tensor)

print(variable)

# tensor([[1., 2.],
#         [3., 4.]])
# tensor([[1., 2.],
#         [3., 4.]], requires_grad=True)

t_out = torch.mean(tensor * tensor)

v_out = torch.mean(variable * variable)

print(t_out)

print(v_out)

# tensor(7.5000)
# tensor(7.5000, grad_fn=<MeanBackward1>)

v_out.backward()

print(variable.grad)

# tensor([[0.5000, 1.0000],
#         [1.5000, 2.0000]])



print(variable)

print(variable.data)

print(variable.data.numpy())


# 激活函数
# 使用torch.nn.functional


import torch
import torch.nn.functional as F
from torch.autograd import Variable

x = torch.linspace(-5, 5, 200)

x = Variable(x)

x_np = x.data.numpy()

y_relu = F.relu(x).data.numpy()

import matplotlib.pyplot as plt

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')


plt.show()


# 训练网络

import torch
import matplotlib.pyplot as plt
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()