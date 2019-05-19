import torch
from torch.autograd import Variable

tensor = torch.FloatTensor()

variable = Variable(tensor, requires_grad=True)
