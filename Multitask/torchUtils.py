import torch
from torch.autograd import Variable
import numpy as np


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    if (isinstance(ndarray, np.ndarray)):
        return Variable(
            torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
        ).type(dtype)
    elif (torch.is_tensor(ndarray)):
        return Variable(
           ndarray, volatile=volatile, requires_grad=requires_grad
        ).type(dtype)
    elif (isinstance(ndarray, Variable)):
        Variable(ndarray.data, volatile=volatile, requires_grad=requires_grad).type=(dtype)
    else:
        raise TypeError("variable can not be made into a tensor. Requires numpy array, torch.Tensor, or Variable")

# class TimeDistributed(torch.nn.Module):
#     def __init__(self, module, batch_first=False):
#         super(TimeDistributed, self).__init__()
#         self.module = module
#         self.batch_first = batch_first

def ttd(x,layer,batch_first=True):
    if len(x.size()) <= 2:
        return layer(x)

    # Squash samples and timesteps into a single axis
    x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

    y = layer(x_reshape)

    # We have to reshape Y
    if batch_first:
        y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
    else:
        y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

    return y
