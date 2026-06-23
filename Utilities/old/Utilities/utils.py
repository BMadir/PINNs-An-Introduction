import torch
import numpy
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from torch.func import functional_call


__all__ = [
    "vect_parameters_to_dic", "vector_to_parameters", "parameters_to_vector", "functional_call",
    "evaluate"
]

def vect_parameters_to_dic(vect_parameters, dict_parameters):
    if not torch.is_tensor(vect_parameters):
        vect_parameters = torch.Tensor(vect_parameters)
    offset = 0
    for n, p in dict_parameters.items():
        numel = p.numel()
        dict_parameters[n] = vect_parameters[offset : offset + numel].reshape(p.shape)
        offset += numel
    assert offset == vect_parameters.numel(), 'invalid size for vect_parameters'
    return dict_parameters


def evaluate(self, *args):
    inputs = list()
    for arg in args:
        if not torch.is_tensor(arg):
            arg = torch.Tensor(arg)
        inputs.append(arg.to(self.device))
    outputs = self.net(*inputs)
    if torch.is_tensor(outputs):
        return outputs.detach().cpu().numpy()
    else:
        return tuple(out.detach().cpu().numpy() for out in outputs)