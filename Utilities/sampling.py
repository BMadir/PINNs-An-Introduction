"""
Sampling and adaptive Sampling for PINNs

This module provides:
- Sampler: Latin hypercube / Sobol / Halton / Hammersley sampling in bounded domains
- Join_samplers: combination of multiple sampling strategies with weighted proportions
- Adaptor: residual-based adaptive refinement (RAR) and density-based sampling (RAD)
- Data_sampler: sampling from fixed datasets with optional importance weighting

Key features:
- Supports NumPy and PyTorch outputs
- Adaptive sampling based on model residuals
- Integration with PINNs and PDE training pipelines
- Flexible composition of multiple sampling strategies
"""


import numpy as np
import torch
import skopt
import copy

from utils import  torch_out

__all__ = ["Sampler", "Join_samplers", "Adaptor", "Data_sampler"]


### Helper functions:
def _rar_fn(sample_fn, func, num_in, num_out, **kwargs):
    """ Residual-based adaptive refinement """
    *sample, output = sample_fn(num_in, **kwargs)
    abs_value = func(*sample).abs()
    with torch.no_grad():
        values, indices = torch.topk(abs_value, num_out, dim=0)
        indices = indices.flatten()
        return *[s[indices] for s in sample], output[indices]


def _rad_fn(sample_fn, func, num_in, num_out, k, c, seed=1234, **kwargs):
    # generator = torch.Generator("cuda").manual_seed(seed)
    try:
        generator = kwargs.pop("generator")
    except:
        generator = None

    *sample, output = sample_fn(num_in, **kwargs)
    abs_value = func(*sample).abs().flatten()
    with torch.no_grad():
        density = (abs_value ** k) / (abs_value ** k).mean() + c
        density /= density.sum()
        indices = torch.multinomial(density, num_out, generator=generator)
        indices = indices.flatten()
        return *[s[indices] for s in sample], output[indices]


### Sampler class
class Sampler:
    def __init__(self, l_bounds, u_bounds, func=None, sampler='lhs', seed=1234):

        assert len(l_bounds) == len(u_bounds)
        self.dim = len(l_bounds)
        self.bounds = l_bounds, u_bounds
        self.sampler = self.__sampler_fn(sampler.lower())
        self.unit_sq = [(0., 1.) for l, u in zip(l_bounds, u_bounds)]
        if func is None:
            func = lambda *args: np.zeros_like(args[0])
        self.func = func
        self.seed = seed

    @staticmethod
    def __sampler_fn(sampler):
        if sampler == 'lhs':
            sampler_fn = skopt.sampler.Lhs(lhs_type='centered', criterion='maximin', iterations=0)
        elif sampler == 'sobol':
            sampler_fn = skopt.sampler.Sobol(skip=0, randomize=False)
        elif sampler == 'halton':
            sampler_fn = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
        elif sampler == 'hammersly':
            sampler_fn = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
        else:
            raise Exception('unsupported sampler')
        return sampler_fn

    def scale(self, sample):
        lower = np.broadcast_to(self.bounds[0], self.dim)
        upper = np.broadcast_to(self.bounds[1], self.dim)
        return sample * (upper - lower) + lower

    @staticmethod
    def _Array_to_Tensor(list_A, device, half):
        list_T = []
        for a in list_A:
            t = torch.Tensor(a).to(device)
            if half:
                t = t.half()
            list_T.append(t)
        return tuple(list_T)
    
    def sample(self, n, tensor=True, device='cpu', half=False, seed=None):
        if not seed:
            seed = self.seed
        x = self.sampler.generate(dimensions=self.unit_sq, n_samples=n, random_state=seed)
        x = np.array(x)
        x = self.scale(x)
        x = np.split(x, self.dim, 1)

        if isinstance(self.func, list):
            sample = *x, *[f(*x) for f in self.func]
        else:
            sample = *x, self.func(*x)

        if tensor:
            return self._Array_to_Tensor(sample, device, half)
        else:
            return sample

### Join multiple samplers
class Join_samplers:
    def __init__(self, samplers, percentages=None):
        self.samplers = samplers
        if percentages is not None:
            assert len(percentages) == len(samplers)
            assert sum(percentages) == 1
        else:
            percentages = len(samplers)* [1/len(samplers)]
        self.percentages = percentages

    def sample(self, n, **kwargs):
        # q, r = divmod(n, len(self.samplers))
        l = [int(n* p) for p in self.percentages]
        l[0] += n - sum(l)

        samples = list(self.samplers[0].sample(l[0], **kwargs))
        for j, sampler in enumerate(self.samplers[1:]):
            sample = sampler.sample(l[j+1], **kwargs)
            for i, s in enumerate(sample):
                if torch.is_tensor(s):
                    samples[i] = torch.vstack([samples[i], s])
                else:
                    samples[i] = np.vstack([samples[i], s])
        return tuple(samples)

### Adapt a sample function
class Adaptor:
    def __init__(self, method, sample_fn, func, num_in=None, seed=1234, **adaptor_kwargs):
        sample_kwargs = copy.deepcopy(adaptor_kwargs)
        if method.lower() == "rad":
            self._k = sample_kwargs.pop("k")
            self._c = sample_kwargs.pop("c")
        elif method.lower() == "rar":
            pass
        else:
            raise NotImplementedError(f"method {method} is not implemented")
        self._kwargs = sample_kwargs

        self.method = method
        self.sample_fn = sample_fn
        self.func = func
        self.num_in = num_in


        self.state_dict = None
        self.seed = seed

    def _rar_fn(self, num_out):
        return _rar_fn(self.sample_fn, self.func, self.num_in, num_out, **self._kwargs)
    
    def _rad_fn(self, num_out):
        return _rad_fn(self.sample_fn, self.func, self.num_in, num_out, k=self._k, c=self._c, seed=self.seed, **self._kwargs)

    def sample(self, n):
        if self.method.lower() == "rar":
            return self._rar_fn(num_out=n)
        elif self.method.lower() == "rad":
            return self._rad_fn(num_out=n)

### Sample from data
class Data_sampler:
    def __init__(self, data, adapt=False, out_dims=[-1], k=2, c=1, device="cpu"):
        n, m = data.shape
        data = np.hsplit(data, m)
        output = np.sqrt(sum(np.square(data[i]) for i in out_dims)).flatten()
        if adapt:
            abs_value = np.abs(output)
            density = (abs_value ** k) / (abs_value ** k).mean() + c
            density /= density.sum()
        else:
            density = (1 / n) * np.ones(n)
        self.data = data
        self.density = torch.Tensor(density)
        self.device = device
        self.n = n

    @staticmethod
    def _Array_to_Tensor(list_A, device):
        list_T = []
        for a in list_A:
            t = torch.Tensor(a).to(device)
            list_T.append(t)
        return tuple(list_T)

    def sample(self, n, device=None, **kwargs):
        if device is None:
            device = self.device
        sample = self._Array_to_Tensor(self.data, device)
        n = min(n, self.n)
        idx = torch.multinomial(self.density, n)
        return tuple(s[idx] for s in sample)
