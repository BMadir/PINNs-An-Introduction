"""
Sampling for PINNs

This module provides:
- Sampler: Latin hypercube / Sobol / Halton / Hammersley sampling in bounded domains
- Join_samplers: combination of multiple sampling strategies with weighted proportions
"""


import numpy as np
import torch
import skopt
import copy

from utils import  torch_out

__all__ = ["Sampler", "Join_samplers"]


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
