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


"""
def _rad_fn(sample_fn, func, num_in, num_out, k, c, in_dim=None, seed=1234, **kwargs):

    r = num_out/num_in
    coef = 1/(r* np.log(2* k + np.log(1 + r)* c))
    
    ''' Residual-based adaptive distribution '''
    # generator = torch.Generator("cuda").manual_seed(seed)
    try:
        generator = kwargs.pop("generator")
    except:
        generator = None
    sample = sample_fn(num_in, **kwargs)
    if in_dim is None:
        in_dim = len(sample) - 1
    inputs, outputs = sample[:in_dim], sample[in_dim:]
    abs_value = func(*inputs).abs().flatten()
    with torch.no_grad():
        density = (abs_value ** k) / (abs_value ** k).mean() + c
        density /= density.sum()
        indices = torch.multinomial(density, num_out, generator=generator)
        indices = indices.flatten()
        #return tuple(s[indices] for s in sample)
        return *[s[indices] for s in sample], coef* density[indices].reshape(-1, 1)
"""

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

# Adapt a sample function
class Adaptor:
    """ Adapt a given sampler to acordin to a certain function.

       Args:
           method: RAR or RAD.
           sample_fn:
           func:
           num_in=None:
        """
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

    """ To review
    def _update(self, sample, set_2=False, set_1_size=None, save=False):
        if self.method.lower() == "rar":
            adapted = self.rar_fn()
        elif self.method.lower() == "rad":
            adapted = self.rad_fn()
        else:
            raise Exception("unsupported adaptor")
        if save:
            self._save(adapted)
        if set_2 and set_1_size:
            return [torch.vstack([s[0:set_1_size, :], adapted[i]]) for i, s in enumerate(sample)]
        else:
            return adapted

    def _save(self, sample):
        if isinstance(sample, list):
            pass
        else:
            sample = list(sample)

        if self.state_dict is None:
            keys = ["sample_" + str(i) for i in range(len(sample))]
            values = [[] for i in range(len(sample))]
            self.state_dict = dict(zip(keys, values))

        num = len(sample)
        for i in range(num):
            self.state_dict["sample_" + str(i)].append(sample[i].detach().cpu())
    """

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

""" To review
class SeqToSeq(Sampler):
    def __init__(self, n_seq, n_col, n_iter, l_bounds, u_bounds, func=None, sampler='lhs', **kwargs):
        super().__init__(l_bounds, u_bounds, func, sampler, **kwargs)
        self.n_seq = n_seq
        self.n_col = n_col
        self.seq_bounds = self._seq_bounds(n_seq, self.bounds, 0)
        self.seq_sizes = self._seq_sizes(n_seq, n_col)

        self.n_iter = n_iter
        iterations = [n_iter for _ in self.seq_sizes]
        self.iterations = [sum(iterations[:i]) for i in range(len(iterations))]

        self.samples = self._samples(self.seq_sizes)

    def _seq_bounds(self, n_seq, bounds, indice):
        seq = lambda k: bounds[0][indice] + k * (bounds[1][indice] - bounds[0][indice]) / n_seq
        seq_bounds = []
        for k in range(1, n_seq + 1):
            _bounds = bounds[1][:]
            _bounds[indice] = seq(k)
            _bounds = [bounds[0], _bounds]
            seq_bounds.append(_bounds)
        return seq_bounds

    def _seq_sizes(self, n_seq, size):
        size_0 = int(0.1 * size)
        size_k = int((size - size_0) / (n_seq - 1)) + 1
        tot = size_0 + (n_seq - 1) * size_k
        size_0 = size_0 - (tot - size)
        return [size_0 + k * size_k for k in range(n_seq)]

    def __scale(self, sample, bounds):
        dim = len(bounds[0])
        lower = np.broadcast_to(bounds[0], dim)
        upper = np.broadcast_to(bounds[1], dim)
        return sample * (upper - lower) + lower

    def __sample(self, n, bounds, tensor=True, device='cpu'):
        x = self.sampler.generate(dimensions=self.unit_sq, n_samples=n, random_state=self.seed)
        x = np.array(x)
        x = self.__scale(x, bounds)
        x = np.split(x, self.dim, 1)

        if isinstance(self.func, list):
            sample = *x, *[f(*x) for f in self.func]
        else:
            sample = *x, self.func(*x)

        if tensor:
            return self._Array_to_Tensor(sample, device)
        else:
            return sample

    def _samples(self, seq_sizes):
        samples = []
        for (size, bounds) in zip(seq_sizes, self.seq_bounds):
            sample = self.__sample(size, bounds, device="cuda")
            samples.append(sample)

        self.samples = samples
        return samples
"""
