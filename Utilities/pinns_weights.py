import torch
from AutoDiff import AG_grad

__all__ = ["lr_annealing", "_new_params", "penalty", "augmented_lagrangian", "soft_attention", "rad_loss", "my_algo", "my_algo2"]

# Learning rate annealing algorithme:
def _grad_losses(losses, params, func=lambda x: x):
    params = tuple(p for p in params if p.requires_grad)
    grads = list()
    for loss in losses:
        grad = AG_grad(loss, params)
        grad = torch.cat([g.detach().view(-1) for g in grad])
        grads.append(func(grad))
    return grads


def lr_annealing(
        losses,
        params,
        weights=None,
        res_indices=None,
        alpha=0.6,
        sup=1e5,
        **kwargs
):
    if weights is None:
        weights = [1. for _ in losses]
    if res_indices is None:
        res_indices = [len(losses) - 1]
    grads = _grad_losses(losses, params)
    res_grads = [weights[i] * grads[i] for i in res_indices]
    max_res = sum(res_grads).abs().max().item()
    new_weights = weights[:]
    eps = 0
    for i, (w, grad) in enumerate(zip(weights, grads)):
        if i in res_indices:
            pass
        else:
            m = grad.abs().mean().item()
            w = (1. - alpha) * w + alpha * (max_res / (w * m + eps))
            new_weights[i] = min(sup, w)
    return new_weights


"""
def lr_annealing(
        losses,
        params,
        weights=None,
        res_indices=None,
        alpha=0.6,
        sup=1e5,
        **kwargs
):
    if weights is None:
        weights = [1. for _ in losses]
    if res_indices is None:
        res_indices = [len(losses) - 1]
    grads = _grad_losses(losses, params)
    res_grads = [weights[i] * grads[i] for i in res_indices]
    max_res = sum(res_grads).abs().max().item()
    new_weights = weights[:]
    eps = 0
    for i, (w, grad) in enumerate(zip(weights, grads)):
        if i in res_indices:
            pass
        else:
            m = grad.abs().mean().item()
            w = (1. - alpha) * w + alpha * (max_res / (w * m + eps))
            new_weights[i] = min(sup, w)
    return new_weights


def lr_annealing(
        losses,
        params,
        weights=None,
        alpha=0.6,
        **kwargs
):
    new_weights = weights[:]
    grads = _grad_losses(losses, params, func=lambda x: x.abs())
    idx, Gmax = max(enumerate(grads), key=lambda g: g[1].max())
    Gmax = Gmax.max()

    for i, G in enumerate(grads):
        if i != idx:
            w = (1 - alpha) * new_weights[i] + (alpha / new_weights[i]) * (Gmax / G.mean())
            new_weights[i] = w.item()
    return  new_weights


def lr_annealing(
        losses,
        params,
        weights=None,
        res_indices=None,
        alpha=0.6,
        **kwargs
):
    res_indices = set(res_indices) if res_indices is not None else set()
    grads = _grad_losses(losses, params, func=lambda x: x.abs())
    idx, Gmax_tensor = max(enumerate(grads), key=lambda g: g[1].norm())
    Gmax = float(Gmax_tensor.max())
    new_weights = list(weights)
    skip = res_indices | {idx}
    for i, G in (
        (i, G) for i, G in enumerate(grads) if i not in skip
    ):
        Gi = float(G.mean())
        wi = new_weights[i]
        new_weights[i] = (1 - alpha) * wi + (alpha / wi) * (Gmax / Gi)
    return new_weights

def my_algo(
        losses,
        params,
        weights,
        res_indices=None,
        alpha=0.99,
        sup=1e4,
        **kwargs
):
    if res_indices is None:
        res_indices = [len(losses) - 1]
    new_weights = weights[:]
    norms = _grad_losses(losses, params, func=lambda x: x.norm())
    norm_res = max([norms[i] for i in res_indices])
    weights_ = [norm_res / norm for norm in norms]

    loss_res = sum(weights_[i] * losses[i] for i in res_indices)
    norm_res = _grad_losses([loss_res], params, func=lambda x: x.norm())[0]

    for i, w in enumerate(weights):
        w = (1. - alpha) * w + alpha * (norm_res / (w * norms[i]))
        new_weights[i] = min(sup, w.item())
    return new_weights

def my_algo(
        losses,
        params,
        weights,
        res_indices=None,
        alpha=0.99,
        **kwargs
):
    if res_indices is None:
        res_indices = [len(losses) - 1]
    new_weights = weights[:]
    norms = _grad_losses(losses, params, func=lambda x: x.norm())
    max_res = max([norms[i] for i in res_indices])
    min_res = min([norms[i] for i in res_indices])

    for i, w in enumerate(weights):
        if i in res_indices:
            coef = min_res / norms[i]
        else:
            coef = max_res / norms[i]
        w = (1. - alpha) * w + alpha * (coef / w)
        new_weights[i] = w.item()
    return new_weights


def lr_annealing(
        losses,
        params,
        weights=None,
        res_indices=None,
        alpha=0.6,
        sup=1e5,
        **kwargs
):
    if weights is None:
        weights = [1.0] * len(losses)

    if res_indices is None:
        res_indices = {len(losses) - 1}
    else:
        res_indices = set(res_indices)

    grads = _grad_losses(losses, params)

    # Compute max residual gradient magnitude
    max_res = sum(weights[i] * grads[i] for i in res_indices)
    max_res = max_res.abs().max()

    new_weights = list(weights)
    eps = 1e-12

    for i, (w, g) in enumerate(zip(weights, grads)):
        if i in res_indices:
            continue

        m = g.abs().mean()
        upd = (1 - alpha) * w + alpha * (max_res / (w * m + eps))
        new_weights[i] = min(sup, float(upd))

    return new_weights
"""

def my_algo(
        losses,
        params,
        weights=None,
        res_indices=None,
        alpha=0.6,
        **kwargs
):
    res_indices = set(res_indices) if res_indices is not None else set()
    grads = _grad_losses(losses, params, func=lambda x: x.abs())
    idx, Gmax = max(enumerate(grads), key=lambda g: g[1].norm())
    Gmax = float(Gmax.norm())
    new_weights = list(weights)
    skip = res_indices | {idx}
    for i, G in (
        (i, G) for i, G in enumerate(grads) if i not in skip
    ):
        Gi = float(G.norm())
        #wi = new_weights[i]
        #new_weights[i] = (1 - alpha) * wi + (alpha / wi) * (Gmax / Gi)
        new_weights[i] = Gmax / Gi
    return new_weights

def my_algo2(
        losses,
        params,
        weights,
        res_indices=None,
        alpha=0.7,
        **kwargs
):
    if res_indices is None:
        res_indices = {len(losses) - 1}
    else:
        res_indices = set(res_indices)

    norms = _grad_losses(losses, params, func=lambda x: x.norm())
    res_norms = [norms[i] for i in res_indices]
    max_res = max(res_norms)
    min_res = min(res_norms)

    new_weights = []
    for i, w in enumerate(weights):
        norm_i = norms[i]
        coef = (min_res if i in res_indices else max_res) / norm_i
        updated = (1 - alpha) * w + alpha * (coef / w)
        new_weights.append(float(updated))
    return new_weights












































# define the Lagrange multiplier (weights adjusted using GD):
def _new_params(data, device):
    params = torch.nn.ParameterList()
    for d in data:
        if torch.is_tensor(d):
            p = torch.nn.Parameter(d.to(device))
        else:
            p = torch.tensor([1.]).to(device).requires_grad_(False)
        params.append(p)
    return params

# Penalty method:
def penalty(
        errors,
        weights,
        indices=None,
        copy=None,
        **kwargs
):
    if indices is None:
        indices = [len(errors) - 1]
    loss = sum(errors[i].pow(2).mean() for i in indices)
    errors = [errors[i] for i in set(range(len(errors))) - set(indices)]
    if copy is not None:
        _weights = []
        for i, p in enumerate(LM):
            for _ in range(copy[i]):
                _weights.append(p)
    else:
        _weights = weights

    for i, e in enumerate(errors):
        c = _weights[i] * e.pow(2)
        loss += c.mean()
    return loss

# Augmented Lagrangian method:
def augmented_lagrangian(
        errors,
        weights,
        indices=None,
        beta=1.,
        copy=None,
        **kwargs
):
    if indices is None:
        indices = [len(errors) - 1]
    loss = sum(errors[i].pow(2).mean() for i in indices)
    errors = [errors[i] for i in set(range(len(errors))) - set(indices)]
    if copy is not None:
        _weights = []
        for i, p in enumerate(weights):
            for _ in range(copy[i]):
                _weights.append(p)
    else:
        _weights = weights

    for i, e in enumerate(errors):
        c = beta * e.pow(2) + _weights[i] * e
        loss += c.mean()
    return loss

# Soft Attention mechanism:
def soft_attention(
        errors,
        weights,
        mask_fns,
        copy=None,
        no_grad=None,
        **kwargs
):

    if copy is not None:
        _weights = []
        for i, p in enumerate(weights):
            for _ in range(copy[i]):
                _weights.append(p)
    else:
        _weights = weights
    if callable(mask_fns):
        fn = len(_weights) *  [mask_fns]
    else:
        fn = mask_fns

    if no_grad is not None:
        for i in no_grad:
            weights[i].requires_grad_(False)
            fn[i] = lambda x:x
    loss = 0.
    for i, (w, e) in enumerate(zip(_weights, errors)):
        c = fn[i](w) * e.pow(2)
        loss += c.mean()
    return loss

# NTK weighting: Soon!
# EW_functions:

def rad_loss(errors, weights):
    loss = 0
    for i, error in enumerate(errors):
        loss += (weights[i] * error.pow(2)).mean()
    return loss
