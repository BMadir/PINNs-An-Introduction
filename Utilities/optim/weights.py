import torch
from torch.func import jacrev, vjp, jvp, vmap

@torch.no_grad
def __grad_norm(loss_fn, params):
    
    grads = jacrev(loss_fn)(params)
    grads_norm = [torch.linalg.norm(grad) for grad in grads]
    mean_grads_norm = torch.tensor(grads_norm).mean()
    
    weights = map(
        lambda x: mean_grads_norm / x,
        grads_norm
    )
    return weights

@torch.no_grad
def __lr_annealing(loss_fn, params, weights_old, res_index=[-1], a=0.6, sup=1e4):
    
    grads = jacrev(loss_fn)(params)
    grads_mean = [grad.abs().mean().item() for grad in grads]
    res_grads = [grads[i] for i in res_index]
    res_weights = [weights_old[i] for i in res_index]
    max_res = sum(w* g for w, g in zip(res_weights, res_grads)).abs().max().item()
    
    weights = map(  
        lambda w, m: min(sup, (1 - a)* w + (a/w)* (max_res / m)), 
        weights_old,
        grads_mean
    )
    return weights

def ntk(func, params, x1, x2, compute='full'):
    
    def get_ntk(x1, x2):
        func_x1 = lambda params: func(params, x1)
        func_x2 = lambda params: func(params, x2)
        
        output, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            vjps = vjp_fn(vec)
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps
            
        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
        return vmap(get_ntk_slice)(basis)
        
    result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)
    if compute == 'full':
        return result
    if compute == 'trace':
        return torch.einsum('NMKK->NM', result)
    if compute == 'diagonal':
        return torch.einsum('NMKK->NMK', result)

