""" 
!!!! FINAL VERSION IN PROGRESS !!!!
Training framework for PINNs.

This module implements a flexible Trainer class handling:
- supervised / PINN training loops
- validation monitoring and best-model checkpointing
- adaptive loss weighting strategies
- custom optimization and scheduling

Key features:
- Automatic validation splitting and evaluation
- Multiple weighting strategies (annealing, custom algorithms, penalty methods)
- Support for custom optimizers and learning rate schedules
- Checkpoint saving / loading and restart capability
- Flexible closure-based optimization (compatible with LBFGS-style training)

Also includes utilities for:
- saving/loading training state
- computing weighted loss combinations
- handling error-based regularization (AL, SA, penalty methods)

"""




import torch
import numpy as np
import copy
from AutoDiff import AG_grad
from pinns_weights import*
import optimizers

__all__ = ["Trainer"]

class Trainer:
    def __init__(self, model, validation_data, save_path, *args, **kwargs):

        self._set_model(model)
         
        self.optimizer = None
        self.lr_scheduler = None
        
        self.dict_losses = {'training': [], 'validation': []}
        self.dict_weights = {'cst': [], 'vect': []}

        # treatment of validation data
        self.__in_dim = model.layers_dim[0]
        self.validation_data_is_available = False
        if validation_data is not None:
            n, m = validation_data.shape
            validation_data = torch.Tensor(validation_data).to(self.device)
            validation_data = torch.hsplit(validation_data, m)
            self.val_in, self.val_out = self.split_fn(validation_data)
            self.validation_data_is_available = True

        if save_path is None:
            save_path = './'
        self.save_path = save_path

    def _set_model(self, model):
        if model is not None:
            self.model = model
            self.device = model.device

            self.net = model.net
            self.net_grad = model.net_grad
            self.net_res = model.net_res

            self.l2_error = model.l2_error

            self._parameters = model.parameters
            self.__parameters = model.parameters
            self.__model_weights = model.weights
            self.__zero_grad = model.zero_grad

    def split_fn(self, samples):
        return samples[:self.__in_dim], samples[self.__in_dim:]
        
    def validation_error(self, epoch):
        error = self.l2_error(self.val_in, self.val_out)
        try:
            sum_ = sum(error)/len(error)
        except:
            sum_ = error

        if isinstance(error, list):
            error.append(sum_)
        self.dict_losses["validation"].append(error)

        check_list = self.dict_losses["validation"]
        if isinstance(check_list[0], list):
            check_list = torch.Tensor(self.dict_losses["validation"])[:, -1]

        if sum_ == min(check_list):
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                epoch=epoch,
                path=self.save_path,
                name='best_model.pth'
            )
        return  sum_

    @staticmethod
    def __mse_error(outputs, targets):
        error = (targets - outputs)
        loss = error.pow(2).mean()
        return loss, error
    
    def mse_error(self, outputs, targets=0.):
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            if isinstance(targets, float):
                targets = [targets]*len(outputs)
            loss, error = 0, 0
            for out, tar in zip(outputs, targets):
                l, e = self.__mse_error(out, tar)
                loss += l
                error += e
            return loss, error
        else:
            return self.__mse_error(outputs, targets)

    # Compute the loss weights
    def compute_weights(self, method, losses, weights, **kwargs):
        if method == "lr_annealing":
            return lr_annealing(
                losses=losses,
                #params=self.__model_weights(),
                params=self.__first_weight,
                weights=weights,
                **kwargs
            )
        elif method == "my_algo":
            return my_algo(
                losses=losses,
                params=self.__model_weights(),
                weights=weights,
                **kwargs
            )
        elif method == "my_algo2":
            return my_algo2(
                losses=losses,
                params=self.__first_weight,
                weights=weights,
                **kwargs
            )
        else:
            return weights
        
    # Compute the total loss
    def compute_loss(self, method, losses, errors, weights, **kwargs):
        if method in (None, "lr_annealing", "my_algo", "my_algo2"):
            return sum(w * l for w, l in zip(weights, losses))
        elif method == "penalty":
            return penalty(errors=errors, weights=self.error_weights, **kwargs)
        elif method == "AL":
            return augmented_lagrangian(errors=errors, weights=self.error_weights, **kwargs)
        elif method == "SA":
            return soft_attention(errors=errors, weights=self.error_weights, **kwargs)
        elif method == "rad":
            return rad_loss(errors=errors, weights=weights)
        else:
            raise RuntimeError(f"{method} is not supported")

    # Closure function
    def closure_fn(self, iteration, weighting_dict, func=None, **kwargs):
        frequency = weighting_dict.get("frequency")
        if frequency is None:
            frequency = 1
        self.__zero_grad()
        losses, errors = self.loss_fn()
        if iteration % frequency == 0:
            self._last_weights = self.compute_weights(losses=losses, weights=self._last_weights, **weighting_dict)
        total_loss = self.compute_loss(losses=losses, errors=errors, weights=self._last_weights, **weighting_dict, **kwargs)

        # backward
        if callable(func):
            total_loss = func(total_loss)
        total_loss.backward()

        # save last values of the losses:
        self._last_losses = [loss.item() for loss in losses]
        self._last_losses.append(total_loss.item())
        return total_loss

    def _set_weights(self, weighting_dict):
        """ helper function initializes the weights """
        if weighting_dict is None:
            weighting_dict = dict(method=None)
        weighting_dict_ = copy.deepcopy(weighting_dict)

        # Loss weights
        try:
            self._last_weights = weighting_dict_.pop("weights")
        except:
            self._set_res_data(batch_size=1)
            self.res_in = self._check_device(next(iter(self.res_dataloader)))
            self._last_weights = [1.]* len(self.loss_fn()[0])

        # Error weights
        method = weighting_dict.get("method")
        if method in ("penalty", "AL", "SA"):
            params_data = weighting_dict.get("params_data")
            if self.error_weights is None:
                self.error_weights = _new_params(params_data, device=self.device)
        else:
            self.error_weights = None
        return weighting_dict_

    def _set_opt(self, weighting_dict, optim_dict):

        if weighting_dict is None:
            weighting_dict = dict(method=None)
        if optim_dict is None:
            optim_dict = dict()
        optim_dict_ = copy.deepcopy(optim_dict)

        # Parameters with/without the error weights
        method = weighting_dict.get("method")
        if method in  ("penalty", "AL", "SA"):
            lr_max = weighting_dict.get('lr_max')
            if lr_max is None:
                optim_dict['lr_max'] = 1e-2
            else:
                optim_dict['lr_max'] = lr_max

            if self.error_weights is not None:
                params_list = [
                    {'params': self.__parameters()},
                    {'params': self.error_weights, 'maximize': True, 'lr': optim_dict['lr_max']}
                ]
        else:
            params_list = self.__parameters()

        # Optimizer & lr_scheduler
        try:
            optim_dict['optimizer'] = optim_dict_.pop('optimizer')
        except:
            optim_dict['optimizer'] = "Adam"
            optim_dict['lr'] = 1e-3
        try:
            exp_lr = optim_dict_.pop('exp_lr')
        except:
            exp_lr = False
        try:
            exp_dr = optim_dict_.pop('exp_dr')
        except:
            exp_dr = .9
        try:
            exp_ds = optim_dict_.pop('exp_ds')
        except:
            exp_ds = 8000

        # set the optimizer & zero_grad_fn
        if hasattr(torch.optim, optim_dict['optimizer']):
            self.optimizer = getattr(torch.optim, optim_dict['optimizer'])(params_list, **optim_dict_)
        elif hasattr(optimizers, optim_dict['optimizer']):
            self.optimizer = getattr(optimizers, optim_dict['optimizer'])(params_list, **optim_dict_)
        else:
            raise Exception('problem in the optimizer')
        
        self.__zero_grad = self.optimizer.zero_grad
        if exp_lr:
            optim_dict['exp_lr'] = (exp_dr, exp_ds)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda t: exp_dr ** (t / exp_ds))
        return optim_dict

    # Define the step_fn
    def step(self, closure):
        if self.optimizer is not None:
            loss = self.optimizer.step(closure)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return loss
    
    @torch.no_grad()
    def weights_norm(self, *args):
        _norm = list()
        for w in args:
            _norm.append(torch.linalg.norm(w).item())
        return  _norm
    
    # SAVE
    def save_losses_weights(self, losses=True, weights=False, file='', name=''):
        if losses:
            for key, val in self.dict_losses.items():
                np.savetxt(file + name + 'losses_' + key, val)
        if weights:
            for key, val in self.dict_weights.items():
                np.savetxt(file + name + 'weights_' + key, val)

    def save_all(self, epoch):
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            epoch=epoch,
            path=self.save_path,
            name='model.pth'
        )
        self.save_losses_weights(
            losses=True,
            weights=True,
            file=self.save_path,
        )
        print("save at epoch %i"%(epoch+1))

    def restart(
            self,
            weighting_dict, 
            optim_dict,
            restart_opt, 
            **kwargs
        ):
        training = np.loadtxt(fname=self.save_path + 'losses_training')
        validation = np.loadtxt(fname=self.save_path + 'losses_validation')
        weights = np.loadtxt(fname=self.save_path + 'weights_cst')
        self.dict_losses['training'] = [list(loss) for loss in list(training)]
        self.dict_losses['validation'] = [list(loss) for loss in list(validation)]
        self.dict_weights['cst'] = [list(weight) for weight in list(weights)]
        self._last_losses = self.dict_losses['training'][-1]
        self._last_weights = self.dict_weights['cst'][-1]

        self._set_opt(weighting_dict, optim_dict)
        self.model, self.optimizer, self.lr_scheduler = load_checkpoint(
            path=self.save_path,
            name='model.pth',
            model=self.model,
            device=self.device,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler
        )
        self.train(
            restart_opt=restart_opt,
            weighting_dict=weighting_dict,
            optim_dict=optim_dict,
            **kwargs
        )


def save_checkpoint(model, optimizer=None, lr_scheduler=None, epoch=None, path='', name='checkpoint.pth'):
    checkpoint = dict(model=model.state_dict())
    try:
        checkpoint['layers_dim'] = model.layers_dim
        checkpoint['activations'] = model.activations
    except:
        pass
    try:
        checkpoint['mean_std'] = model.mean_std
    except:
        pass
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
    torch.save(checkpoint, path + name)

def load_checkpoint(model, device, optimizer=None, lr_scheduler=None, path='', name='checkpoint.pth'):
    checkpoint = torch.load(f=path+name, weights_only=True)
    model.load_state_dict(checkpoint['model'], device)
    try:
        model.mean_std = checkpoint['mean_std']
    except:
        pass
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    return model, optimizer, lr_scheduler
