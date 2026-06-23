import torch

__all__ = ["feedforward", "xfeedforward", "ff", "st_ff", "mfeedforward"]

# Feed Forward
class feedforward(torch.nn.Module):
    def __init__(
            self,
            layers_dim,
            activations,
            adapt_act=False,
            device='cuda',
            seed=1234
    ):
        super().__init__()
        if len(layers_dim) < 3:
            raise Exception('len(layers_dim) < 3')
        else:
            self.layers_dim = layers_dim
            
        if isinstance(activations, list) and len(activations) == len(layers_dim) - 2:
            self.activations = activations
        elif isinstance(activations, str):
            self.activations = (len(self.layers_dim) - 2)* [activations]
        else:
            raise Exception('problem in activations')

        if device == 'cpu':
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        
        self.layers = self._layers(layers_dim)
        self._act_fns = [self._act_fn(n) for n in self.activations]
        self._act_params = self.__act_params(adapt_act)
        self._initialize(seed)
        self.seed = seed
        self.to(self.device)

    def weights(self):
        for n, p in self.named_parameters():
            if "weight" in n:
                yield p

    def bias(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                yield p

    def _numel(self):
        self.numel = sum(
            2 * p.numel() if torch.is_complex(p) else p.numel() for p in self.parameters()
        )
        return self.numel

    def _layers(self, layers_dim):
        layers = torch.nn.ModuleList()
        dim = layers_dim[0]
        for hdim in layers_dim[1 :]:
            layers.append(torch.nn.Linear(dim, hdim))
            dim = hdim
        return layers

    def _initialize(self, seed):
        torch.manual_seed(seed)
        for l in self.layers:
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.constant_(l.bias, 0)

    def _act_fn(self, name):

        if name == "sf":
            return  lambda x: torch.sin(2* torch.pi* x)
        if hasattr(torch.nn, name):
            return getattr(torch.nn, name)()
        if hasattr(torch, name):
            return getattr(torch, name)
        elif hasattr(torch.nn.functional, name):
            return getattr(torch.nn.functional, name)
        else:
            raise Exception('problem in activations')
    
    def __act_params(self, adapt_act, a=0.1, n=10):
        len_act = len(self.activations)
        if adapt_act:
            a = torch.tensor(a, device=self.device)
            return torch.nn.ParameterList(n* torch.nn.Parameter(a, requires_grad=True) for _ in range(len_act))
        else:
            return list(1. for _ in range(len_act))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[: -1]):
            a = self._act_params[i]
            f = self._act_fns[i]
            x = f(a* layer(x))
        return self.layers[-1](x)

    def save(self, name='model', file=''):
        save_dict = {
            'layers_dim': self.layers_dim,
            'activations': self.activations,
            'state_dict': self.state_dict()
        }
        torch.save(save_dict, file + name + '.pth')

# Extended
from itertools import chain

class xfeedforward(torch.nn.Module):
    def __init__(
            self,
            layers_dim,
            activations,
            device='cuda',
            seed=1234
    ):
        super().__init__()
        models = [feedforward(layers, fn, device=device, seed=seed) for layers, fn in zip(layers_dim, activations)]
        self.models = torch.nn.ModuleList(models)
        
        self.device = torch.device(device)
        self.layers_dim = layers_dim[0]
        self.layers_dim[-1] = sum(layers[-1] for layers in layers_dim)
        self.activations = activations
        self.seed = seed
	
    def weights(self):
        params = [model.weights() for model in self.models]
        return chain(*params)
        
    def freeze(self, i):
        for p in self.models[i].parameters():
            p.requires_grad = False

    def unfreeze(self, i):
        for p in self.models[i].parameters():
            p.requires_grad = True
            
    def forward(self, x):
        return torch.cat([model(x) for model in self.models], dim=1)
        

# Fourier features
class ff(feedforward):
    def __init__(self, layers_dim, activations, trainable_ff=False, mean=0., std=1., **kwargs):
        super().__init__(layers_dim=layers_dim, activations=activations, **kwargs)
        self.mean = mean
        self.std = std

        assert layers_dim[1] % 2 == 0, "layers_dim[1] % 2 != 0"
        in_features = layers_dim[0]
        out_features = layers_dim[1]

        torch.manual_seed(self.seed)
        ff = torch.nn.Linear(in_features, out_features // 2, bias=False, device=self.device)
        torch.nn.init.normal_(ff.weight, mean=mean, std=std)

        if not trainable_ff:
            ff.weight.requires_grad_(False)

        self.layers[0] = ff
        self.activations.insert(0, "ff")
        self._act_fns.insert(0, None)

    @staticmethod
    def ff_act(x):
        return torch.cat([torch.cos(2 * torch.pi * x), torch.sin(2 * torch.pi * x)], 1)

    def forward(self, inputs):
        x = self.ff_act(self.layers[0](inputs))
        for i, layer in enumerate(self.layers[1:-1]):
            a = self._act_params[i + 1]
            f = self._act_fns[i + 1]
            x = f(a * layer(x))
        return self.layers[-1](x)

# (space-time) Fourier features
class st_ff(feedforward):
    def __init__(self, layers_dim, activations, trainable_ff=False, mean=[0., 0.], std=[1., 1.], **kwargs):
        super().__init__(layers_dim=layers_dim, activations=activations, **kwargs)
        self.mean = mean
        self.std = std

        assert layers_dim[1] % 2 == 0, "layers_dim[1] % 2 != 0"
        in_features = layers_dim[0]
        out_features = layers_dim[1]

        torch.manual_seed(self.seed)
        fft = torch.nn.Linear(1, out_features // 2, bias=False, device=self.device)
        ffx = torch.nn.Linear(in_features - 1, out_features // 2, bias=False, device=self.device)
        torch.nn.init.normal_(fft.weight, mean=mean[0], std=std[0])
        torch.nn.init.normal_(ffx.weight, mean=mean[1], std=std[1])

        if not trainable_ff:
            fft.weight.requires_grad_(False)
            ffx.weight.requires_grad_(False)

        self.layers[0] = torch.nn.ModuleList([fft, ffx])
        self.activations.insert(0, "st_ff")
        self._act_fns.insert(0, None)

    @staticmethod
    def ff_act(x):
        return torch.cat([torch.cos(2 * torch.pi * x), torch.sin(2 * torch.pi * x)], 1)

    def forward(self, inputs):
        t, x = inputs[:, 0:1], inputs[:, 1:]
        t = self.ff_act(self.layers[0][0](t))
        x = self.ff_act(self.layers[0][1](x))
        for i, layer in enumerate(self.layers[1:-1]):
            a = self._act_params[i + 1]
            f = self._act_fns[i + 1]
            t = f(a * layer(t))
            x = f(a * layer(x))
        return self.layers[-1](t * x)

class mfeedforward(feedforward):
    def __init__(self, layers_dim, activations, **kwargs):
        super().__init__(layers_dim=layers_dim, activations=activations, **kwargs)

        hidden_dims = layers_dim[1:-1]
        assert hidden_dims.count(hidden_dims[0]) == len(hidden_dims), "hidden dims should be equal"

        torch.manual_seed(self.seed)
        e1 = torch.nn.Linear(layers_dim[0], layers_dim[1], device=self.device)
        e2 = torch.nn.Linear(layers_dim[0], layers_dim[1], device=self.device)

        torch.nn.init.xavier_normal_(e1.weight)
        torch.nn.init.constant_(e1.bias, 0.)
        torch.nn.init.xavier_normal_(e2.weight)
        torch.nn.init.constant_(e2.bias, 0.)

        self.encoders = torch.nn.ModuleList([e1, e2])

    @staticmethod
    def mff_act(x):
        return torch.tanh(x)

    def forward(self, x):
        U = self.mff_act(self.encoders[0](x))
        V = self.mff_act(self.encoders[1](x))
        for i, layer in enumerate(self.layers[:-1]):
            a = self._act_params[i]
            f = self._act_fns[i]
            x = f(a * layer(x))
            x = (1 - x)* U + x * V
        return self.layers[-1](x)


######################
# To review
######################

# Residual network
class ResNetBlock(torch.nn.Module):
    
    def __init__(self, input_size, output_size, nonlinearity='tanh', id_parameter=True, device='cpu'):
        super().__init__()
        
        self.nonlinearity = getattr(torch.nn.functional, nonlinearity)
        self.linear_1 = torch.nn.Linear(input_size, output_size)
        self.linear_2 = torch.nn.Linear(output_size, output_size)
        
        if id_parameter:
            self.id_parameter = torch.nn.Parameter(torch.randn((1, output_size), device=torch.device(device)), requires_grad=True)
        else:
            self.id_parameter = torch.ones((1, output_size), device=torch.device(device))
            
    def pad(self, x, y):
        "pad x with zeros (unpad is also possible but not recommended), to match y size"
        
        n = y.shape[1] - x.shape[1]
        pad = (int(n/2), n - int(n/2))
        
        return torch.nn.functional.pad(x, pad)

    def forward(self, x):
        
        y = self.linear_2(self.nonlinearity(self.linear_1(x)))
        
        return y + self.id_parameter *self.pad(x, y)
    

class ResNet(torch.nn.Module):
    def __init__(self, layers_dim, activations, device='cuda', seed=1234):
        super().__init__()
        
        if device == 'cpu':
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            
        if len(layers_dim) < 3:
            raise Exception('len(layers_dim) < 3')
        else:
            self.layers_dim = layers_dim
            
        if isinstance(activations, list) and len(activations) == len(layers_dim) - 2:
            self.activations = activations
        elif isinstance(activations, str):
            self.activations = (len(self.layers_dim) - 2)* [activations]
        else:
            raise Exception('problem in activations')
        
        self.layers = self.create_layers()
        self.init_net(seed)
        
        self.to(self.device)
        
    def create_layers(self):
        layers = torch.nn.ModuleList()
        dim = self.layers_dim[0]
        for hdim in self.layers_dim[1 :]:
            layers.append(ResNetBlock(dim, hdim, device=self.device))
            dim = hdim
        return layers
    
    def init_net(self, seed):
        torch.manual_seed(seed)
        for p in self.parameters():
            try:
                torch.nn.init.xavier_uniform_(p)
            except:
                torch.nn.init.constant_(p, 0)

    def weights(self):
        for n, p in self.named_parameters():
            if "weight" in n:
                yield p

    def bias(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                yield p

    def _numel(self):
        self.numel = sum(
            2 * p.numel() if torch.is_complex(p) else p.numel() for p in self.parameters()
        )
        return self.numel
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[: -1]):
            f = getattr(torch.nn.functional, self.activations[i])
            x = f(layer(x))
        return self.layers[-1](x)
