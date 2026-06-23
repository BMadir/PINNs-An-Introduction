import torch

class feedforward(torch.nn.Module):
    
    def __init__(self, layers_dim, activations, adaptive_activations=False, device='cuda', seed=1234):
        
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
        
        self.layers = self.create_layers()
        self.init_net(seed)
        
        if device == 'cpu':
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            
        self.to(self.device)
        
        self.activations_parameters = self.create_activations_parameters(adaptive_activations)
        
        
    def create_layers(self):
        layers = torch.nn.ModuleList()
        dim = self.layers_dim[0]
        for hdim in self.layers_dim[1 :]:
            layers.append(torch.nn.Linear(dim, hdim))
            dim = hdim
        return layers
    
    def init_net(self, seed):
        torch.manual_seed(seed)
        for p in self.parameters():
            try:
                torch.nn.init.xavier_uniform_(p)
            except:
                torch.nn.init.constant_(p, 0)
    
    def activation(self, name):
        try:
            f = getattr(torch, name)
        except:
            f = getattr(torch.nn.functional, name)
        return f
    
    def create_activations_parameters(self, adaptive_activations, a=0.1, n=10):
        
        if adaptive_activations:
            a = torch.tensor(a, device=self.device)
            return torch.nn.ParameterList(n* torch.nn.Parameter(a, requires_grad=True) for _ in range(len(self.activations)))
        else:
            return list(1. for _ in range(len(self.activations)))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[: -1]):
            a = self.activations_parameters[i]
            f = self.activation(self.activations[i])
            x = f(a* layer(x))
        return self.layers[-1](x)

    
    
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
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[: -1]):
            f = getattr(torch.nn.functional, self.activations[i])
            x = f(layer(x))
        return self.layers[-1](x)