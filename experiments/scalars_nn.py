import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
import numpy as np
import itertools


def comp_inner_products(x, stype, simplified=True):
    """
    INPUT:
    N: number of datasets
    n: number of particles
    dim: dimension of each particle 
    x: torch tensor of size [N, n, dim]
    stype: "Euclidean" or "Minkowski"
   
    """
    _, n, d = x.shape
    if stype ==  "Euclidean":
        scalars = torch.einsum('bix,bjx->bij', x, x)
    elif stype == "Minkowski":
        G = torch.diag(-torch.ones(d))
        G[0,0] = 1
        G = torch.einsum('bix,bxj->bij', x, G.unsqueeze(0))
        scalars = torch.einsum('bij,bkj->bik', G, x)
    if simplified:
        scalars = torch.triu(scalars).view(-1, n**2)
        scalars = scalars[:, torch.nonzero(scalars[0]).squeeze(-1)]
    return scalars 

def comp_outer_products(x):
    N = x.shape[0]
    scalars = torch.einsum('bik,bjl->bijkl', x, x) #[N, n, n, dim, dim]
    return scalars.view(N,-1)


def dataset_transform(data):
    """
    data: numpy dataset of two attributes: data.X, data.Y
    """
    X = torch.from_numpy(data.X)
    Y = torch.from_numpy(data.Y)
    
    if data.symname == "O5invariant":
        X = X.reshape(-1,2,5)
        scalars = comp_inner_products(X, stype="Euclidean")
        
    elif data.symname == "O3equivariant":
        n=5 # five data points
        mi = X[:,:n]
        ri = X[:,n:].reshape(-1,n,3)
        x_outer = torch.einsum('bik,bjl->bijkl', ri, ri) #[N, n, n, dim, dim]
        x_inner = torch.einsum('bik,bjk->bij', ri, ri) #[N, n, n]
        index = np.array(list(itertools.combinations_with_replacement(np.arange(0,n), r=2)))

        N = len(X)
        X = X.new_zeros(N, 16, 3, 3)
        X[:,:15,:,:] = x_outer[:,index[:,0],index[:,1],:,:]
        X[:,-1,:,:] = torch.eye(3).repeat(N,1,1)
        
        
        ri = x_inner[:,index[:,0], index[:,1]] # [N, 15]
        mi1 = mi[:,index[:,0]]
        mi2 = mi[:,index[:,1]]
        
        scalars = torch.stack((mi1,mi2,ri),dim=-1) # [N, 15, 3]
    elif data.symname == "Lorentz":
        X = X.reshape(-1,4,4)
        scalars = comp_inner_products(X, stype="Minkowski")
    else:
        raise ValueError("Wrong symname???")
    dim_scalars = scalars.shape[-1]
    
    return {'dataset': TensorDataset(scalars, X, Y), 'dim_scalars': dim_scalars}

class BasicMLP(nn.Module):
    def __init__(
        self, 
        n_in, 
        n_out,
        n_hidden=100, 
        n_layers=2,
        layer_norm=True
    ):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), nn.ReLU()]
        for _ in range(n_layers):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_hidden, n_out))
        if layer_norm:
            layers.append(nn.LayerNorm(n_out))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)




class EquivariancePermutationLayer(nn.Module):
    def __init__(
        self, 
        n_in, 
        n_hidden, 
        n_layers, 
        layer_norm, 
    ):
        super(self.__class__, self).__init__()
       
        self.f_Mij = BasicMLP(
            n_in=4, 
            n_out=1, 
            n_hidden=n_hidden, 
            n_layers=n_layers,
            layer_norm=layer_norm,
        )
        self.f_Mii = BasicMLP(
            n_in=4, 
            n_out=1, 
            n_hidden=n_hidden, 
            n_layers=n_layers,
            layer_norm=layer_norm,
        )
        self.f_Mrij = BasicMLP(
            n_in=15, 
            n_out=1, 
            n_hidden=n_hidden, 
            n_layers=n_layers,
            layer_norm=layer_norm,
        )
        self.f_Mrii = BasicMLP(
            n_in=15, 
            n_out=1, 
            n_hidden=n_hidden, 
            n_layers=n_layers,
            layer_norm=layer_norm,
        )
        self.f_Mmij = BasicMLP(
            n_in=5, 
            n_out=1, 
            n_hidden=n_hidden, 
            n_layers=n_layers,
            layer_norm=layer_norm,
        )

        self.f_Mmii = BasicMLP(
            n_in=5, 
            n_out=1, 
            n_hidden=n_hidden, 
            n_layers=n_layers,
            layer_norm=layer_norm,
        )

        self.f_Mmimj = BasicMLP(
            n_in=1, 
            n_out=1, 
            n_hidden=n_hidden, 
            n_layers=n_layers,
            layer_norm=layer_norm,
        )

        self.f_Mmimi = BasicMLP(
            n_in=1, 
            n_out=1, 
            n_hidden=n_hidden, 
            n_layers=n_layers,
            layer_norm=layer_norm,
        )

        self.f_Iij = BasicMLP(
            n_in=2, 
            n_out=1, 
            n_hidden=n_hidden, 
            n_layers=n_layers,
            layer_norm=layer_norm,
        )
        self.f_Iii = BasicMLP(
            n_in=2, 
            n_out=1, 
            n_hidden=n_hidden, 
            n_layers=n_layers,
            layer_norm=layer_norm,
        )

        self.f_Imij = BasicMLP(
            n_in=1, 
            n_out=1, 
            n_hidden=n_hidden, 
            n_layers=n_layers,
            layer_norm=layer_norm,
        )

        self.f_Imii = BasicMLP(
            n_in=1, 
            n_out=1, 
            n_hidden=n_hidden, 
            n_layers=n_layers,
            layer_norm=layer_norm,
        )

        self.n_in = n_in
       
    def forward(self, x): 
        scalars, x = x
        
        ## Indentity matrix 
        inputIij = torch.cat(
            (
                scalars[:,5:,-1].unsqueeze(-1),
                self.f_Imij(scalars[:,5:,0].unsqueeze(-1)) + self.f_Imij(scalars[:,5:,1].unsqueeze(-1))
            ),
            dim=-1
        ) # [_, 10, 2]
        inputIii = torch.cat(
            (
                scalars[:,:5,-1].unsqueeze(-1),
                self.f_Imii(scalars[:,:5,0].unsqueeze(-1))
            ),
            dim=-1
        ) # [_, 5, 2]
        outI = torch.sum(self.f_Iij(inputIij), dim=1, keepdim=True) # [_, 1, 1]
        outI += torch.sum(self.f_Iii(inputIii), dim=1, keepdim=True) # [_, 1, 1]

        outI = outI * x[:,15,:,:] # [_, 3, 3]
        
        ## Matrices of different particles and same particles
        inputMij = torch.cat(
            (
                scalars[:,5:,-1].unsqueeze(-1),
                self.f_Mmimj(scalars[:,5:,0].unsqueeze(-1)) + self.f_Mmimj(scalars[:,5:,1].unsqueeze(-1)),
                torch.sum(self.f_Mrij(scalars[:,:,-1]), dim=-1, keepdim=True).repeat(1,10).unsqueeze(-1),
                torch.sum(self.f_Mmij(scalars[:,:5,0]), dim=-1, keepdim=True).repeat(1,10).unsqueeze(-1),
            ),
            dim=-1
        ) # [_, 10, 4]

        inputMii = torch.cat(
            (
                scalars[:,:5,-1].unsqueeze(-1),
                self.f_Mmimi(scalars[:,:5,0].unsqueeze(-1)),
                torch.sum(self.f_Mrii(scalars[:,:,-1]), dim=-1, keepdim=True).repeat(1,5).unsqueeze(-1),
                torch.sum(self.f_Mmii(scalars[:,:5,0]), dim=-1, keepdim=True).repeat(1,5).unsqueeze(-1),
            ),
            dim=-1
        ) # [_, 5, 4]
        outMii = self.f_Mii(inputMii).squeeze(-1) # [_, 5, 1]
        outMii = torch.einsum('bi,bijk->bjk', outMii, x[:,:5,:,:]) # [_, 3, 3]

        outMij = self.f_Mij(inputMij).squeeze(-1) # [_, 10, 1]
        outMij = torch.einsum('bi,bijk->bjk', outMij, x[:,5:15,:,:]) # [_, 3, 3]
        
        return (outI+outMij+outMii).view(-1,9) # [_, 9]
        
class EquivarianceLayer(nn.Module):
    def __init__(
        self, 
        n_in, 
        n_hidden, 
        n_layers, 
        layer_norm, 
    ):
        super(self.__class__, self).__init__()
       
        self.f = BasicMLP(
            n_in=n_in, 
            n_out=16, 
            n_hidden=n_hidden, 
            n_layers=n_layers,
            layer_norm=layer_norm,
        )
        

        self.n_in = n_in
       
    def forward(self, x): 
 
        scalars, x = x
        scalars = torch.cat((scalars[:,:,-1],scalars[:,:5,0]),dim=-1) # [_, 20]
        out = torch.sum(self.f(scalars).unsqueeze(-1).unsqueeze(-1) * x, dim=1) # [_, 3, 3]
        # out = torch.einsum('bi,bijk->bjk', self.f(scalars), x) # [_, 3, 3]
        return out.view(-1,9) # [_, 9]
        


