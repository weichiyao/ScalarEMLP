import jax.numpy as jnp
import objax.nn as nn
import objax.functional as F
import numpy as np
from scalaremlp.utils import Named,export
from objax.module import Module

def Sequential(*args):
    """ Wrapped to mimic pytorch syntax"""
    return nn.Sequential(args)

@export
def radial_basis_transform(x, nrad = 100):
    """
    x is a vector
    """
    xmax, xmin = x.max(), x.min()
    gamma = 2*(xmax - xmin)/(nrad - 1)
    mu    = np.linspace(start=xmin, stop=xmax, num=nrad)
    return mu, gamma


def comp_inner_products(x, take_sqrt=True):
    """
    INPUT: batch (q1, q2, p1, p2)
    N: number of datasets
    dim: dimension  
    x: numpy tensor of size [N, 4, dim] 
    """
   
    n = x.shape[0]
    scalars = np.einsum('bix,bjx->bij', x, x).reshape(n, -1) # (n,16)
    if take_sqrt:
        xxsqrt = np.sqrt(np.einsum('bix,bix->bi', x, x)) # (n,4)
        scalars = np.concatenate([xxsqrt, scalars], axis = -1)  # (n,20)
    return scalars 


@export
def compute_scalars(x):
    """Input x of dim [n, 4, 3]"""
    x = np.array(x)    
    xx = comp_inner_products(x)  # (n,20)

    g  = np.array([0,0,-1])
    xg = np.inner(g, x) # (n,4)

    y  = x[:,0,:] - x[:,1,:] # x1-x2 (n,3)
    yy = np.sum(y*y, axis = -1, keepdims=True) # <x1-x2, x1-x2> | (n,) 
    yy = np.concatenate([yy, np.sqrt(yy)], axis = -1) # (n,2)

    yx = np.einsum('bx,bjx->bj', y, x) # <q1-q2, u>, u=q1-q0, q2-q0, p1, p2 | (n, 4)
    
    scalars = np.concatenate([xx,xg,yy,yx], axis=-1) # (n,30)
    return scalars

def comp_inner_products_jax(x:jnp.ndarray, take_sqrt=True):
    """
    INPUT: batch (q1, q2, p1, p2)
    N: number of datasets
    dim: dimension  
    x: numpy tensor of size [N, 4, dim] 
    """ 
    n = x.shape[0]
    scalars = jnp.einsum('bix,bjx->bij', x, x).reshape(n, -1) # (n, 16)
    if take_sqrt:
        xxsqrt = jnp.sqrt(jnp.einsum('bix,bix->bi', x, x)) # (n, 4)
        scalars = jnp.concatenate([xxsqrt, scalars], axis = -1)  # (n, 20)
    return scalars 

def compute_scalars_jax(x:jnp.ndarray, g:jnp.ndarray=jnp.array([0,0,-1])):
    """Input x of dim [n, 4, 3]"""     
    xx = comp_inner_products_jax(x)  # (n,20)

    xg = jnp.inner(g, x) # (n,4)

    y  = x[:,0,:] - x[:,1,:] # q1-q2 (n,3)
    yy = jnp.sum(y*y, axis = -1, keepdims=True) # <q1-q2, q1-q2> | (n,) 
    yy = jnp.concatenate([yy, jnp.sqrt(yy)], axis = -1) # (n,2)

    yx = jnp.einsum('bx,bjx->bj', y, x) # <q1-q2, u>, u=q1-q0, q2-q0, p1, p2 | (n, 4)

    scalars = jnp.concatenate([xx,xg,yy,yx], axis=-1) # (n,30)
    return scalars

@export
class BasicMLP_objax(Module):
    def __init__(
        self, 
        n_in, 
        n_out,
        n_hidden=100, 
        n_layers=2, 
    ):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), F.relu]
        for _ in range(n_layers):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(F.relu)
        layers.append(nn.Linear(n_hidden, n_out))
        
        self.mlp = Sequential(*layers)
    
    def __call__(self,x,training=True):
        return self.mlp(x)

@export
class InvarianceLayer_objax(Module):
    def __init__(
        self,  
        n_hidden, 
        n_layers, 
    ):
        super().__init__()
        self.mlp = BasicMLP_objax(
            n_in=30, n_out=1, n_hidden=n_hidden, n_layers=n_layers
        ) 
        self.g = jnp.array([0,0,-1])
    
    def H(self, x):  
        scalars = compute_scalars_jax(x, self.g)
        out = self.mlp(scalars)
        return out.sum()  
    
    def __call__(self, x:jnp.ndarray):
        x = x.reshape(-1,4,3) # (n,4,3)
        return self.H(x)

@export
class EquivarianceLayer_objax(Module):
    def __init__(
        self,  
        n_hidden, 
        n_layers,
        mu, 
        gamma
    ):
        super().__init__()  
        self.mu       = jnp.array(mu) # (n_rad,)
        self.gamma    = jnp.array(gamma)
        self.n_in_mlp = len(mu)*30
        self.mlp      = BasicMLP_objax(n_in=self.n_in_mlp, n_out=24, n_hidden=n_hidden, n_layers=n_layers) 
        self.g        = jnp.array([0,0,-1]) 
    
    def __call__(self,x,t):
        x = x.reshape(-1,4,3) # (n,4,3)
        scalars = compute_scalars_jax(x, self.g) # (n,30)
        scalars = jnp.expand_dims(scalars, axis=-1) - jnp.expand_dims(self.mu, axis=0)      # (n, 30, n_rad)
        scalars = jnp.exp(-self.gamma * scalars**2)                                         # (n, 30, n_rad)
        scalars = scalars.reshape(-1, self.n_in_mlp)                                        # (n, 30*n_rad)
        out = jnp.expand_dims(self.mlp(scalars), axis=-1)                                   # (n, 24, 1)
         
        y = x[:,0,:] - x[:,1,:]                                                             # x1-x2 (n,3)
        output_x = out[:,:16].reshape(-1,4,4) @ x                                           # (n,4,3)
        output_y = out[:,16:20] * jnp.expand_dims(y,1)                                      # (n,4,3)
        output_g = out[:,20:]   * self.g                                                    # (n,4,3)
        
        output = (output_x+output_y+output_g).reshape(-1,12)
        return output
 
