import jax
import jax.numpy as jnp
import objax.nn as nn
import objax.functional as F
import numpy as np
from emlp.reps import T,Rep,Scalar
from emlp.reps import bilinear_weights
from emlp.reps.product_sum_reps import SumRep
import collections
from oil.utils.utils import Named,export
import scipy as sp
import scipy.special
import random
import logging
from objax.variable import TrainVar, StateVar
from objax.nn.init import kaiming_normal, xavier_normal
from objax.module import Module
import objax
from objax.nn.init import orthogonal
from scipy.special import binom
from jax import jit,vmap
from functools import lru_cache as cache

def Sequential(*args):
    """ Wrapped to mimic pytorch syntax"""
    return nn.Sequential(args)

@export
class Linear(nn.Linear):
    """ Basic equivariant Linear layer from repin to repout."""
    def __init__(self, repin, repout):
        nin,nout = repin.size(),repout.size()
        super().__init__(nin,nout)
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.w = TrainVar(orthogonal((nout, nin)))
        self.rep_W = rep_W = repout*repin.T
        
        rep_bias = repout
        self.Pw = rep_W.equivariant_projector()
        self.Pb = rep_bias.equivariant_projector()
        logging.info(f"Linear W components:{rep_W.size()} rep:{rep_W}")
    def __call__(self, x): # (cin) -> (cout)
        logging.debug(f"linear in shape: {x.shape}")
        W = (self.Pw@self.w.value.reshape(-1)).reshape(*self.w.value.shape)
        b = self.Pb@self.b.value
        out = x@W.T+b
        logging.debug(f"linear out shape:{out.shape}")
        return out

@export
class BiLinear(Module):
    """ Cheap bilinear layer (adds parameters for each part of the input which can be
        interpreted as a linear map from a part of the input to the output representation)."""
    def __init__(self, repin, repout):
        super().__init__()
        Wdim, weight_proj = bilinear_weights(repout,repin)
        self.weight_proj = jit(weight_proj)
        self.w = TrainVar(objax.random.normal((Wdim,)))#xavier_normal((Wdim,))) #TODO: revert to xavier
        logging.info(f"BiW components: dim:{Wdim}")

    def __call__(self, x,training=True):
        # compatible with non sumreps? need to check
        W = self.weight_proj(self.w.value,x)
        out= .1*(W@x[...,None])[...,0]
        return out

@export
def gated(sumrep): #TODO: generalize to mixed tensors?
    """ Returns the rep with an additional scalar 'gate' for each of the nonscalars and non regular
        reps in the input. To be used as the output for linear (and or bilinear) layers directly
        before a :func:`GatedNonlinearity` to produce its scalar gates. """
    return sumrep+sum([Scalar(rep.G) for rep in sumrep if rep!=Scalar and not rep.is_permutation])

@export
class GatedNonlinearity(Module): #TODO: add support for mixed tensors and non sumreps
    """ Gated nonlinearity. Requires input to have the additional gate scalars
        for every non regular and non scalar rep. Applies swish to regular and
        scalar reps. (Right now assumes rep is a SumRep)"""
    def __init__(self,rep):
        super().__init__()
        self.rep=rep
    def __call__(self,values):
        gate_scalars = values[..., gate_indices(self.rep)]
        activations = jax.nn.sigmoid(gate_scalars) * values[..., :self.rep.size()]
        return activations

@export
class EMLPBlock(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """
    def __init__(self,rep_in,rep_out):
        super().__init__()
        self.linear = Linear(rep_in,gated(rep_out))
        self.bilinear = BiLinear(gated(rep_out),gated(rep_out))
        self.nonlinearity = GatedNonlinearity(rep_out)
    def __call__(self,x):
        lin = self.linear(x)
        preact =self.bilinear(lin)+lin
        return self.nonlinearity(preact)

def uniform_rep_general(ch,*rep_types):
    """ adds all combinations of (powers of) rep_types up to
        a total of ch channels."""
    #TODO: write this function
    raise NotImplementedError

@export
def uniform_rep(ch,group):
    """ A heuristic method for allocating a given number of channels (ch)
        into tensor types. Attempts to distribute the channels evenly across
        the different tensor types. Useful for hands off layer construction.
        
        Args:
            ch (int): total number of channels
            group (Group): symmetry group

        Returns:
            SumRep: The direct sum representation with dim(V)=ch
        """
    d = group.d
    Ns = np.zeros((lambertW(ch,d)+1,),int) # number of tensors of each rank
    while ch>0:
        max_rank = lambertW(ch,d) # compute the max rank tensor that can fit up to
        Ns[:max_rank+1] += np.array([d**(max_rank-r) for r in range(max_rank+1)],dtype=int)
        ch -= (max_rank+1)*d**max_rank # compute leftover channels
    sum_rep = sum([binomial_allocation(nr,r,group) for r,nr in enumerate(Ns)])
    sum_rep,perm = sum_rep.canonicalize()
    return sum_rep

def lambertW(ch,d):
    """ Returns solution to x*d^x = ch rounded down."""
    max_rank=0
    while (max_rank+1)*d**max_rank <= ch:
        max_rank += 1
    max_rank -= 1
    return max_rank

def binomial_allocation(N,rank,G):
    """ Allocates N of tensors of total rank r=(p+q) into
        T(k,r-k) for k=0,1,...,r to match the binomial distribution.
        For orthogonal representations there is no
        distinction between p and q, so this op is equivalent to N*T(rank)."""
    if N==0: return 0
    n_binoms = N//(2**rank)
    n_leftover = N%(2**rank)
    even_split = sum([n_binoms*int(binom(rank,k))*T(k,rank-k,G) for k in range(rank+1)])
    ps = np.random.binomial(rank,.5,n_leftover)
    ragged = sum([T(int(p),rank-int(p),G) for p in ps])
    out = even_split+ragged
    return out

def uniform_allocation(N,rank):
    """ Uniformly allocates N of tensors of total rank r=(p+q) into
        T(k,r-k) for k=0,1,...,r. For orthogonal representations there is no
        distinction between p and q, so this op is equivalent to N*T(rank)."""
    if N==0: return 0
    even_split = sum((N//(rank+1))*T(k,rank-k) for k in range(rank+1))
    ragged = sum(random.sample([T(k,rank-k) for k in range(rank+1)],N%(rank+1)))
    return even_split+ragged

@export
class EMLP(Module,metaclass=Named):
    """ Equivariant MultiLayer Perceptron. 
        If the input ch argument is an int, uses the hands off uniform_rep heuristic.
        If the ch argument is a representation, uses this representation for the hidden layers.
        Individual layer representations can be set explicitly by using a list of ints or a list of
        representations, rather than use the same for each hidden layer.

        Args:
            rep_in (Rep): input representation
            rep_out (Rep): output representation
            group (Group): symmetry group
            ch (int or list[int] or Rep or list[Rep]): number of channels in the hidden layers
            num_layers (int): number of hidden layers

        Returns:
            Module: the EMLP objax module."""
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):#@
        super().__init__()
        logging.info("Initing EMLP (objax)")
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        
        self.G=group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch,int): middle_layers = num_layers*[uniform_rep(ch,group)]#[uniform_rep(ch,group) for _ in range(num_layers)]
        elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
        else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
        #assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        logging.info(f"Reps: {reps}")
        self.network = Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        )
    def __call__(self,x,training=True):
        return self.network(x)

def swish(x):
    return jax.nn.sigmoid(x)*x

def MLPBlock(cin,cout):
    return Sequential(nn.Linear(cin,cout),swish)#,nn.BatchNorm0D(cout,momentum=.9),swish)#,

@export
class MLP(Module,metaclass=Named):
    """ Standard baseline MLP. Representations and group are used for shapes only. """
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = Sequential(
            *[MLPBlock(cin,cout) for cin,cout in zip(chs,chs[1:])],
            nn.Linear(chs[-1],cout)
        )
    def __call__(self,x,training=True):
        y = self.net(x)
        return y

@export
class Standardize(Module):
    """ A convenience module to wrap a given module, normalize its input
        by some dataset x mean and std stats, and unnormalize its output by
        the dataset y mean and std stats. 

        Args:
            model (Module): model to wrap
            ds_stats ((μx,σx,μy,σy) or (μx,σx)): tuple of the normalization stats
        
        Returns:
            Module: Wrapped model with input normalization (and output unnormalization)"""
    def __init__(self,model,ds_stats):
        super().__init__()
        self.model = model
        self.ds_stats=ds_stats
    def __call__(self,x,training):
        if len(self.ds_stats)==2:
            muin,sin = self.ds_stats
            return self.model((x-muin)/sin,training=training)
        else:
            muin,sin,muout,sout = self.ds_stats
            y = sout*self.model((x-muin)/sin,training=training)+muout
            return y



# Networks for hamiltonian dynamics (need to sum for batched Hamiltonian grads)
@export
class MLPode(Module,metaclass=Named):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = Sequential(
            *[Sequential(nn.Linear(cin,cout),swish) for cin,cout in zip(chs,chs[1:])],
            nn.Linear(chs[-1],cout)
        )
    def __call__(self,z,t):
        return self.net(z)

@export
class EMLPode(EMLP):
    """ Neural ODE Equivariant MLP. Same args as EMLP."""
    #__doc__ += EMLP.__doc__.split('.')[1]
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):#@
        #super().__init__()
        logging.info("Initing EMLP")
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        self.G=group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch,int): middle_layers = num_layers*[uniform_rep(ch,group)]#[uniform_rep(ch,group) for _ in range(num_layers)]
        elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
        else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
        #print(middle_layers[0].reps[0].G)
        #print(self.rep_in.G)
        reps = [self.rep_in]+middle_layers
        logging.info(f"Reps: {reps}")
        self.network = Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        )
    def __call__(self,z,t):
        return self.network(z)

# Networks for hamiltonian dynamics (need to sum for batched Hamiltonian grads)
@export
class MLPH(Module,metaclass=Named):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = Sequential(
            *[Sequential(nn.Linear(cin,cout),swish) for cin,cout in zip(chs,chs[1:])],
            nn.Linear(chs[-1],cout)
        )
    def H(self,x):#,training=True):
        y = self.net(x).sum()
        return y
    def __call__(self,x):
        return self.H(x)

@export
class EMLPH(EMLP):
    """ Equivariant EMLP modeling a Hamiltonian for HNN. Same args as EMLP"""
    #__doc__ += EMLP.__doc__.split('.')[1]
    def H(self,x):#,training=True):
        y = self.network(x)
        return y.sum()
    def __call__(self,x):
        return self.H(x)

@export
@cache(maxsize=None)
def gate_indices(sumrep): #TODO: add support for mixed_tensors
    """ Indices for scalars, and also additional scalar gates
        added by gated(sumrep)"""
    assert isinstance(sumrep,SumRep), f"unexpected type for gate indices {type(sumrep)}"
    channels = sumrep.size()
    perm = sumrep.perm
    indices = np.arange(channels)
    num_nonscalars = 0
    i=0
    for rep in sumrep:
        if rep!=Scalar and not rep.is_permutation:
            indices[perm[i:i+rep.size()]] = channels+num_nonscalars
            num_nonscalars+=1
        i+=rep.size()
    return indices




##############################################################################
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
        self.mu = mu # (n_rad,)
        self.gamma = gamma
        self.n_in_mlp = len(mu)*30
        self.mlp = BasicMLP_objax(
          n_in=self.n_in_mlp, n_out=24, n_hidden=n_hidden, n_layers=n_layers
        ) 
        self.g = jnp.array([0,0,-1])

    def __call__(self, x, t): 
        x = x.reshape(-1,4,3) # (n,4,3)
        scalars = compute_scalars_jax(x, self.g) # (n,30)
        scalars = jnp.expand_dims(scalars, axis=-1) - jnp.expand_dims(self.mu, axis=0) #(n,30,n_rad)
        scalars = jnp.exp(-self.gamma*(scalars**2)) #(n,30,n_rad)
        scalars = scalars.reshape(-1, self.n_in_mlp) #(n,26*n_rad)
        out = jnp.expand_dims(self.mlp(scalars), axis=-1) # (n,24,1)
         
        y = x[:,0,:] - x[:,1,:] # x1-x2 (n,3) 
        output = jnp.sum(out[:,:16].reshape(-1,4,4,1) * jnp.expand_dims(x, 1), axis=1) # (n,4,3)
        output = output + out[:,16:20] * jnp.expand_dims(y,1)                          # (n,4,3)
        output = output + out[:20:] * jnp.expand_dims(self.g,1)                        # (n,4,3)
    
        # x1 = jnp.sum(out[:,0:4,:]  *x, axis = 1) + out[:,16,:] * y + out[:,20,:] * g #(n,3)
        # x2 = jnp.sum(out[:,4:8,:]  *x, axis = 1) + out[:,17,:] * y + out[:,21,:] * g #(n,3)
        # p1 = jnp.sum(out[:,8:12,:] *x, axis = 1) + out[:,18,:] * y + out[:,22,:] * g #(n,3)
        # p2 = jnp.sum(out[:,12:16,:]*x, axis = 1) + out[:,19,:] * y + out[:,23,:] * g #(n,3)
        # jnp.concatenate([x1,x2,p1,p2], axis=-1)
        return output.reshape(-1, 12) #(n,12)
 
