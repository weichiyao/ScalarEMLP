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
def compute_scalars(zs, zps):
    """Input zs of dim [n, 4, 3], zps of dim [n, 9] = (g, (m1,k1,l1), (m2,k2,l2))"""
    g = zps[...,:3] # (n,6)
    mkl = zps[...,3:] # (n,6)
    
    zs = np.array(zs)    
    zs_prod = comp_inner_products(zs)  # (n,20) 
    xg = np.einsum('...j, ...ij', g, zs)  # (n,4)
    zs_diff  = zs[:,0,:] - zs[:,1,:] # x1-x2 (n,3)
    zs_diff = np.sum(zs_diff*zs_diff, axis=-1, keepdims=True) # <x1-x2, x1-x2> | (n,1) 
    scalars = np.concatenate([zs_prod, xg, zs_diff, np.sqrt(zs_diff), mkl], axis=-1) # (n,32)
    return scalars # (n,32)


def comp_inner_products_jax(x, take_sqrt=True):
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
        n_layers
    ):
        super().__init__()
        self.mu = mu # (n_rad,)
        self.gamma = gamma 
        
        self.n_in_mlp = 26
            
        self.mlp = BasicMLP_objax(
          n_in=self.n_in_mlp, n_out=1, n_hidden=n_hidden, n_layers=n_layers
        )  
        
        idx = jnp.array(
            list(itertools.combinations_with_replacement(jnp.arange(0,4), r=2))
        )
        self.idx_map = jnp.array([2,3,0,1])
        self.idx = self.idx_map[idx] 
        self.idx_sqrt = jnp.concatenate(
            [jnp.zeros(1,dtype=int), jnp.cumsum(jnp.arange(2,5)[::-1])], 
            axis=0
        ) 

    def compute_scalars_jax(self, x, g, mkl):
        """Input x: (n, 4, 3), g: (n, 3), mkl: (n, 6)""" 
        # get q2 - q1 replacing q2
        x = x.at[:,1,:].set(x[:,1,:]-x[:,0,:])  
        xx = jnp.sum(
            x[:,self.idx[:,0],:]*x[:,self.idx[:,1],:], 
            axis=-1
        ) # (n,10)
        ## take square root of p_1^\top p_1, p_2^\top p_2, q_1^\top q_1, (q_2-q_1)^\top (q_2-q_1)
        xs = jnp.sqrt(xx[:,self.idx_sqrt]) # (n,4)
        gx = jnp.einsum("...j,...ij", g, x[:,self.idx_map]) # (n,4)
        gg = jnp.sum(g*g, axis = -1, keepdims=True) #(n,1)
        
        ## all the current scalars we have 
        scalars = jnp.concatenate([mkl, gg, jnp.sqrt(gg), gx, xx, xs], axis = -1) # (n, 26)
        return scalars  
    
    def H(self,x, xp): 
        x = x.reshape(-1,4,3) 
        g, mkl = xp[...,:3], xp[...,3:] # (n,3), (n,6)  
        
        scalars = self.compute_scalars_jax(x, g.reshape(-1,3), mkl.reshape(-1,6))
        if self.mu is not None:
            scalars = jnp.expand_dims(scalars, axis=-1) - jnp.expand_dims(self.mu, axis=0) #(n,32,n_rad)
            scalars = jnp.exp(-self.gamma*(scalars**2)) #(n,32,n_rad)
            scalars = scalars.reshape(-1, self.n_in_mlp) #(n,32*n_rad)
        out = self.mlp(scalars)
        return out.sum()  
    
    def __call__(self, x, xp):
        return self.H(x, xp)

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
        self.n_in_mlp = len(mu)*26
        self.mlp = BasicMLP_objax(
          n_in=self.n_in_mlp, n_out=24, n_hidden=n_hidden, n_layers=n_layers
        ) 

        idx = jnp.array(
            list(itertools.combinations_with_replacement(jnp.arange(0,4), r=2))
        )
        self.idx_map = jnp.array([2,3,0,1])
        self.idx = self.idx_map[idx] 
        self.idx_sqrt = jnp.concatenate(
            [jnp.zeros(1,dtype=int), jnp.cumsum(jnp.arange(2,5)[::-1])], 
            axis=0
        ) 
        
    def compute_scalars_jax(self, x, g, mkl):
        """Input x of dim [n, 4, 3]"""       
        # get q2 - q1 replacing q2
        x = x.at[:,1,:].set(x[:,1,:]-x[:,0,:])  
        xx = jnp.sum(
            x[:,self.idx[:,0],:]*x[:,self.idx[:,1],:], 
            axis=-1
        ) # (n,10)
        ## take square root of p_1^\top p_1, p_2^\top p_2, q_1^\top q_1, (q_2-q_1)^\top (q_2-q_1)
        xs = jnp.sqrt(xx[:,self.idx_sqrt]) # (n,4)
        gx = jnp.einsum("...j,...ij", g, x[:,self.idx_map]) # (n,4)
        gg = jnp.sum(g*g, axis = -1, keepdims=True) #(n,1)
        
        ## all the current scalars we have 
        scalars = jnp.concatenate([mkl, gg, jnp.sqrt(gg), gx, xx, xs], axis = -1) # (n, 26)
        return scalars  

    def __call__(self, x, t, xp):
        g, mkl = xp[...,:3], xp[...,3:] # (n,3), (n,6)  
         
        x = x.reshape(-1,4,3) # (n,4,3)
        scalars = self.compute_scalars_jax(x, g.reshape(-1,3), mkl.reshape(-1,6)) # (n,26)
        scalars = jnp.expand_dims(scalars, axis=-1) - jnp.expand_dims(self.mu, axis=0) #(n,26,n_rad)
        scalars = jnp.exp(-self.gamma*(scalars**2)) #(n,26,n_rad)
        scalars = scalars.reshape(-1, self.n_in_mlp) #(n,26*n_rad)
        out = jnp.expand_dims(self.mlp(scalars), axis=-1) # (n,24,1)
        
        y = x[:,0,:] - x[:,1,:] # x1-x2 (n,3)
        x1 = jnp.sum(out[:,0:4,:]  *x, axis = 1) + out[:,17,:] * y + out[:,21,:] * g #(n,3)
        x2 = jnp.sum(out[:,4:8,:]  *x, axis = 1) + out[:,18,:] * y + out[:,22,:] * g #(n,3)
        p1 = jnp.sum(out[:,8:12,:] *x, axis = 1) + out[:,19,:] * y + out[:,23,:] * g #(n,3)
        p2 = jnp.sum(out[:,12:16,:]*x, axis = 1) + out[:,20,:] * y + out[:,24,:] * g #(n,3)
        
        return jnp.concatenate([x1,x2,p1,p2], axis=-1) #(n,12)
 
class Dimensionless(object):
    def __init__(self):
        self.create_mapping()
        self.h = jnp.array(self.h)
        self.nh = jnp.array(self.nh)

    def make_base(self):
        """
        Make powers of the base dimensions 
        """
        names = np.array(
            ["m_1", "m_2", "k_1", "k_2", "l_1", "l_2", 
             "g", "p_1", "p_2", "q_1", "(q_2-q_1)"]
        )
        a_xs = np.array([[1, 0,  0], # kg
                         [1, 0,  0], # kg
                         [1, 0, -2], # N / m = kg / s^2
                         [1, 0, -2], # N / m = kg / s^2
                         [0, 1,  0], # m
                         [0, 1,  0], # m
                         [0, 1, -2], # m / s^2
                         [1, 1, -1], # kg m / s
                         [1, 1, -1], # kg m / s
                         [0, 1,  0], # m
                         [0, 1,  0]] # m
                        ).astype(int)
        scalars = np.array([0, 1, 2, 3, 4, 5]).astype(int)
        vectors = np.array([6, 7, 8, 9, 10]).astype(int)
        a_y = np.array([1, 2, -2]).astype(int) # J = kg m^2 / s^2 
        return a_xs, a_y, names, scalars, vectors

    def make_flat_features(self, a_xs, x_names, scalars, vectors):
        """
        reformat the blobby features into flat, scalar features:
        hack; totally non-pythonic
        """
        J, S_f = a_xs.shape
        foo = len(vectors)
        J_f = len(scalars) + foo * (foo + 1) // 2
        names_f = np.zeros(J_f).astype(str)
        a_xs_f = np.zeros((J_f, S_f)).astype(int)
        j_f = 0
        powers_f = np.zeros(J_f).astype(int)
        for i,j in enumerate(scalars):
            names_f[i] = x_names[j]
            a_xs_f[i] = a_xs[j]
            powers_f[i] = 1
        i += 1
        for ii,j1 in enumerate(vectors):
            names_f[i] = "|" + names[j1] + "|"
            a_xs_f[i] = a_xs[j1]
            powers_f[i] = 1
            i += 1
            for j2 in vectors[ii+1:]:
                names_f[i] = "(" + names[j1] + "^\\top " + names[j2] + ")"
                a_xs_f[i] = a_xs[j1] + a_xs[j2] # because we are multiplying
                powers_f[i] = 2
                i+= 1
        return a_xs_f, names_f, powers_f
    
    def hogg_msv_integer_solve(self, A, b):
        """
        Find all solutions to Ax=b where A, x, b are integer.

        ## inputs:
        - A - [n, m] integer matrix - n < m, please
        - b - [n] integer vector

        ## outputs:
        - vv - [m] integer vector solution to the problem Ax=b
        - us - [k, m] set of k integer vector solutions to the problem Ax=0

        ## bugs / issues:
        - Might get weird when k <= 1.
        - Might get weird if k > m - n.
        - Depends EXTREMELY strongly on everything being integer.
        - Uses smithnormalform package, which is poorly documented.
        - Requires smithnormalform package to have been imported as follows:
            !pip install smithnormalform
            from smithnormalform import snfproblem
            from smithnormalform import matrix as snfmatrix
            from smithnormalform import z as snfz
        """
        ## perform the horrifying packing into SNF Matrix format; HACK
        n, m = A.shape
        assert(m >= n)  
        assert(len(b) == n)
        assert A.dtype is np.dtype(int)
        assert b.dtype is np.dtype(int)
        smat = snfmatrix.Matrix(n, m, [snfz.Z(int(a)) for a in A.flatten()])
        ## calculate the Smith Normal Form 
        prob = snfproblem.SNFProblem(smat)
        prob.computeSNF()
        ## perform the horrifying unpacking from SNF Matrix form; HACK
        SS = np.array([a.a for a in prob.S.elements]).reshape(n, n)
        TT = np.array([a.a for a in prob.T.elements]).reshape(m, m)
        JJ = np.array([a.a for a in prob.J.elements]).reshape(n, m)
        ## Find a basis for the lattice of null vectors
        us = None
        zeros = np.sum(JJ ** 2, axis=0) == 0
        us = (TT[:, zeros]).T
        DD = SS @ b
        v = np.zeros(m)
        v[:n] = DD / np.diag(JJ)
        vv = (TT @ v).astype(int) 
        return vv, us

    def create_mapping(self): 
        """
        nh: final scaling (2, 21)
        h: dimentionless matrix (18,21)
        """
        a_xs, a_y, names, scalars, vectors = self.make_base()
        a_xs_flat, names_flat, _ = self.make_flat_features(a_xs, names, scalars, vectors)
        _, self.h = self.hogg_msv_integer_solve(a_xs_flat.T, a_y) 
        _, m = self.h.shape
        self.nh = np.zeros((2,m))
        # manually make the scaling  
        self.nh[0][names_flat=="m_1"], self.nh[0][names_flat=="|p_1|"] = -1, 2
        self.nh[1][names_flat=="m_2"], self.nh[1][names_flat=="|p_2|"] = -1, 2

    def __call__(self, x):
        """
        Input 
        - x: scalars with dimensions (n, 21)
        Output
        - x_dl: dimensionless scalars (n, 18)
        - x_s: scaling (n,1)  
        """
        ## Broadcasting: (n, 21) & (18, 21) => (n, 18, 21) => (n, 18)
        x_dl = jnp.prod(jnp.expand_dims(x, axis = 1) ** self.h, axis=-1) 
        x_sc = jnp.prod(jnp.expand_dims(x, axis = 1) ** self.nh, axis=-1)
        x_sc = jnp.sum(x_sc, axis = -1, keepdims=True) 
        return x_dl, x_sc 
     
@export
class InvarianceLayerDL_objax(Module):
    def __init__(
        self,   
        n_hidden, 
        n_layers, 
    ):
        super().__init__() 
        self.n_in_mlp = 36
        self.mlp = BasicMLP_objax(
          n_in=self.n_in_mlp, n_out=1, n_hidden=n_hidden, n_layers=n_layers
        )  
        self.createDimensionless = Dimensionless()

        idx = jnp.array(
            list(itertools.combinations_with_replacement(jnp.arange(0,4), r=2))
        )
        self.idx_map = jnp.array([2,3,0,1])
        self.idx = self.idx_map[idx] 
        self.idx_sqrt = jnp.concatenate(
            [jnp.zeros(1,dtype=int), jnp.cumsum(jnp.arange(2,5)[::-1])], 
            axis=0
        ) 
        print("Invariance Dimensionless")
        
    def compute_scalars_jax(self, x, g, mkl):
        """Input x: (n, 4, 3), g: (n, 3), mkl: (n, 6)""" 
        # get q2 - q1 replacing q2
        x = x.at[:,1,:].set(x[:,1,:]-x[:,0,:])  
        xx = jnp.sum(
            x[:,self.idx[:,0],:]*x[:,self.idx[:,1],:], 
            axis=-1
        )
        ## take square root of p_1^\top p_1, p_2^\top p_2, q_1^\top q_1, (q_2-q_1)^\top (q_2-q_1)
        xx = xx.at[:,self.idx_sqrt].set(jnp.sqrt(xx[:,self.idx_sqrt]))
        gx = jnp.einsum("...j,...ij", g, x[:,self.idx_map])
        gg = jnp.sqrt(jnp.sum(g*g, axis = -1, keepdims=True))
  
        ## all the current scalars we have 
        scalars = jnp.concatenate([mkl, gg, gx, xx], axis = -1) # (n, 21)
        return scalars  
    
    def H(self, x, xp):  
        g, mkl = xp[...,:3], xp[...,3:] # (n,3), (n,6)   
        scalars = self.compute_scalars_jax(
            x.reshape(-1,4,3), 
            g.reshape(-1,3), 
            mkl.reshape(-1,6)
        ) 
        ## make dimensionless
        scalars, scaling = createDimensionless(scalars) 
        scalars = jnp.concatenate(
            [scalars,  jnp.reciprocal(scalars)], 
            axis = -1
        ) # (n, 36)
        out = scaling * self.mlp(scalars).sum() 
        return out
    
    def __call__(self, x, xp):
        return self.H(x, xp)
