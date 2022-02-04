import jax
import jax.numpy as jnp
import jax.scipy.stats as jss

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
from functools import partial
import itertools 
from smithnormalform import snfproblem
from smithnormalform import matrix as snfmatrix
from smithnormalform import z as snfz

from typing import Callable, List

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
class BasicMLP_objax(Module):
    def __init__(
        self, 
        n_in: int, 
        n_out: int,
        n_hidden: int = 100, 
        n_layers: int = 2,
        div: int = 2 
    ):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), swish]
        for _ in range(n_layers):
            layers.append(nn.Linear(n_hidden, n_hidden//div))
            layers.append(swish)
            n_hidden //= div 
        layers.append(nn.Linear(n_hidden, n_out))
        
        self.mlp = Sequential(*layers)
    
    def __call__(self,x,training=True):
        return self.mlp(x)

 
@export
class ScalarMLP(Module, metaclass=Named):
    """ Scalar Invariant MultiLayer Perceptron. 
    Arguments 
    -------
    n_in : int
        number of inputs to MLP
    n_hidden: int
        number of hidden units in MLP
    n_layers: int
        number of layers between input and output layers
    div: int
        scale the number of hidden units at each subsequent layer of MLP 
    transformer: Callable
        transformation attributes and functions
          
    Returns:
    -------
    Module: 
        the ScalarMLP objax module.
    """
    def __init__(
        self, 
        n_hidden: int, 
        n_layers: int, 
        div: int,
        transformer: Callable, 
    ): 
        super().__init__()  
         
        n_in = transformer.n_features
        self.transformer = transformer  

        self.mlp = BasicMLP_objax(
            n_in=n_in, n_out=1, n_hidden=n_hidden, n_layers=n_layers, div=div
        )   
    
    def H(self, x, xp):  
        scalars, _ = self.transformer(x,xp)  
        out = self.mlp(scalars)
        return out.sum()  
    
    def __call__(self, x, xp, training = True):
        return self.H(x, xp)

@export
class EquivarianceLayer_objax(ScalarMLP):
    def __call__(self, x, t, xp):
        scalars, _ = self.transformer(x, xp) # (n, n_in)
        out = jnp.expand_dims(self.mlp(scalars), axis=-1) # (n,24,1)
        
        y = x[:,0,:] - x[:,1,:] # x1-x2 (n,3)
        out = jnp.sum(out[:,:16].reshape(-1,4,4,1) * jnp.expand_dims(x, 1), axis=1) # (n,4,3)
        out += out[:,16:19] * jnp.expand_dims(y,1) # (n,4,3)
        out += out[:19:] * jnp.expand_dims(g,1) # (n,4,3)
    
        # x1 = jnp.sum(out[:,0:4,:]  *x, axis = 1) + out[:,16,:] * y + out[:,20,:] * g #(n,3)
        # x2 = jnp.sum(out[:,4:8,:]  *x, axis = 1) + out[:,17,:] * y + out[:,21,:] * g #(n,3)
        # p1 = jnp.sum(out[:,8:12,:] *x, axis = 1) + out[:,18,:] * y + out[:,22,:] * g #(n,3)
        # p2 = jnp.sum(out[:,12:16,:]*x, axis = 1) + out[:,19,:] * y + out[:,23,:] * g #(n,3)
        
        return out.reshape(-1, 12) #(n,12)

@export  
class InvarianceLayer_objax(ScalarMLP):
    def __init__(
        self, 
        n_hidden: int, 
        n_layers: int, 
        div: int,
        transformer: Callable, 
    ):   
        n_in = transformer.n_features
        self.transformer = transformer  

        self.mlp1 = BasicMLP_objax(
            n_in=n_in, n_out=1, n_hidden=n_hidden, n_layers=n_layers, div=div
        )   
        self.mlp2 = BasicMLP_objax(
            n_in=n_in, n_out=1, n_hidden=n_hidden, n_layers=n_layers, div=div
        )   
    
    def H(self, x, xp):  
        scalars, scaling = self.transformer(x,xp)  
        out = scaling[...,0] * self.mlp1(scalars) + scaling[...,1] * self.mlp2(scalars)
        return out.sum()  
    
    def __call__(self, x, xp, training = True):
        return self.H(x, xp)
 

@export
class Dimensionless(object):
    def __init__(self):
        self.create_mapping()
        self.adjust_mapping()
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
        _, S_f = a_xs.shape
        foo = len(vectors)
        J_f = len(scalars) + foo * (foo + 1) // 2
        names_f = np.zeros(J_f).astype(str)
        a_xs_f = np.zeros((J_f, S_f)).astype(int) 
        powers_f = np.zeros(J_f).astype(int)
        for i, j in enumerate(scalars):
            names_f[i] = x_names[j]
            a_xs_f[i] = a_xs[j]
            powers_f[i] = 1
        i += 1
        for ii, j1 in enumerate(vectors):
            names_f[i] = "|" + x_names[j1] + "|"
            a_xs_f[i] = a_xs[j1]
            powers_f[i] = 1
            i += 1
            for j2 in vectors[ii+1:]:
                names_f[i] = "(" + x_names[j1] + "^\\top " + x_names[j2] + ")"
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
        nh: final scaling matrix (2, 21)
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

    def adjust_mapping(self):
        self.h[0] = -self.h[0]
        self.h[3] = -self.h[3]
        self.h[4] = -self.h[4]

        h0_new = np.concatenate([[0], self.h[0][:-1]])
        h0_new[4] = 0
        h0_new[2] = -1

        h3_new = np.concatenate([[0], self.h[3][:-1]])
        h3_new[7] = 0
        h3_new[6] = -1

        h4_new = np.concatenate([[0], self.h[4][:-1]]) 

        
        
        self.h[5] = -self.h[5]
        assert self.h[5][7] == 3
        assert self.h[5][8] == -1 
        self.h[5][7] = 0
        self.h[5][8] = 0 
        h5_new = np.concatenate([[0], self.h[5][:-1]])
        self.h[5][6] = -1 
        self.h[5][11] = 2 
        h5_new[6] = -1
        h5_new[15] = 2
        assert self.h[10][7] == -1 
        assert self.h[10][13] == 1 
        assert self.h[11][7] == -1 
        assert self.h[11][14] == 1 
        assert self.h[11][0] == -1
        assert self.h[11][2] == 1 
        self.h[10][7] = 0
        self.h[10][13] = 0 
        self.h[11][7] = 0 
        self.h[11][14] = 0 
        self.h[10][9] = 1
        self.h[11][10] = 1 

        self.h[11][0] = 0
        self.h[11][2] = 0
        self.h[11][1] = -1 
        self.h[11][3] = 1
        assert self.h[15][4] == -1
        assert self.h[16][4] == -2
        assert self.h[17][4] == -1
        assert self.h[15][18] == 1
        assert self.h[17][20] == 1
        self.h[16][4] = -1
        self.h[16][5] = -1

        self.h[15][4] = -2 
        self.h[17][4] = 0
        self.h[17][5] = -2

        self.h[15][18] = 2
        self.h[17][20] = 2

        keepidx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17] 
        h_ad = np.zeros((len(keepidx),self.h.shape[1]))
        for i in range(len(keepidx)):
            h_ad[i] = self.h[keepidx[i]] 
        h_ad[6] = h0_new 
        h_ad[7] = h3_new
        h_ad[8] = h4_new
        h_ad[9] = h5_new 
        h_ad[12] = -self.h[1] 
        h_ad[13] = -self.h[2]    

        self.h = h_ad
        self.nh = np.zeros((2,self.h.shape[1]))
        self.nh[0][2] = 1
        self.nh[1][3] = 1
        self.nh[0][4] = 2
        self.nh[1][5] = 2

    def map_m_func(self, params, x):
        """x (d,) and h (m,d) gives output (m,)"""
        return jnp.prod(x**params, axis=-1)

    # def map_s_func(self, params, x):
    #     """x (d,) and h (2,d) gives output (1,)"""
    #     return jnp.sum(jnp.prod(x**params, axis=-1), keepdims=True)

    def __call__(self, x, expand=True):
        """
        Input 
        - x: scalars with dimensions (n, d)
        Output
        - x_dl: dimensionless scalars (n, m)
        - x_sc: scaling (n,2)  
        """
        ## Broadcasting: (n, d) & (m, d) => (n, m, d) => (n, m)
        x_dl = jit(vmap(jit(partial(self.map_m_func, self.h))))(x)
        if expand:
            # Expand the scalar set by considering functions of the scalars 
            x_dl = jnp.concatenate(
                [x_dl, jnp.sqrt(x_dl[...,[0,1,2,3,4,5,6,7,8,9,12,13,14,16]])], 
                axis=-1
            )
        ## Broadcasting: (n, d) & (2, d) => (n, 2, d) => (n, 2)
        x_sc = jit(vmap(jit(partial(self.map_m_func, self.nh))))(x) 
        return x_dl, x_sc 
      

@export
class ScalarTransformer(object):
    """Transform (dimensionless) features using quantiles info or radial basis function.
    
    During the initialization stage, this method takes the whole training data set, 
    zs (n,4,3) and zps (n,9), and conducts the following transformation: 
                zs, zps
        Step 0  => inner product scalars 
        Step 1  => dimensionless scalars                   (optional, dimensionless = True)
        Step 2  => rbf (or quantile) transformed scalars   (optional, method = 'rbf' (method = 'qt'))
    Either one or both of the two optional steps can be skipped. 
    If method is not 'none', transformer information (parameters) is recorded.
    
    When the Object is called, the input scalars go through: 
        Step 1 (optional)
        Step 2 (optional)
    depending on the values of the arguments, dimensionless and method
            
    
    Arguments
    ----------
    zs : jax.numpy.ndarray (n, 4, 3)
        Double pendulum TRAINING data positions and velocities q1, q2, p1, p2

    zps : jax.numpy.ndarray (n, 9) 
        Double pendulum TRAINING data parameters g, m1, m2, k1, k2, l1, l2
    
    dimensionless : bool
        whether we want to make the scalars dimensionless 
    
    method : str, 'qt' or 'rbf' or 'none'

    n_quantiles : int, default=1000 or n
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.


    Attributes
    ----------
    n_features : int 
        number of features in the (dimensionless) (transformed) output

    dimensionless_operator : lambda x : (x,1) or emlp.nn.objax.Dimensionless() funcion
        

    ===========================================================================
    rbf : Transform features using radial basis function

    qt: Transform features using quantiles information  
        Adapted from scikit_learn.preprocessing.QuantileTransformer. 

        This method transforms the features to follow a uniform or a normal
        distribution. Therefore, for a given feature, this transformation tends
        to spread out the most frequent values. It also reduces the impact of
        (marginal) outliers: this is therefore a robust preprocessing scheme.
        
        The transformation is applied on each feature independently. 
        
        Features values of new/unseen data that fall below or above the fitted range 
        will be mapped to the bounds of the output distribution. 
        
        Note that this transform is non-linear. It may distort linear
        correlations between variables measured at the same scale but renders
        variables measured at different scales more directly comparable. 

    none : Only perform standardization for each feature 
    """
    def __init__(
        self, 
        zs, 
        zps, 
        method: str = 'none', 
        dimensionless: bool = False,
        n_rad: int = 100,
        n_quantiles: int = 1000, 
        transform_distribution: str = 'uniform'   
    ):  
        zs, zps = jnp.array(zs), jnp.array(zps)
        # Create mapping idex for inner product scalars computation
        self._create_index()

        # Create inner product scalars
        scalars = self._compute_scalars(
            zs, 
            zps[:,:3], 
            zps[:,3:]
        )   

        self.dimensionless_operator = lambda x: (x, jnp.array([1,1])) 
        self.scaling_standardization = jnp.array([[0,0],[1,1]])
        if dimensionless:
            # Create dimensionless features 
            self.dimensionless_operator = Dimensionless() 
            scalars, scaling = self.dimensionless_operator(scalars)
            print(f"scaling max={jnp.max(scaling)} and min={jnp.min(scaling)}")
            self.scaling_standardization = jnp.stack(
                [jnp.min(scaling, axis=0), jnp.max(scaling, axis=0) - jnp.min(scaling, axis=0)],
                axis = 0
            )
            

        self.method = method
        self.dimensionless = dimensionless
        self.n_rad = n_rad
        self.n_quantiles = n_quantiles
        self.transform_distribution = transform_distribution
        # Create the quantiles of reference
        self.references = jnp.linspace(0, 1, self.n_quantiles, endpoint=True)
        self.BOUNDS_THRESHOLD = 1e-7 
        self.spacing = jnp.array(np.spacing(1)) 
        
        self._GETPARAMS = {
            'qt': self._get_qt_params,
            'rbf': self._get_rbf_params, 
            'none': self._get_none_params
        }
        
        # Compute the global quansformation parameters
        self._GETPARAMS[self.method](scalars)
       
        self._TRANSFORMS = {
            'qt': self._qt_transform,
            'rbf': self._rbf_transform,
            'none': self._none_transform
        }

    def _create_index(self):
        """create the indexing for the construction of the inner product scalars"""
        idx = jnp.array(
            list(itertools.combinations_with_replacement(jnp.arange(0,4), r=2))
        )
        self.idx_map = jnp.array([2,3,0,1])
        self.idx = self.idx_map[idx] 
        self.idx_sqrt = jnp.concatenate(
            [jnp.zeros(1, dtype=int), jnp.cumsum(jnp.arange(2,5)[::-1])], 
            axis=0
        ) 

    def _get_none_params(self, x):
        """Gets parameters for standardization transformation:
        
        Arguments 
        -----------
        x : jax.numpy.ndarray (n, d)

        Returns
        -----------
        params : jax.numpy.ndarray (2, d)
            params[0] gives mean
            params[1] gives std

        n_features: int 
            number of features in the transformed output

        """ 
        # If transformer method == "none":
        self.parameters = jnp.stack([jnp.mean(x, axis = 0), jnp.std(x, axis = 0)], axis = 0)
        self.n_features = x.shape[-1]

    def _get_rbf_params(self, x):
        """Gets parameters for Radial Basis Function Transformation:
        
        Arguments 
        -----------
        x : jax.numpy.ndarray (n, d)

        n_rad : int
            number of transformed outputs for each d

        Returns
        -----------
        params : jax.numpy.ndarray (n_rad+1, d)
            params[0] gives gamma (1, d)
            params[1:] gives mu (n_rad, d)

        n_features: int 
            number of features in the transformed output

        Remarks
        -----------
        Given mu (n,d) and gamma (d,), RBF for x (n,d) gives x_trans (n,n_rad)
        x_trans[:,i] = exp(-gamma[:,i]*(x[:,i]-mu[:,i])**2), i=1,...,d

        """ 
        xmin = jnp.min(x, axis=0, keepdims=True) # (1,d) 
        xmax = jnp.max(x, axis=0, keepdims=True) # (1,d)
        gamma = 2*(xmax - xmin)/(self.n_rad - 1) # (1,d)
        mu    = jnp.linspace(start=xmin[0], stop=xmax[0], num=self.n_rad) # (nrad, d)
        self.parameters = jnp.concatenate([gamma, mu], axis=0) # (n_rad+1,d)
        self.n_features = x.shape[1]*self.n_rad
        
    def _get_qt_params(self, x): 
        """Gets parameters for Quantile Transformation
        
        Arguments:
        ----------
        x : jax.numpy.ndarray (n, d)

        self.references : (array_like of float) 
            Percentile or sequence of percentiles to compute, 
            which must be between 0 and 100 inclusive.

        Returns 
        ---------
        quantiles : numpy.ndarray of shape (n_quantiles, d)
            The values corresponding the quantiles of reference.

        n_features: int 
            number of features in the transformed output
        """ 
        self.parameters = jnp.nanpercentile(x, self.references*100, axis=0)
        self.n_features = x.shape[1] 
    
    def _none_transform(self, X):
        return (X-self.parameters[0])/self.parameters[1]

    def _rbf_transform(self, X):
        """RBF : Transform features using radial basis function 

        Arguments
        ----------
        X : jax.numpy.ndarray (n, d)

        self.parameters : jax numpy ndarray (n_rad+1, d)
            self.parameters[0] gives gamma (1, d)
            self.parameters[1:] gives mu (n_rad, d)

        Returns
        ----------
        X : jax.numpy.ndarray (n, n_rad*d)
        """
        n = X.shape[0]
        return jnp.exp(-self.parameters[0] * (X-self.parameters[1:])**2).reshape(n,-1)
    
    def _qt_transform(self, X): 
        """Forward quantile transform.
        
        Arguments
        ----------
        X : jax.numpy.ndarray of shape (n, d)
            The data used to scale along the features axis. 

        self.parameters : jax.numpy.ndarray of shape (n_quantiles, d)
            The values corresponding the quantiles of reference.
        
        self.transform_distribution : {'uniform', 'normal'}, default='uniform'
            Marginal distribution for the transformed data. The choices are
            'uniform' (default) or 'normal'.

        Returns
        -------
        X : jax ndarray of shape (n, d)
            Projected data.
        """ 
        
        X = vmap(jit(self._qt_transform_col), in_axes=(1,1), out_axes=1)(X, self.parameters) 
        return X

    def _qt_transform_col(self, X_col, params):
        """Private function to forward transform a single feature."""
        lower_bound_x = params[0]
        upper_bound_x = params[-1]
        lower_bound_y = 0
        upper_bound_y = 1
        n = X_col.shape[0]

        lower_bounds_idx = jnp.nonzero(
            X_col - self.BOUNDS_THRESHOLD < lower_bound_x, 
            size = n, 
            fill_value = n+1
        )
         
        upper_bounds_idx = jnp.nonzero(
            X_col + self.BOUNDS_THRESHOLD > upper_bound_x, 
            size = n, 
            fill_value = n+1
        )
        
        # Interpolate in one direction and in the other and take the
        # mean. This is in case of repeated values in the features
        # and hence repeated quantiles
        #
        # If we don't do this, only one extreme of the duplicated is
        # used (the upper when we do ascending, and the
        # lower for descending). We take the mean of these two
          
        X_col = 0.5 * (
            jnp.interp(X_col, params, self.references)
            - jnp.interp(-X_col, -params[::-1], -self.references[::-1])
        )
         
        X_col = X_col.at[upper_bounds_idx].set(upper_bound_y)
        X_col = X_col.at[lower_bounds_idx].set(lower_bound_y)

        # for forward transform, match the output PDF
        if self.transform_distribution == "normal":  
            X_col = jss.norm.ppf(X_col)
            # find the value to clip the data to avoid mapping to
            # infinity. Clip such that the inverse transform will be
            # consistent
            clip_min = jss.norm.ppf(self.BOUNDS_THRESHOLD - self.spacing)
            clip_max = jss.norm.ppf(1 - (self.BOUNDS_THRESHOLD - self.spacing))
            X_col = jnp.clip(X_col, clip_min, clip_max)
        
        # else output distribution is uniform and the ppf is the
        # identity function so we let X_col unchanged
        return X_col
    
    def _compute_scalars(self, x, g, mkl):
        """Input x of dim (n,4,3), g of dim (n,3), mkl = (m1, m2, k1, k2, l1, l2) of (n,6)"""
        # get q2 - q1 replacing q2
        x = x.at[:,1,:].set(x[:,1,:]-x[:,0,:])  
        xx = jnp.sum(
            x[:,self.idx[:,0],:] * x[:,self.idx[:,1],:], 
            axis=-1
        ) # (n, 10)
        gx = jnp.einsum("...j,...ij", g, x[:,self.idx_map]) # (n, 4)
        gg = jnp.sum(g*g, axis = -1, keepdims=True) # (n, 1)
        if self.dimensionless:
            xx = xx.at[:,self.idx_sqrt].set(jnp.sqrt(xx[:,self.idx_sqrt])) 
            ## all the current scalars we have 
            scalars = jnp.concatenate([mkl, jnp.sqrt(gg), gx, xx], axis = -1) # (n, 21)
        else:
            xxsqrt = jnp.sqrt(xx[:,self.idx_sqrt]) # (n, 4)
            ggsqrt = jnp.sqrt(gg) # (n, 1)
            scalars = jnp.concatenate(mkl, gg, gx, xx, ggsqrt, xxsqrt, 1/mkl) # (n, 32)
        return scalars  

    def __call__(self, xs, xps):
        g, mkl = xps[...,:3], xps[...,3:] # (n,3), (n,6)  
        # Compute inner product scalars 
        scalars = self._compute_scalars(
            xs.reshape(-1,4,3), g.reshape(-1,3), mkl.reshape(-1,6)
        )
        # Create dimensionless features 
        scalars, scaling = self.dimensionless_operator(scalars)
        
        # Standardize scalings 
        scaling = (scaling - self.scaling_standardization[0]) / self.scaling_standardization[1]
        
        return jit(self._TRANSFORMS[self.method])(scalars), scaling

    
    
