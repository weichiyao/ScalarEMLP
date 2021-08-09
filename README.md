# Scalars are universal: Gauge-equivariant machine learning, structured like classical physics

## Authors:
**This project is currently anonymous as per Neurips 2021 requiments.** Some contributor names appear to the right to this github page because we imported their codes from their public github repository [equivariant-MLP](https://github.com/mfinzi/equivariant-MLP.git). 

## Introduction
Scalar products and scalar contractions of the scalar, vector, and tensor inputs are powerful. Deep learning models using these scalars respect the gauge symmetries—or coordinate freedom—of physical law.

This repository provides codes to construct invariant and equivariant neural networks on the same datasets used in ICML2021 paper [A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups](https://arxiv.org/abs/2104.09459) 

To compare the simulation results with those using *EMLP* proposed in [A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups](https://arxiv.org/abs/2104.09459), this repository imported codes from [equivariant-MLP](https://github.com/mfinzi/equivariant-MLP.git). 


## Experimental Results from Paper
To run the scripts you will need to clone the repo and install it locally which you can do with
```
!git clone https://github.com/Pamplemousse-Elaina/Comparison_EMLP.git
cd Comparison_EMLP
pip install -e .
```

### Synthetic Experiments
#### `O(5)`-invariant task
Evaluation on a synthetic `O(5)` invariant regression problem `2T1 → T0` in `d = 5` dimensions given by the function
`f(x1,x2) = sin(||x1||) - ||x2||^3/2 + <x1, x2>/(||x1||*||x2||)`.
See Section 7.1 in [A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups](https://arxiv.org/abs/2104.09459) for more details. 

The neural networks used for this example are multi-layer perceptrons based on the inner products `<x1,x1>, <x1,x2>, <x2,x2>`.

For example, given 3000 training datasets, you can run the following codes with `ntrain=3000`, which produce the test MSE results from  the neural networks based on the scalars `<xi,xj>, i,j=1,2`.
```python
import pytorch_lightning as pl
from train_regression_scalars import makeTrainerScalars

ntrain=3000

trainer_config={
    'log_dir':"./logs/",
    'lr':0.001,
    'num_gpus':1,
    'max_epochs':1000,
    'min_epochs':0,
    'early_stopping':False,
    'early_stopping_patience':3,
    'check_val_every_n_epoch':1,
    'milestones':[120,200,400,600,800],
    'gamma':0.5,
    'n_out_net':1,
    'n_hidden_mlp':500, 
    'n_layers_mlp':5,
    'layer_norm_mlp':False
}

test_mse = makeTrainerScalars(
  dataset=O5Synthetic,  
  ndata=ntrain+2000,
  epoch_samples=4096,
  bs=512,
  trainer_config=trainer_config,
  progress_bar=True
) 

```

#### `O(3)`-equivariant task
This task is to predicting the moment of inertia matrix from `n=5` point masses and positions. 
The inputs `X = {(m_i,x_i)}_{i=1}^5` are of type `5T0+5T1` (5 scalars and vectors) and outputs are of type `T2` (a matrix), both transform under the group. See Section 7.1 in [A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups](https://arxiv.org/abs/2104.09459) for more details. 

For example, given 3000 training datasets, you can run the following codes with `ntrain=3000`, which produce the test MSE results from  the neural networks based on the scalars `<xi,xj>, i,j=1,...,5`, which are designed to be equivariant and permutation invariant.
```python
import pytorch_lightning as pl
from train_regression_scalars import makeTrainerScalars

ntrain=3000

trainer_config={
    'log_dir':"./logs/",
    'lr':0.005,
    'num_gpus':1,
    'max_epochs':1000,
    'min_epochs':0,
    'check_val_every_n_epoch':1,
    'milestones':[50,120,200,300,400],
    'gamma':0.5,
    'n_hidden_mlp':300, 
    'n_layers_mlp':2,
    'layer_norm_mlp':False
}
test_mse = makeTrainerScalars(
  dataset=Inertia,
  ndata=ntrain+2000,
  epoch_samples=16384,
  bs=512,
  trainer_config=trainer_config,
  permutation=True,
  progress_bar=True
)
```

#### Figures to visualize comparison results for the above two examples

<img src="https://github.com/Pamplemousse-Elaina/Comparison_EMLP/blob/b6a77f3cf21a951e06729bd45a83b5b957695e32/docs/notebooks/imgs/data_efficiency_O5Synthetic.png?raw=true" height="250"/> <img src="https://github.com/Pamplemousse-Elaina/Comparison_EMLP/blob/b6a77f3cf21a951e06729bd45a83b5b957695e32/docs/notebooks/imgs/data_efficiency_Inertia.png?raw=true" height="250"/>

The purple line labeled as "scalars" corresponds to our method.

### Modeling dynamical systems with symmetries
We consider the task of learning the dynamics of a double pendulum with springs in 3-dimensional space, analogous to the numerical experiment from [Finzi et al. ICML 2021](https://arxiv.org/abs/2104.09459). The approach consists of learning the Hamiltonian that characterizes the dynamical system. Finzi at al. consider O(2) or SO(2) equivariance, because the behavior in the z-direction is not the same as the behavior in the xy-plane due to gravity. Our model considers the gravity vector to be an input of the hamiltonian and models the Hamiltonian as an O(3)-invariant function: ![formula](https://render.githubusercontent.com/render/math?math=H(q_1,q_2,p_1,p_2,g,k_1,k_2,L_1,L2,m_1,m_2)) an invariant function of vectors indicating the positions of the masses, the momentums, the gravity vector, and the scalars corresponding to the constant of the springs, the natural lengths, and the masses.
     
For the dynamical systems modeling experiments you can use the scripts
 [`experiments/hnn_scalars.py`](https://github.com/Pamplemousse-Elaina/Comparison_EMLP/blob/master/experiments/hnn_scalars.py) to train (equivariant) Hamiltonian Neural Networks using scalars construction.  
```
python experiments/hnn_scalars.py
```

The following table shows geometric mean of rollout errors (relative error) over `T=30s` for the scalar HNNs vs the various EMLP-G symmetric HNNs vs ordinary MLP HNNs. Errorbars are 1 standard deviation computed over 3 trials, with notation `.012(2)` meaning `.012 ± .002`.


Methods | Scalars (ours) | EMLP-O(2) | EMLP-SO(2) | EMLP-D6 | MLP
---------- | ---------- | ---------- | ---------- | ---------- | ----------
HNNs | 0.007(2) | 0.012(2) | 0.015(3) | 0.013(2) | 0.028 
