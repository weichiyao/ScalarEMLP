# Scalar-based multi-layer perceptron models

## Authors:
*ScalarEMLP* contains the codes to implement the idea of enforcing the equivariance in deep learning using the scalar method based on the NeurIPS paper (to appear) [**Scalars are universal: Equivariant machine learning, structured like classical physics**](https://arxiv.org/abs/2106.06610); in particular, for learning the system dynamics, for example, given a double pendulum with springs, from NeurIPS 2021 workshop paper on Machine Learning and the Physical Sciences (to appear) [**A simple equivariant machine learning method for dynamics based on scalars**](https://arxiv.org/abs/2110.03761).

Some contributor names appear to the right to this github page because we imported their codes from their public github repository [equivariant-MLP](https://github.com/mfinzi/equivariant-MLP.git). 

## Introduction
Scalar products and scalar contractions of the scalar, vector, and tensor inputs are powerful. Deep learning models using these scalars respect the gauge symmetries—or coordinate freedom—of physical law.

This repository provides codes to construct invariant and equivariant neural networks on the same datasets used in ICML2021 paper [A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups](https://arxiv.org/abs/2104.09459) 

To compare the simulation results with those using *EMLP* proposed in [A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups](https://arxiv.org/abs/2104.09459), this repository imported codes from [equivariant-MLP](https://github.com/mfinzi/equivariant-MLP.git). 


## Experimental Results from Paper
To run the scripts you will need to clone the repo and install it locally which you can do with
```
git clone https://github.com/weichiyao/ScalarEMLP.git
cd ScalarEMLP
pip install -e .
```

### Modeling dynamical systems with symmetries
We consider the task of learning the dynamics of a double pendulum with springs in 3-dimensional space, analogous to the numerical experiment from [Finzi et al. ICML 2021](https://arxiv.org/abs/2104.09459). The approach consists of learning the Hamiltonian that characterizes the dynamical system. Finzi at al. consider O(2) or SO(2) equivariance, because the behavior in the z-direction is not the same as the behavior in the xy-plane due to gravity. Our model considers the gravity vector to be an input of the hamiltonian and models the Hamiltonian as an O(3)-invariant function: ![formula](https://render.githubusercontent.com/render/math?math=H(q_1,q_2,p_1,p_2,g,k_1,k_2,L_1,L2,m_1,m_2)) an invariant function of vectors indicating the positions of the masses, the momentums, the gravity vector, and the scalars corresponding to the constant of the springs, the natural lengths, and the masses.
     
For the dynamical systems modeling experiments you can use the scripts [`experiments/hnn_scalars.py`](experiments/hnn_scalars.py) 
```
python experiments/hnn_scalars.py
```
to train dimensionless Scalar-Based EMLP Hamiltonian Neural Networks.  

#### Figures to visualize scalar-based results for the springy dynamic system
<img src=https://github.com/weichiyao/ScalarEMLP/blob/9bad3323e50b652f07ce314150f3f4f423ebefe8/docs/notebooks/imgs/phase_springy_dynamic_system.png height="400" width="600"/>

The above figure shows the ground truth and predictions of mass 1 (top) and 2 (bottom) in the phase space w.r.t. each dimension. HNNs exhibits more accurate predictions for longer time scales.

