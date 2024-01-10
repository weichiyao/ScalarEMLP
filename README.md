# Dimensionless scalar-based equivariant machine learning models

## Authors:
*Dimensionless-branch ScalarEMLP*  contains the codes to implement the idea of enforcing the equivariance in deep learning (including unit-equivariance) using the scalar method based on the JMLR paper [**Dimensionless machine learning: Imposing exact units equivariance**](https://arxiv.org/pdf/2204.00887.pdf).

Some contributor names appear to the right to this github page because we imported their codes from their public github repository [equivariant-MLP](https://github.com/mfinzi/equivariant-MLP.git). 

## Introduction
Scalar products and scalar contractions of the scalar, vector, and tensor inputs are powerful. Deep learning models using these scalars respect the gauge symmetries—or coordinate freedom—of physical law.

This repository provides codes to construct invariant and equivariant neural networks on the same datasets used in ICML2021 paper [A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups](https://arxiv.org/abs/2104.09459) 

## Experimental Results from Paper
To run the scripts you will need to clone the repo and install it locally which you can do with
```
git clone -b dimensionless https://github.com/weichiyao/ScalarEMLP.git
cd ScalarEMLP
pip install -e .
```

### Modeling dynamical systems with symmetries
We consider the task of learning the dynamics of a double pendulum with springs in 3-dimensional space, analogous to the numerical experiment from [Finzi et al. ICML 2021](https://arxiv.org/abs/2104.09459). The approach consists of learning the Hamiltonian that characterizes the dynamical system. Finzi at al. consider O(2) or SO(2) equivariance, because the behavior in the z-direction is not the same as the behavior in the xy-plane due to gravity. Our model considers the gravity vector to be an input of the hamiltonian and models the Hamiltonian as an O(3)-invariant function: $H(q_1,q_2,p_1,p_2,g,k_1,k_2,L_1,L_2,m_1,m_2)$ an invariant function of vectors indicating the positions of the masses, the momentums, the gravity vector, and the scalars corresponding to the constant of the springs, the natural lengths, and the masses.

Consider the following three experiments with the same training data. 
The test data used in Experiment 1 is generated from the same distribution as the
training dataset. The test data used in Experiment 2 consists of applying a transformation
to the test data in Experiment 1, where each of the input parameters that include a power of
kg in its units ($m_1$, $m_2$, $k_1$, $k_2$, $p_1(0)$ and $p_2(0)$) is scaled by a factor randomly generated
from Unif(3, 7). The test data used in Experiment 3 has the input parameters $m_1$, $m_2$, $k_1$, $k_2$, $L_1$ and $L2_$ generated from 
Unif(1, 5). We use the same training data $N=30000$ for all
three experiments and each test set consists of 500 data points. That is, Experiments 2 and
3 have out-of-distribution test data, relative to their training data.
     
For the dynamical systems modeling experiments, you can run
```
python experiments/hnn_scalars.py
```
to train dimensionless Scalar-Based EMLP Hamiltonian Neural Networks.  

#### Figures to visualize scalar-based results for the springy dynamic system
<img src=https://github.com/weichiyao/ScalarEMLP/blob/f70aa79effeeaedf5d91528e0179d14e21d045e9/docs/imgs/Experiment_1.png height="400" width="600"/>
<img src=https://github.com/weichiyao/ScalarEMLP/blob/f0aae531b01810a7dd75a1b4b41ba1fbef00d326/docs/imgs/Experiment_2.png height="400" width="600"/>


The above figure shows the ground truth and predictions of mass 1 (top) and 2 (bottom) in the phase space
w.r.t. each dimension. Top 6 panels: Results from Experiment 1, where the test
data are generated from the same distribution as those used for training. Here
the dimensional scalar based MLPs exhibit slightly more accurate predictions for
longer time scales. Bottom 6 panels: Results from Experiment 2, where we use
the same test data in Experiment 1 but each with its inputs that have units of
kg randomly scaled by a factor generated from Unif(3, 7). Here the dimensionless
scalar based MLP is able to provide comparable performance to Experiment 1,
while using the dimensional scalars gives much worse predictions

