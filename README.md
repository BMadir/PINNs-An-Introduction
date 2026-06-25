# Physics-Informed Neural Networks: An Introduction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green)


This repository contains several implementations of Physics-Informed Neural Networks (PINNs) using PyTorch.

Implemented methods include:

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| ``` PINNs_Classical.ipynb ```  | Value B  | Value C  |
| ``` PINNs_Classical.ipynb ```  | Value E  | Value F  |

* Classical PINNs
* hp-VPINNs (Variational Physics-Informed Neural Networks)
* Deep Ritz Method

These methods are applied to solve the two-dimensional Poisson problem

$$
-\nabla^2 u = f \qquad \text{in } \Omega = (0,1)^2,
$$

subject to homogeneous Dirichlet boundary conditions

$$
u = 0 \qquad \text{on } \partial\Omega.
$$

The repository provides implementations, notebooks, and examples illustrating the formulation, training, and comparison of these physics-informed approaches.


---

## Repository Structure

```text
.
├── PINNs_Classical.ipynb
├── PINNs_Galerkin.ipynb
├── PINNs_Ritz.ipynb
└── Utilities
    ├── AutoDiff.py
    ├── quadrature.py
    └── sampling.py
    
```


---

## References

* Classical PINNs

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
*Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*
Journal of Computational Physics, 378, 686–707.

* hp-VPINNs

Kharazmi, E., Zhang, Z., & Karniadakis, G. E. (2021).
*hp-VPINNs: Variational physics-informed neural networks with domain decomposition.*
Computer Methods in Applied Mechanics and Engineering, 374, 113547.


* Deep Ritz Method

E, W., & Yu, B. (2018).
*The Deep Ritz Method: A deep learning-based numerical algorithm for solving variational problems.*
Communications in Mathematics and Statistics, 6, 1–12.


