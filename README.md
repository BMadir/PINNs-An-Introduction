# Physics-Informed Neural Networks: An Introduction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green)


This repository contains several implementations of Physics-Informed Neural Networks (PINNs) using PyTorch.

Implemented methods include:

| Notebook | Referance Method |
|----------|----------|
| ``` PINNs_Classical.ipynb ``` | [[1]](https://www.sciencedirect.com/science/article/pii/S0021999118307125?casa_token=qzMbq7Wr9jEAAAAA:MRoYMppatcGHNw72Y_LutAbEYzhwcwSGnJzPZV94thoPg0nYZwt56UfNHyWmngIpYcyBwoYkiQ98)  |
| ``` PINNs_Galerkin.ipynb ``` | [[2]](https://www.sciencedirect.com/science/article/pii/S0045782520307325?casa_token=-MZ-e3sE-u4AAAAA:yGfrbrNHuOkp62Xzk8zeU4LOSo2XyoZRlpE0r7R7XHGb_ssbupr1Ob10fciODemuT-RsrpQ3s4pf)  |
| ``` PINNs_Ritz.ipynb ``` | [[3]](https://link.springer.com/article/10.1007/s40304-018-0127-z) |

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


