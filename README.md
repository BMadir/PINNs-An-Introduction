# Physics-Informed Neural Networks: An Introduction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green)


This repository contains several implementations of Physics-Informed Neural Networks (PINNs) using PyTorch.

Implemented methods include:

| Notebook | Reference Method |
|----------|----------|
| ``` PINNs_Classical.ipynb ``` | [[1]](https://www.sciencedirect.com/science/article/pii/S0021999118307125?casa_token=qzMbq7Wr9jEAAAAA:MRoYMppatcGHNw72Y_LutAbEYzhwcwSGnJzPZV94thoPg0nYZwt56UfNHyWmngIpYcyBwoYkiQ98)  |
| ``` PINNs_Galerkin.ipynb ``` | [[2]](https://www.sciencedirect.com/science/article/pii/S0045782520307325?casa_token=-MZ-e3sE-u4AAAAA:yGfrbrNHuOkp62Xzk8zeU4LOSo2XyoZRlpE0r7R7XHGb_ssbupr1Ob10fciODemuT-RsrpQ3s4pf)  |
| ``` PINNs_Ritz.ipynb ``` | [[3]](https://link.springer.com/article/10.1007/s40304-018-0127-z) |

These methods are applied to solve the two-dimensional Poisson problem

$$
-\nabla^2 u = f \qquad \text{in } \Omega = (0,1)^2,
$$

subject to homogeneous Dirichlet boundary conditions

$$
u = 0 \qquad \text{on } \partial\Omega.
$$

Don't hesitate to contribute !


