# Physics-Informed Neural Networks (PINNs): An Introduction

This repository contains several implementations of Physics-Informed Neural Networks (PINNs) using PyTorch for the numerical solution of partial differential equations (PDEs).

Implemented methods include:

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
---

## References

### Classical PINNs

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
*Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*
Journal of Computational Physics, 378, 686–707.

<!--
```bibtex
@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, M. and Perdikaris, P. and Karniadakis, G. E.},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019}
}
```
-->

### hp-VPINNs

Kharazmi, E., Zhang, Z., & Karniadakis, G. E. (2021).
*hp-VPINNs: Variational physics-informed neural networks with domain decomposition.*
Computer Methods in Applied Mechanics and Engineering, 374, 113547.

<!--
```bibtex
@article{kharazmi2021hpvpinns,
  title={hp-VPINNs: Variational physics-informed neural networks with domain decomposition},
  author={Kharazmi, E. and Zhang, Z. and Karniadakis, G. E.},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={374},
  pages={113547},
  year={2021}
}
```
-->

### Deep Ritz Method

E, W., & Yu, B. (2018).
*The Deep Ritz Method: A deep learning-based numerical algorithm for solving variational problems.*
Communications in Mathematics and Statistics, 6, 1–12.

<!--
```bibtex
@article{e2018deepritz,
  title={The Deep Ritz Method: A deep learning-based numerical algorithm for solving variational problems},
  author={E, Weinan and Yu, Bing},
  journal={Communications in Mathematics and Statistics},
  volume={6},
  number={1},
  pages={1--12},
  year={2018}
}
```
-->

