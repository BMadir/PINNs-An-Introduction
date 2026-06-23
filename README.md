# Physics-Informed Neural Networks (PINNs) and Variational Methods

This repository contains implementations and experiments related to Physics-Informed Neural Networks (PINNs) and classical variational methods, including:

- Classical PINNs
- Galerkin methods
- Ritz methods
- Fourier-feature neural networks
- Adaptive sampling strategies
- Custom training framework for PDE problems

The goal is to compare numerical and learning-based approaches for solving partial differential equations (PDEs).

---

## Overview

We focus on solving PDEs using different formulations:

### 1. Physics-Informed Neural Networks (PINNs)
Neural networks trained by minimizing PDE residuals, boundary conditions, and initial conditions.

### 2. Variational Methods
- Ritz method: minimization of an energy functional
- Galerkin method: projection of residuals onto test function space

These classical methods are compared with PINNs in terms of accuracy, stability, convergence, and computational cost.

---

## Repository Structure

- notebooks/
  - classical_pinns.ipynb
  - galerkin_method.ipynb
  - ritz_method.ipynb

- models/
- sampling/
- training/
- autodiff/
- utils/

---

## Features


## Notebooks

- Classical PINNs: standard formulation
- Galerkin method: projection-based approach
- Ritz method: variational energy minimization


## Author

Bahae-Eddine Madir
PhD in Applied Mathematics / Scientific Machine Learning
