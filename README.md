# Scientific Machine Learning with JAX

Exercises and implementations from the Scientific Machine Learning course
at Johannes Kepler University Linz (Summer Semester 2026).

## About
This repository contains hands-on implementations of core JAX concepts
applied to scientific computing and physics simulation.

## Topics Covered
- **JIT Compilation** — Accelerating numerical code with XLA
- **Automatic Vectorization** — Batched simulations with `vmap`
- **Automatic Differentiation** — Exact gradients with `grad`
- **ODE Solving** — Time-stepping with `lax.scan`
- **Lorenz63 Simulation** — Chaotic systems with differentiable solvers

## Repository Structure
| Notebook | Description |
|---|---|
| `jit_Compilation.ipynb` | JIT compilation and XLA basics |
| `JIT_Control_Flow_and_Logic.ipynb` | Control flow under jit |
| `Automatic_Vectorization.ipynb` | Batched computation with vmap |
| `Automatic_Differentiation.ipynb` | Gradients and higher-order derivatives |
| `exercise_jax_patterns.ipynb` | Core JAX patterns exercises |
| `exercise_lorenz63_simulation.ipynb` | Chaotic ODE simulation |

## Key Implementations
- Differentiable Euler ODE solver compiled with XLA
- Parameter sweeps over 100,000 initial conditions using vmap
- Physics loss gradients for Physics-Informed Neural Networks
- Monte Carlo π estimation parallelized across independent PRNG keys
- Lorenz63 chaotic attractor simulation using lax.scan

## Technologies
![JAX](https://img.shields.io/badge/JAX-0.9.1-purple)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-orange)

## Setup
```bash
git clone https://github.com/lolorikos/sciml-jax.git
cd sciml-jax
pip install "jax[cpu]" jupyter ipykernel
jupyter notebook
```

## Course
**Scientific Machine Learning** — Johannes Kepler University Linz  
Summer Semester 2026
