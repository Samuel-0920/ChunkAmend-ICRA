# Core environment

| Component | Version |
|---|---|
| Python | 3.11.9 |
| NumPy | 1.26.4 |
| Numba | 0.61.2 |
| llvmlite | 0.44.0 |

These versions describe the CPU numerical core. CUDA, the policy model, simulator and robot SDK are outside this source-only package. `configs/` records the policy and execution settings used by the two domains; it is not a complete deployment environment.
