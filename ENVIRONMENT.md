# Core environment

The released numerical kernels and scheduling components use the following CPU environment:

| Component | Version | Role |
|---|---|---|
| Python | 3.11.9 | Runtime |
| NumPy | 1.26.4 | Numerical arrays and linear algebra |
| Numba | 0.61.2 | Compiled numerical kernels |
| llvmlite | 0.44.0 | Numba's LLVM bindings |

The package pins are recorded in [`requirements.txt`](requirements.txt). These versions were used to check the released modules and numerical equivalence with their source implementations.

## Dependency boundary

| Layer | Status in this repository |
|---|---|
| Correction kernels, scheduling, coordinate/feedback contracts | Included as source |
| NumPy, Numba, llvmlite | External dependencies with fixed versions |
| π0.5 / OpenPI model and inference runtime | External; inference is supplied through callbacks |
| CUDA, LIBERO, robot SDK, cameras, kinematics and device configuration | Part of the surrounding experimental application; not bundled |

[`configs/simulation.json`](configs/simulation.json) and [`configs/hardware.json`](configs/hardware.json) describe the experiment's action and timing settings. They are not dependency lock files for a complete model, simulator, or robot deployment. The rates in these configurations are nominal targets, not measured-throughput claims.

Upstream model software and public weights: [Physical Intelligence / OpenPI](https://github.com/Physical-Intelligence/openpi).
