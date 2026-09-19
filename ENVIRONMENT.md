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
| π0.5 / OpenPI | Task-specific source snapshots, `pyproject.toml` and `uv.lock` under `real_robot/training/` |
| CUDA, LIBERO, robot SDK and camera drivers | External dependencies; device settings are recorded per task |

[`configs/simulation.json`](configs/simulation.json) and [`configs/hardware.json`](configs/hardware.json) describe the experiment's action and timing settings. They are not dependency lock files for a complete model, simulator, or robot deployment. The rates in these configurations are nominal targets, not measured-throughput claims.

Upstream model software and public weights: [Physical Intelligence / OpenPI](https://github.com/Physical-Intelligence/openpi).

## Real-robot scripts

The three training snapshots pin their resolved dependencies in `real_robot/training/task*/training/openpi/uv.lock`; their Python source target is 3.11. LeRobot is pinned to upstream commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`. These training environments are separate from the small CPU-core requirements above.

Collection uses xArm SDK 1.17.3, Quest/WebXR and ADB, OpenCV/V4L2, NumPy, pandas and PyArrow. The original robot firmware was v2.7.1. The collector accepts `UF850_COLLECTION_PYTHON` and `UF850_ADB` overrides; camera identity and saved home are task configuration. Those device-specific dependencies are not installed by importing this repository.
