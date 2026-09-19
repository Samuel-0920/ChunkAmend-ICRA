# ChunkAmend · Real-Robot Deployment

**Correcting Action Chunks with Execution Feedback**

English · [简体中文](README.zh-CN.md)

ChunkAmend is an execution-time correction method for action-chunking policies. It uses completed command–response pairs to estimate a local six-input, three-output displacement model, then constructs a bounded correction to the incoming action prefix. The correction is adopted only at its valid execution boundary. Policy weights remain unchanged.

This repository contains ChunkAmend's algorithm core, simulation and UF850 execution settings, the real-robot data-collection and training scripts, and three fine-tuned task checkpoints. **Datasets are not distributed.**

## Released components

| Component | Contents |
|---|---|
| [Correction factory](chunkamend/core.py) | Constructs the selected Q6 correction configuration for either domain |
| [Simulation](chunkamend/simulation/) | Numerical kernels, asynchronous execution state, and raw-ready publication logic |
| [Hardware](chunkamend/hardware/) | Numerical kernels, reserved-boundary executor, completed-feedback contracts, and joint-action coordinate conversion |
| [Configurations](configs/) | Selected correction parameters and domain-specific execution settings |
| [Implementation notes](docs/IMPLEMENTATION.md) | Correction scope, action conventions, and scheduling semantics |
| [Data collection](real_robot/collection/) | Task-specific Quest/WebXR teleoperation, dual-camera recording, reset and episode curation scripts |
| [Training](real_robot/training/) | Three task configurations, OpenPI source snapshots, preprocessing and training entry points |
| [Task configurations](real_robot/task_configs/) | Prompts, saved reference poses, action conventions and normalization statistics |
| [Checkpoints](checkpoints/README.md) | Three step-9999 task models, release downloads and SHA-256 verification |
| [Environment](ENVIRONMENT.md) | Core dependencies and task-specific training lock files |

The two numerical source trees retain their respective domain implementations. A common factory selects the paper configuration; inference and execution are supplied through callbacks.

## Simulation and hardware

Both configurations overlap policy inference with execution. They use different rules for adopting a new action chunk.

| Setting | Simulation | UF850 hardware |
|---|---|---|
| Action horizon | 10 | 15 |
| Nominal control rate | 20 Hz | 30 Hz |
| Minimum query stride | 5 commands | 5 commands |
| Model / physical action dimensions | 32 / 7 | 32 / 7 |
| Adoption rule | Publish ready raw actions before the next control tick if correction is still pending | Continue the old plan until an explicit reserved boundary |
| Correction timing | Accept a fresh result before raw publication; discard stale or late corrections | Nominal three-slot reservation, bounded by the available horizon; use raw actions if the deadline is missed |

The selected configuration amends a three-command translation prefix and uses a five-command rotational return to the raw plan. The gripper and the suffix outside the writable window are preserved. Exact settings are in [chunkamend.json](configs/chunkamend.json), [simulation.json](configs/simulation.json), and [hardware.json](configs/hardware.json).

## Methods in the comparison

| Method | Configuration used in this project | Attribution |
|---|---|---|
| **ChunkAmend** | Feedback-based action-prefix correction with domain-specific asynchronous adoption | This project |
| **RTC** | Asynchronous action inpainting with full 32-dimensional guidance | [RTC](https://arxiv.org/abs/2506.07339) |
| **TE / stepTE** | Query at every logical step; temporally average predictions for the same target step with weights proportional to `exp(-0.01 i)`, ordered oldest to newest | Temporal ensembling from [ACT](https://arxiv.org/abs/2304.13705) |
| **Vanilla** | Asynchronous execution of raw π0.5 predictions | [π0.5](https://arxiv.org/abs/2504.16054) |

These names identify the comparison configurations. TE uses π0.5 predictions with ACT-style temporal ensembling; it does not replace the policy with an ACT model. The released files provide the numerical and scheduling components listed above; full policy services and evaluation runners are outside this package.

## Release scope

The released checkpoints cover lemon-to-basket, fruit organization, and lemon-to-goblet. They contain inference parameters and the matching normalization statistics; optimizer state is not included. The public π0.5 base model comes from [OpenPI](https://github.com/Physical-Intelligence/openpi); these task checkpoints were fine-tuned on this project's self-collected demonstrations.

See the [real-robot component index](real_robot/README.md) for script roles and source boundaries. Recorded robot addresses, camera bindings and home poses belong to the original setup and require local configuration. Raw demonstrations, dataset archives, recorded images/videos and experimental logs are excluded. This release does not claim a new end-to-end deployment validation on another machine.

## License and attribution

Project source is distributed under the [Apache License 2.0](LICENSE). Checkpoints also carry the applicable [model terms and Gemma notices](MODEL_LICENSE.md). Project contributions, upstream components, and method references are described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
