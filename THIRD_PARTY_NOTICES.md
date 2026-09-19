# Attribution and third-party notices

## Project implementation

ChunkAmend's feedback-based correction, selected configuration, domain-specific adoption logic, and UF850 coordinate/feedback contracts are provided as this project's implementation. The modules were developed in an OpenPI-based source tree. The source tree's Apache License 2.0 is retained in [`LICENSE`](LICENSE).

Release preparation relocated package imports, selected the paper configuration, and extracted model-response and feedback contracts. The numerical kernels were retained from their respective simulation and hardware source implementations.

## Upstream software

[OpenPI](https://github.com/Physical-Intelligence/openpi), by Physical Intelligence, provides the upstream policy framework. Task-specific OpenPI source snapshots are included under `real_robot/training/`, retaining their licenses and dependency lock files. The original public π0.5 base checkpoint remains an upstream resource; the release assets contain this project's UF850 fine-tuned derivatives.

[NumPy](https://numpy.org/), [Numba](https://numba.pydata.org/), and [llvmlite](https://llvmlite.readthedocs.io/) are external dependencies, not vendored source. Their versions are listed in [`ENVIRONMENT.md`](ENVIRONMENT.md). Their own licenses apply to those distributions.

## Method references

| Method or component | Original work | Use in this project |
|---|---|---|
| π0.5 | Physical Intelligence et al., *π0.5: a Vision-Language-Action Model with Open-World Generalization*, 2025. [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) | Underlying action-chunking policy; raw asynchronous execution defines Vanilla |
| RTC | Kevin Black, Manuel Y. Galliker, and Sergey Levine, *Real-Time Execution of Action Chunking Flow Policies*, 2025. [arXiv:2506.07339](https://arxiv.org/abs/2506.07339) | Asynchronous action-inpainting comparator, configured with full 32-dimensional guidance |
| Temporal ensembling | Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) | ACT-style per-step temporal averaging of π0.5 predictions, denoted TE or stepTE |

Comparator configurations and integration choices belong to this project; the underlying baseline methods are attributed to their original authors. Referencing a method does not imply that this repository distributes its complete upstream implementation, model weights, or training data.

## Collection, training and model assets

UF850 observation/action adapters, task registrations, dataset conversion checks, collection/reset/curation integration and task configurations are project additions or adaptations. The training optimizer, model architecture and general training loop originate in OpenPI; the base policy is not a project contribution. The inherited Quest/WebXR collector was adapted for UF850 recording and reset behavior; this release does not claim authorship of the upstream headset, browser or robot SDK.

The xArm SDK and LeRobot remain external dependencies. Model distributions retain [Gemma terms](LICENSE_GEMMA.txt), the [notice](NOTICE.txt) and [model provenance](MODEL_LICENSE.md). `real_robot/SOURCE_FILES.json` records hashes of selected source files and identifies release edits; no dataset payloads are represented by that code inventory.
