# Attribution

The numerical modules and execution components are drawn from the project implementation developed in an OpenPI-based source tree. That source tree's Apache License 2.0 is retained in `LICENSE`. Distribution changes are package-import relocation, selection of the paper configuration, and extraction of the model-response/feedback contracts. Numerical kernels are not rewritten.

OpenPI: https://github.com/Physical-Intelligence/openpi (Apache-2.0).

Method references:
- π0.5: https://arxiv.org/abs/2504.16054
- Real-Time Execution of Action Chunking Flow Policies (RTC): https://arxiv.org/abs/2506.07339
- Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT / temporal ensembling): https://arxiv.org/abs/2304.13705

NumPy and Numba are external dependencies, not vendored source. Robot SDKs, model weights and simulator code are not included.
