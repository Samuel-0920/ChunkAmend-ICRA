# ChunkAmend — ICRA Paper

Core implementation for **ChunkAmend: Correcting Action Chunks with Execution Feedback**.

ChunkAmend uses completed command–response pairs to fit a local six-input, three-output displacement model, constructs a bounded amendment to an incoming action prefix, and accepts it at its valid adoption boundary. It operates outside the policy without changing its weights.

| Path | Contents |
|---|---|
| `chunkamend/core.py` | Factory for the paper's Q6 correction configuration |
| `chunkamend/simulation/core/` | Simulation numerical kernels for H10 action windows |
| `chunkamend/simulation/reference_executor.py` | Asynchronous raw-ready publication and stale-correction rejection |
| `chunkamend/hardware/core/` | Hardware numerical kernels for H15 action windows |
| `chunkamend/hardware/executor.py` | Reserved-boundary correction adoption while the previous plan executes |
| `chunkamend/hardware/contracts.py`, `coordinates.py` | Completed-feedback, model-output and joint-action coordinate contracts |
| `configs/` | Selected correction parameters and domain-specific execution settings |
| `ENVIRONMENT.md`, `requirements.txt` | Core dependency versions |
| `LICENSE`, `THIRD_PARTY_NOTICES.md` | License and source/method attribution |

## Execution settings

Simulation uses a ten-action horizon at 20 Hz and publishes raw predictions before the next control tick when correction is not ready. Hardware uses a fifteen-action horizon, a target rate of 30 Hz and a nominal three-slot reservation bounded by the current execution window. Both use a minimum five-command stride. The writable window contains three amended translations and a five-command rotational return; gripper and suffix are preserved.

## Comparators

| Method | Paper configuration | Reference |
|---|---|---|
| Vanilla | π0.5 with asynchronous raw-action execution and no amendment | [π0.5](https://arxiv.org/abs/2504.16054) |
| RTC | Asynchronous action inpainting with full 32-dimensional guidance | [RTC](https://arxiv.org/abs/2506.07339) |
| stepTE | A query per logical step, temporal averaging with weights proportional to exp(-0.01 i), oldest to newest | [ACT](https://arxiv.org/abs/2304.13705) |

This is a core-algorithm source release, not an end-to-end training or robot deployment package.
