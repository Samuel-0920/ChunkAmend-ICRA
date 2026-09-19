# UF850 real-robot components

This directory records the project's collection, training and task configuration code. It is a component description, not a step-by-step deployment tutorial.

| Task | Description | Collection | Training | Task configuration |
|---|---|---|---|---|
| task1 | Pick up the yellow lemon and place it in the basket | [Scripts](collection/task1/) | [Scripts and source](training/task1/) | [Configuration](task_configs/task1/) |
| task2 | Organize the fruit | [Scripts](collection/task2/) | [Scripts and source](training/task2/) | [Configuration](task_configs/task2/) |
| task3 | Pick up the lemon and place it into the goblet | [Scripts](collection/task3/) | [Scripts and source](training/task3/) | [Configuration](task_configs/task3/) |

## Collection

Each task includes its Quest/WebXR control and recorder source, camera bindings, saved reference pose and episode curation code. `START_COLLECTION.sh` is the explicit hardware-starting entry; `launch.py plan` reports configuration without connecting to the robot. The recorded configuration describes the original setup, not a universally valid robot home pose. Task three includes both the precision-placement collector and its [dynamic-batch collector](collection/task3_dynamic/), including the relocation plan used by that collection script.

Collection produces dual-camera videos and timestamped robot command/state records. Those generated files are excluded from this repository. `UF850_COLLECTION_PYTHON` and `UF850_ADB` replace machine-specific interpreter and ADB locations in the published launchers.

## Preprocessing and training

Each training directory retains its own OpenPI source snapshot under `training/openpi/`, rather than silently combining potentially different task versions. `training/server.sh` is the task training entry; `scripts/train.py` inside the source tree is the OpenPI training loop. Project configuration is registered in `src/openpi/training/uf850_config.py`, with task three additionally using `uf850_task3_config.py`. The UF850 policy adapter maps front/wrist images and seven-dimensional joint/gripper state into the policy interface.

The released task weights use the full fine-tuning configuration: π0.5, action dimension 32, horizon 15, seed 42, batch 64, four FSDP devices and 10,000 training steps. Six joint dimensions use query-relative delta targets; the gripper remains absolute. Optional LoRA configuration in the task-one/two source is not the configuration used for the released full checkpoints.

`tools/` retains the offline conversion, numerical-statistics and validation scripts. The original converters bind to the selected collection batches and enforce their frame, timestamp, camera and payload-integrity checks; they are historical task conversion scripts, not a generic converter for arbitrary datasets. Raw input location is configurable with `UF850_RAW_ROOT`. Historical selection manifests and data payloads must be supplied separately if those exact conversion checks are used. Training consumes a local LeRobot dataset, with `HF_LEROBOT_HOME` identifying its root. No script automatically downloads the private datasets.

## Deployment components and weights

The hardware correction executor and coordinate contracts are in [`chunkamend/hardware/`](../chunkamend/hardware/). Task-specific prompt, action, normalization and reference-home settings are in `task_configs/`; paths in `deployment.json` are relative to the repository root. These files describe the task binding and still require the surrounding hardware/model application to connect callbacks and device I/O.

[Checkpoint assets](../checkpoints/README.md) contain the fine-tuned inference parameters and matching statistics. They are sufficient model assets for inference with the matching policy configuration; they do not include optimizer state for exact training resumption. Raw demonstrations, dataset archives, recorded observations and experiment logs are excluded.

The upstream OpenPI framework and public π0.5 base weights are distinguished from this project's UF850 adaptation and task fine-tuning in [the attribution file](../THIRD_PARTY_NOTICES.md). No new robot execution or full training run is claimed by this source publication.
