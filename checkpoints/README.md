# UF850 task checkpoints

The [checkpoint release](https://github.com/Samuel-0920/ChunkAmend-RealRobot/releases/tag/uf850-checkpoints-v1) contains three task-specific π0.5 models fine-tuned on the project's UF850 demonstrations. They are distinct from the public upstream π0.5 base checkpoint.

| Task | Prompt | Policy configuration |
|---|---|---|
| task1 | pick up the yellow lemon and place it in the basket | `pi05_uf850_real_lemon_to_basket` |
| task2 | Organize the fruit. | `pi05_uf850_fruit_organization` |
| task3 | Pick up the lemon and place it into the goblet. | `pi05_uf850_task3_lemon_goblet` |

All three are full fine-tuning checkpoints saved at step 9999, with seed 42 and action horizon 15. Each task's four execution methods share its task model; ChunkAmend, RTC and TE are execution/inference mechanisms, not separate task-weight downloads.

## Archive contents

Each model archive contains `9999/params/`, its matching `9999/assets/local/<dataset_id>/norm_stats.json`, and checkpoint metadata where available. The release excludes optimizer state, training logs, recorded observations and all dataset payloads, including dataset archives embedded in the original private backups. Model members are checked byte-for-byte against the locally deployed checkpoint before publication.

The archives are divided into 512 MiB parts. [manifest.json](manifest.json) records exact archive sizes, ordered parts, per-part and full-archive SHA-256 values, and the permitted model-member inventory. The model terms and notices accompany the release assets.

[`tools/download_checkpoint.py`](../tools/download_checkpoint.py) downloads one task, verifies each part and the assembled archive, and copies its model terms beside the archive. For example, `python tools/download_checkpoint.py task2 --output checkpoints/downloads`. It does not extract files or start hardware. The archive's `9999/` directory corresponds to the checkpoint root in that task's [deployment configuration](../real_robot/task_configs/).

## Provenance and terms

The base architecture and pretrained weights come from [Physical Intelligence / OpenPI](https://github.com/Physical-Intelligence/openpi). Task adaptation and fine-tuning are project work. The associated training configuration and source snapshot are under [`real_robot/training/`](../real_robot/training/). The private training datasets are not part of this release.

Model use and redistribution are subject to the retained [model terms](../MODEL_LICENSE.md), [Gemma terms](../LICENSE_GEMMA.txt) and [notice](../NOTICE.txt).
