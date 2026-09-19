import dataclasses
from openpi.training.uf850_config import build_configs as base_configs

def build_configs():
    base = base_configs()[0]
    return [dataclasses.replace(
        base,
        name='pi05_uf850_task3_lemon_goblet',
        data=dataclasses.replace(base.data, repo_id='local/uf850_task3_lemon_goblet_lerobot_v001'),
        keep_period=None,
        log_interval=10,
    )]
