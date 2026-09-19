"""New task data/config binding using the previously verified OpenPI adapter."""
import dataclasses
from pathlib import Path
from openpi.training import config

B = Path(__file__).resolve().parents[1]
REPO_ID = 'local/uf850_fruit_organization_v001'

def get_config(lora=False):
    suffix = '_lora' if lora else ''
    base = config.get_config('pi05_uf850_fruit_organization' + suffix)
    return dataclasses.replace(base,
        name='pi05_uf850_fruit_organization' + suffix,
        data=dataclasses.replace(base.data, repo_id=REPO_ID),
        assets_base_dir=str(B/'training/openpi/assets'),
        checkpoint_base_dir=str(B/'training/openpi/checkpoints'))
