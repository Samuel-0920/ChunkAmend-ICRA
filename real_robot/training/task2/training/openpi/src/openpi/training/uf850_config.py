"""UF850 fruit organization training. All 124 user-selected demonstrations are training data."""
import dataclasses
from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import uf850_policy
from openpi.training import config as c
from openpi.training import optimizer
from openpi.training import weight_loaders

NAME = 'uf850_fruit_organization_v001'
REPO_ID = 'local/' + NAME
CONFIG_NAME = 'pi05_uf850_fruit_organization'

@dataclasses.dataclass(frozen=True)
class UF850DataConfig(c.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        repack = transforms.Group(inputs=[transforms.RepackTransform({
            'observation/front_image': 'observation.images.front',
            'observation/wrist_image': 'observation.images.wrist',
            'observation/state': 'observation.state',
            'actions': 'action', 'prompt': 'prompt'})])
        delta_mask = transforms.make_bool_mask(6, -1)
        data = transforms.Group(inputs=[uf850_policy.UF850Inputs()],
                                outputs=[uf850_policy.UF850Outputs()]).push(
            inputs=[transforms.DeltaActions(delta_mask)],
            outputs=[transforms.AbsoluteActions(delta_mask)])
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack, data_transforms=data,
            model_transforms=c.ModelTransformFactory()(model_config),
            action_sequence_keys=('action',))


def build_configs():
    configs = []
    for lora in [False, True]:
        model = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=15,
            discrete_state_input=True,
            paligemma_variant='gemma_2b_lora' if lora else 'gemma_2b',
            action_expert_variant='gemma_300m_lora' if lora else 'gemma_300m')
        configs.append(c.TrainConfig(
            name=CONFIG_NAME + ('_lora' if lora else ''), model=model,
            data=UF850DataConfig(repo_id=REPO_ID, base_config=c.DataConfig(prompt_from_task=True)),
            batch_size=64, fsdp_devices=4, num_workers=4, seed=42,
            num_train_steps=10_000, save_interval=1000, keep_period=1000,
            lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1000,
                peak_lr=5e-5 if lora else 2e-5, decay_steps=10_000,
                decay_lr=5e-6 if lora else 2e-6),
            optimizer=optimizer.AdamW(clip_gradient_norm=1.0), ema_decay=None,
            freeze_filter=model.get_freeze_filter(),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                'gs://openpi-assets/checkpoints/pi05_base/params'),
            wandb_enabled=False))
    return configs
