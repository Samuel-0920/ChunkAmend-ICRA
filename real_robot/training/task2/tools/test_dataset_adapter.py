import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi import transforms
from openpi.policies.uf850_policy import UF850Inputs, UF850Outputs
from training_data_config import get_config
from openpi.shared import normalize

B=Path(__file__).resolve().parents[1]
D=B/'datasets/local/uf850_fruit_organization_v001'
torch.set_num_threads(2)

def main():
 ds=LeRobotDataset('local/uf850_fruit_organization_v001',root=D,
     delta_timestamps={'action':[i/30 for i in range(15)]},video_backend='pyav')
 assert len(ds)==94473 and ds.num_episodes==124
 mapping=json.loads((D/'provenance/source_map.json').read_text())
 assert len({(x['source_session'],x['source_episode']) for x in mapping})==124
 offset=0;checked=0;first=None
 for ei,source in enumerate(mapping):
  d=pd.read_parquet(D/f'data/chunk-000/episode_{ei:06d}.parquet');n=len(d)
  assert np.array_equal(d['index'],np.arange(offset,offset+n))
  for local in [0,n-2,n-1]:
   sample=ds[offset+local]
   expected=np.stack(d.action)[np.minimum(local+np.arange(15),n-1)]
   np.testing.assert_allclose(sample['action'].numpy(),expected,atol=1e-7)
   assert sample['action'].shape==(15,7)
   for role in ['front','wrist']:
    assert sample['observation.images.'+role].shape==(3,480,640)
   state=sample['observation.state'].numpy()
   parsed=UF850Inputs()({'observation/state':state,'actions':sample['action'].numpy(),
    'observation/front_image':sample['observation.images.front'],
    'observation/wrist_image':sample['observation.images.wrist'],'prompt':sample['task']})
   assert not parsed['image_mask']['right_wrist_0_rgb']
   mask=transforms.make_bool_mask(6,-1)
   original=parsed['actions'].copy()
   delta=transforms.DeltaActions(mask)({**parsed,'actions':original.copy()})
   np.testing.assert_allclose(delta['actions'][:,-1],original[:,-1])
   padded=transforms.PadStatesAndActions(32)(delta)
   assert padded['actions'].shape==(15,32) and padded['state'].shape==(32,)
   absolute=transforms.AbsoluteActions(mask)(padded)
   np.testing.assert_allclose(UF850Outputs()(absolute)['actions'],original,atol=1e-6)
   checked+=1
   if first is None:first=parsed
  offset+=n
  if (ei+1)%20==0:print('Validated boundaries/cameras:',ei+1,'episodes',flush=True)
 for name in ['pi05_uf850_fruit_organization','pi05_uf850_fruit_organization_lora']:
  c=get_config(lora=name.endswith("_lora"))
  assert c.model.discrete_state_input and c.model.action_horizon==15
  assert c.batch_size==64 and c.fsdp_devices==4 and c.data.repo_id=='local/uf850_fruit_organization_v001'
  norm=normalize.load(B/'training/openpi/assets'/name/c.data.repo_id)
  for st in norm.values():
   for k in ['mean','std','q01','q99']:assert np.isfinite(getattr(st,k)).all()
  normalized=transforms.Normalize(norm,use_quantiles=True)({'state':first['state'],'actions':np.zeros((15,7),np.float32)})
  assert np.isfinite(normalized['state']).all()
 # Negative cases: nonzero SDK padding and invalid dimensions must never be silently reinterpreted.
 from convert_dataset import map_vector
 bad=np.zeros((2,8));bad[:,6]=.1
 try:map_vector(bad)
 except ValueError:pass
 else:raise AssertionError('Nonzero discarded slot accepted')
 try:UF850Inputs()({'observation/state':np.zeros(8)})
 except ValueError:pass
 else:raise AssertionError('8D state silently accepted')
 report={'episodes':124,'frames':len(ds),'boundary_and_image_samples_checked':checked,
         'actual_pinned_lerobot_loader':True,'cross_episode_chunks_checked':True,
         'delta_absolute_roundtrip':True,'source_ids_unique':True,
         'config_import_and_norm_load':True,'model_weights_loaded':False,'gpu_training_run':False}
 (B/'evidence/loader_adapter_verification.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
