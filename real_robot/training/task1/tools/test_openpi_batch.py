import dataclasses,json
from pathlib import Path
import numpy as np
import torch
from openpi.training import config,data_loader
from openpi import transforms

B=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
c=dataclasses.replace(config.get_config('pi05_uf850_real_lemon_to_basket'),batch_size=2,num_workers=0,fsdp_devices=1)
d=c.data.create(c.assets_dirs,c.model)
loader=data_loader.create_data_loader(c,shuffle=False,num_batches=1)
obs,action=next(iter(loader))
assert action.shape==(2,15,32) and obs.state.shape==(2,32)
assert all(x.shape==(2,224,224,3) for x in obs.images.values())
assert not np.array(obs.image_masks['right_wrist_0_rgb']).any()
assert np.isfinite(np.array(action)).all()
# Exercise real tokenization: changing state alone must change model tokens.
tokenizer=next(t for t in d.model_transforms.inputs if isinstance(t,transforms.TokenizePrompt))
a=tokenizer({'prompt':'pick up the yellow lemon and place it in the basket','state':np.zeros(7,np.float32)})
b=tokenizer({'prompt':'pick up the yellow lemon and place it in the basket','state':np.ones(7,np.float32)*.5})
assert not np.array_equal(a['tokenized_prompt'],b['tokenized_prompt'])
r={'official_openpi_data_loader_batch':True,'action_shape':list(action.shape),'state_shape':list(obs.state.shape),'image_shape':[2,224,224,3],'state_changes_prompt_tokens':True,'actual_model_training':False}
(B/'evidence/openpi_batch_verification.json').write_text(json.dumps(r,indent=2));print(r)
