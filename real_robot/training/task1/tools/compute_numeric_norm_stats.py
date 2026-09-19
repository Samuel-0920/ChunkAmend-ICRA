"""Compute OpenPI stats over ALL frames and H15 chunks without decoding images.
Same six-joint DeltaActions and terminal-repeat chunk semantics as the official loader.
Unlike compute_norm_stats.py's drop_last batch loop, includes the final partial batch.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from openpi.shared import normalize
from openpi import transforms

B=Path(__file__).resolve().parents[1]
D=B/'datasets/local/uf850_real_lemon_to_basket_v001'
H=15

def main():
 (B/'evidence').mkdir(parents=True,exist_ok=True)
 running={k:normalize.RunningStats() for k in ['state','actions']}
 count=0
 for f in sorted((D/'data/chunk-000').glob('*.parquet')):
  d=pd.read_parquet(f);s=np.stack(d['observation.state']).astype(np.float32);a=np.stack(d.action).astype(np.float32)
  idx=np.minimum(np.arange(len(d))[:,None]+np.arange(H)[None,:],len(d)-1)
  # This is the official transform, with one anchor state for the whole chunk.
  chunk=transforms.DeltaActions(transforms.make_bool_mask(6,-1))({'state':s,'actions':a[idx].copy()})['actions']
  running['state'].update(s);running['actions'].update(chunk);count+=len(d)
 result={k:v.get_statistics() for k,v in running.items()}
 for config in ['pi05_uf850_real_lemon_to_basket','pi05_uf850_real_lemon_to_basket_lora']:
  normalize.save(B/'training/openpi/assets'/config/'local/uf850_real_lemon_to_basket_v001',result)
 (B/'evidence/norm_stats_method.json').write_text(json.dumps({'frames':count,'action_horizon':H,'action_chunk_samples':count*H,'method':'Official RunningStats and DeltaActions; all frames, terminal repeat within each episode; no image decoding required for numeric stats','state_dimensions':7,'action_dimensions':7,'all_training':True},indent=2))
 print('Computed stats:',count,'frames;',count*H,'chunk targets')

if __name__=='__main__':main()
