"""Lossless video copy + explicit raw UF850 -> LeRobot v2.1 conversion. Offline only."""
import hashlib,json,shutil,os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

B=Path(__file__).resolve().parents[1]
RAW=Path(os.environ.get("UF850_RAW_ROOT", str(B/"raw_captures")))
SESSIONS=['session_20260907_191254_505510','session_20260907_193024_201831']
NAME='uf850_real_lemon_to_basket_v001'
TASK='pick up the yellow lemon and place it in the basket'
FPS=30

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def stats(a):
 a=np.asarray(a,dtype=np.float64)
 return {k:np.atleast_1d(v).tolist() for k,v in {'min':a.min(0),'max':a.max(0),'mean':a.mean(0),'std':a.std(0),'count':np.array([len(a)])}.items()}

def map_vector(a):
 a=np.asarray(list(a),dtype=np.float64)
 if a.ndim!=2 or a.shape[1]!=8 or not np.isfinite(a).all():raise ValueError('Expected finite 8-slot raw vector')
 if not np.all(a[:,6]==0):raise ValueError('SDK padding slot is nonzero; cannot discard')
 return a[:,[0,1,2,3,4,5,7]].astype(np.float32)

def decode_stats(p,n):
 c=cv2.VideoCapture(str(p)); count=0;pixels=[]
 try:
  if abs(c.get(cv2.CAP_PROP_FPS)-FPS)>1e-4:raise ValueError('Wrong video fps')
  while True:
   ok,im=c.read()
   if not ok:break
   if im.shape!=(480,640,3):raise ValueError('Wrong video dimensions')
   # RGB statistics from all frames, spatially sampled at stride 8.
   pixels.append(im[::8,::8,::-1].reshape(-1,3));count+=1
 finally:c.release()
 if count!=n:raise ValueError(f'{p}: decoded {count}, expected {n}')
 a=np.concatenate(pixels).astype(np.float64)/255
 st=stats(a)
 for k in ['min','max','mean','std']:st[k]=np.array(st[k]).reshape(3,1,1).tolist()
 st['count']=[n]
 return st

def main():
 dest=B/'datasets/local'/NAME
 if dest.exists():raise FileExistsError(dest)
 stage=dest.with_name(NAME+'.partial');stage.mkdir(parents=True,exist_ok=False)
 sources=[f for s in SESSIONS for f in sorted((RAW/s).glob('episodes/episode_*/manifest.json'))]
 if len(sources)!=100:raise ValueError(f'Expected frozen 100 episodes, found {len(sources)}')
 features={k:{'dtype':'float32','shape':[7],'names':['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','gripper']} for k in ['observation.state','action']}
 for k in ['timestamp','frame_index','episode_index','index','task_index']:
  features[k]={'dtype':'float32' if k=='timestamp' else 'int64','shape':[1],'names':None}
 for role in ['front','wrist']:
  features['observation.images.'+role]={'dtype':'video','shape':[480,640,3],'names':['height','width','channels'],'info':{'video.height':480,'video.width':640,'video.codec':'mpeg4','video.pix_fmt':'yuv420p','video.fps':30.0,'video.channels':3,'video.is_depth_map':False,'has_audio':False}}
 episodes=[];epstats=[];mapping=[];timing_report=[];offset=0
 for ei,mf in enumerate(sources):
  m=json.loads(mf.read_text());src=mf.parent
  if m['task']!=TASK or m.get('exclude_from_training') or m['termination_reason']!='operator_b_end':raise ValueError('Task/eligibility mismatch')
  hashes={name:digest(src/name) for name in ['samples.parquet','front.mp4','wrist.mp4']}
  if hashes!=m['files']:raise ValueError('Raw hash mismatch')
  d=pd.read_parquet(src/'samples.parquet');n=len(d)
  if n!=m['samples'] or n<15 or not np.array_equal(d.frame_index,np.arange(n)):raise ValueError('Bad frame indices')
  s=map_vector(d['observation.state']);a=map_vector(d['action'])
  if np.any(a[:,-1]<0) or np.any(a[:,-1]>1) or np.any(s[:,-1]<-.01) or np.any(s[:,-1]>1.01):raise ValueError('Gripper range mismatch')
  times=np.array(d.sample_monotonic_s);interval=np.diff(times)
  if np.any(interval<=0) or interval.max()>0.05:raise ValueError('Irregular sampling requires review')
  td=[json.loads(x) for x in d.timing_json]
  ages={role:[] for role in ['front','wrist']};state_ages=[];duplicates={r:0 for r in ages}
  for j,t in enumerate(td):
   for name in ['command','gripper_command']:
    if t[name]['sdk_code']!=0 or t[name]['return_ns']>t['sample_ns']:raise ValueError('Rejected/future command in raw')
   state_ages.append((t['sample_ns']-t['state_read_end_ns'])/1e6)
   for role in ages:
    ages[role].append((t['sample_ns']-t['cameras'][role]['read_end_ns'])/1e6)
    if j and t['cameras'][role]['sequence']==td[j-1]['cameras'][role]['sequence']:duplicates[role]+=1
  if min(state_ages)<0 or max(state_ages)>250 or any(min(v)<0 or max(v)>250 for v in ages.values()):raise ValueError('Stale/future observation')
  arrays={'observation.state':s,'action':a,'timestamp':(np.arange(n)/FPS).astype(np.float32),'frame_index':np.arange(n,dtype=np.int64),'episode_index':np.full(n,ei,dtype=np.int64),'index':np.arange(offset,offset+n,dtype=np.int64),'task_index':np.zeros(n,dtype=np.int64)}
  out=stage/f'data/chunk-000/episode_{ei:06d}.parquet';out.parent.mkdir(parents=True,exist_ok=True)
  pd.DataFrame({k:list(v) if v.ndim==2 else v for k,v in arrays.items()}).to_parquet(out,index=False)
  st={k:stats(v) for k,v in arrays.items()}
  for role in ages:
   key='observation.images.'+role
   st[key]=decode_stats(src/f'{role}.mp4',n)
   vp=stage/f'videos/chunk-000/{key}/episode_{ei:06d}.mp4';vp.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src/f'{role}.mp4',vp)
   if digest(vp)!=hashes[f'{role}.mp4']:raise ValueError('Copied video differs')
  timing_path=stage/f'provenance/timing/episode_{ei:06d}.parquet';timing_path.parent.mkdir(parents=True,exist_ok=True)
  d[['frame_index','sample_monotonic_s','timing_json']].to_parquet(timing_path,index=False)
  mapping.append({'episode_index':ei,'source_session':src.parents[1].name,'source_episode':src.name,'source_path':str(src),'source_manifest_sha256':digest(mf),'source_files_sha256':hashes,'samples':n,'task_success':True,'success_source':'operator_confirmation_in_conversation','training':True})
  episodes.append({'episode_index':ei,'tasks':[TASK],'length':n});epstats.append({'episode_index':ei,'stats':st})
  timing_report.append({'episode_index':ei,'interval_ms_quantiles':np.quantile(interval*1000,[0,.5,.95,1]).tolist(),'elapsed_drift_vs_nominal_ms':float((times[-1]-times[0]-(n-1)/FPS)*1000),'camera_read_age_ms_p95_max':{r:np.quantile(v,[.95,1]).tolist() for r,v in ages.items()},'state_age_ms_p95_max':np.quantile(state_ages,[.95,1]).tolist(),'repeated_camera_sequences':duplicates})
  offset+=n;print(f'{ei+1}/100 converted ({offset} frames)',flush=True)
 info={'codebase_version':'v2.1','robot_type':'UF850','total_episodes':100,'total_frames':offset,'total_tasks':1,'total_videos':200,'total_chunks':1,'chunks_size':1000,'fps':FPS,'splits':{'train':'0:100'},'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet','video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4','features':features}
 write(stage/'meta/info.json',info)
 for name,rows in [('episodes',episodes),('episodes_stats',epstats),('tasks',[{'task_index':0,'task':TASK}])]:
  (stage/f'meta/{name}.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in rows))
 write(stage/'provenance/source_map.json',mapping);write(stage/'provenance/timing_audit.json',timing_report)
 write(stage/'provenance/semantics.json',{'raw_to_effective_indices':[0,1,2,3,4,5,7],'joint_unit':'radian','action':'last SDK-accepted absolute joint targets + commanded gripper pulse/850','state':'SDK joint feedback + measured gripper pulse/850','gripper':'0 closed, 1 nominal open; measured slight overshoot retained','images':'front and wrist; no global camera; copied compressed videos unchanged','timestamp':'frame_index/30 nominal video time; actual host timestamps preserved in provenance/timing; no resampling/interpolation','hardware_synchronized':False,'success_source':'operator confirmed all 100 successful; not automated visual verification','all_episodes_used_for_training':True})
 stage.rename(dest)
 write(B/'evidence/conversion_result.json',{'dataset':str(dest),'episodes':100,'frames':offset,'videos':200,'all_videos_fully_decoded':True,'raw_hashes_verified':True})

if __name__=='__main__':main()
