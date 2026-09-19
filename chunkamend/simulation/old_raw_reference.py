"""Original RAW provenance and time-aligned unexecuted-tail extraction."""
from dataclasses import dataclass,field
import hashlib
import numpy as np

def _integer(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<0:
        raise ValueError('Invalid '+name)
    return int(value)

def _hash(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

@dataclass(frozen=True)
class RawFrame:
    generation:int
    observation_step:int
    normalization_identity:str
    raw:np.ndarray
    raw_sha256:str=field(init=False)
    def __post_init__(self):
        _integer(self.generation,'generation');_integer(self.observation_step,'observation_step')
        value=np.asarray(self.raw)
        if value.shape!=(10,32) or not np.issubdtype(value.dtype,np.floating) or not np.isfinite(value).all():
            raise ValueError('Invalid original RAW chunk')
        if not isinstance(self.normalization_identity,str) or not self.normalization_identity:
            raise ValueError('Missing normalization identity')
        value=np.ascontiguousarray(value).copy();value.flags.writeable=False
        object.__setattr__(self,'raw',value);object.__setattr__(self,'raw_sha256',_hash(value))

def extract(frame,*,generation,snapshot_step,cursor,length,normalization_identity):
    generation=_integer(generation,'generation');snapshot_step=_integer(snapshot_step,'snapshot_step')
    cursor=_integer(cursor,'cursor');length=_integer(length,'length')
    if cursor>10 or length not in (0,1,2):raise ValueError('Unsupported cursor/reference length')
    receipt=dict(generation=generation,snapshot_step=snapshot_step,cursor=cursor,length=length,
        normalization_identity=normalization_identity,reference_source='ORIGINAL_RAW',source_rows_zero_based=[])
    if length==0:return None,dict(receipt,status='NOT_REQUESTED')
    if frame is None:return None,dict(receipt,status='MISSING_PRIOR_RAW')
    if not isinstance(frame,RawFrame):raise ValueError('Invalid RAW provenance frame')
    if frame.generation!=generation:raise ValueError('RAW generation mismatch')
    if frame.normalization_identity!=normalization_identity:raise ValueError('RAW normalization mismatch')
    if frame.observation_step+cursor!=snapshot_step:raise ValueError('RAW origin/cursor/feedback mismatch')
    if _hash(frame.raw)!=frame.raw_sha256:raise ValueError('Original RAW mutated')
    receipt.update(prior_raw_sha256=frame.raw_sha256,raw_dtype=frame.raw.dtype.str,raw_shape=list(frame.raw.shape),prior_observation_step=frame.observation_step)
    if cursor+length>len(frame.raw):return None,dict(receipt,status='INSUFFICIENT_PRIOR_RAW_TAIL',available_rows=len(frame.raw)-cursor)
    targets=np.ascontiguousarray(frame.raw[cursor:cursor+length,:3],dtype=np.float64).copy();targets.flags.writeable=False
    receipt.update(status='AVAILABLE',source_rows_zero_based=list(range(cursor,cursor+length)),targets_sha256=_hash(targets))
    return targets,receipt


def make_bundle(frame,*,generation,snapshot_step,cursor,length,rho,normalization_identity,episode_identity,request_id,current_observation_step):
    targets,receipt=extract(frame,generation=generation,snapshot_step=snapshot_step,cursor=cursor,length=length,normalization_identity=normalization_identity)
    if frame is None:raise ValueError('Configured executor must retain initial RAW')
    return dict(frame=dict(generation=frame.generation,observation_step=frame.observation_step,
        normalization_identity=frame.normalization_identity,raw=frame.raw,raw_sha256=frame.raw_sha256),
        generation=generation,snapshot_step=snapshot_step,cursor=cursor,length=length,rho=rho,
        normalization_identity=normalization_identity,episode_identity=episode_identity,request_id=request_id,
        current_observation_step=current_observation_step,targets=targets,extraction_receipt=receipt)

def validate_bundle(bundle,*,expected_length,expected_rho,normalization_identity,episode_identity,completed_pairs,remaining_horizon):
    if bundle['length']!=expected_length or isinstance(bundle['length'],bool) or bundle['rho']!=expected_rho or isinstance(bundle['rho'],bool):
        raise ValueError('Reference configuration mismatch')
    if bundle['normalization_identity']!=normalization_identity or bundle['episode_identity']!=episode_identity:
        raise ValueError('Reference context mismatch')
    if bundle['snapshot_step']!=completed_pairs or 10-(completed_pairs-bundle['current_observation_step'])!=remaining_horizon:
        raise ValueError('Reference completed-feedback mismatch')
    f=bundle['frame'];frame=RawFrame(f['generation'],f['observation_step'],f['normalization_identity'],f['raw'])
    if frame.raw_sha256!=f['raw_sha256']:raise ValueError('Transported original RAW hash mismatch')
    _integer(bundle['request_id'],'request_id');_integer(bundle['current_observation_step'],'current_observation_step')
    if frame.generation!=bundle['generation'] or frame.observation_step+bundle['cursor']!=completed_pairs or frame.normalization_identity!=normalization_identity:
        raise ValueError('Transported RAW frame context mismatch')
    targets,receipt=extract(frame,generation=bundle['generation'],snapshot_step=bundle['snapshot_step'],cursor=bundle['cursor'],length=expected_length,normalization_identity=normalization_identity)
    if receipt!=bundle['extraction_receipt']:raise ValueError('Reference extraction receipt mismatch')
    supplied=bundle['targets']
    if targets is None:
        if supplied is not None:raise ValueError('Unexpected unavailable reference targets')
    elif supplied is None or not np.array_equal(targets,np.asarray(supplied)):
        raise ValueError('Reference targets are not original RAW tail')
    return targets,dict(receipt,episode_identity=episode_identity,request_id=bundle['request_id'],
        current_observation_step=bundle['current_observation_step'],rho=float(expected_rho),
        prior_raw_sha256=frame.raw_sha256,prior_observation_step=frame.observation_step,
        targets=None if targets is None else targets.tolist())
