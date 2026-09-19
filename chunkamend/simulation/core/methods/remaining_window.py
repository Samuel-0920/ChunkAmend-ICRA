"""Real remaining-window adapter; no predicted or unexecuted row enters training."""
from contextlib import ExitStack
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import numpy as np
from chunkamend.simulation.core.methods import numeric_dispatch, exact_cycle, box_certificate, compiled_bridge, compiled_projection
from chunkamend.simulation.core.shared.feedback_window import validate_feedback_window, validate_memory_context
from chunkamend.simulation.core.methods.m1_types import valid_remaining_shape
from chunkamend.simulation.core.methods.rap_cover import RapCoverCore
from chunkamend.simulation.core.methods.rap_response import fit_completed_translation_response
from chunkamend.simulation.core.methods.rap_response_6d import fit_completed_six_d_response, Q6D_ACTION_REPRESENTATION
from chunkamend.simulation.core.methods.rap_response_6d_active import ActiveResponseState
from chunkamend.simulation.core.methods.rotation_boundary import RotationBoundaryCorrector, ENDPOINT_FRACTIONS, minimum_jerk_weights, ROT_REF

def clean(value):
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,Enum): return value.value
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,float): return value if np.isfinite(value) else None
    if is_dataclass(value) and not isinstance(value,type):
        return {field.name:clean(getattr(value,field.name)) for field in fields(value)}
    return value

def rotate_window(source, previous_rotation, rotation_arm, upstream_reason='SELECTED'):
    source=np.asarray(source)
    if not valid_remaining_shape(source,7) or not np.issubdtype(source.dtype,np.floating) or not np.isfinite(source).all():
        raise ValueError('Invalid real remaining controller window')
    if rotation_arm != ROT_REF and rotation_arm not in ENDPOINT_FRACTIONS: raise ValueError('Unknown rotation arm')
    result=source.copy()
    if previous_rotation is None: return result,'EPISODE_START_NO_HISTORY'
    previous_rotation=np.asarray(previous_rotation)
    if previous_rotation.shape!=(3,) or not np.isfinite(previous_rotation).all(): raise ValueError('Invalid actual boundary rotation')
    if rotation_arm==ROT_REF: return result,'ROT_REF'
    if upstream_reason!='SELECTED': return result,'UPSTREAM_EXACT_RAW_FALLBACK'
    prefix=source[:5,3:6]
    if np.max(np.abs(prefix))>1 or np.max(np.abs(previous_rotation))>1: return result,'CONTROLLER_INPUT_BOUND_FALLBACK'
    try:
        corrected=np.stack([RotationBoundaryCorrector._interpolate(prefix[i],previous_rotation,float(w))
            for i,w in enumerate(minimum_jerk_weights(ENDPOINT_FRACTIONS[rotation_arm]))]).astype(source.dtype,copy=False)
    except (FloatingPointError,ValueError): return result,'SO3_NUMERICAL_FALLBACK'
    if not np.isfinite(corrected).all() or np.max(np.abs(corrected))>1: return result,'FINAL_CONTROLLER_BOUND_FALLBACK'
    result[:5,3:6]=corrected
    return result,'APPLIED'

def core_decision(cfg,raw,history,positions,response,*,ranking_correction_penalty=None,reference_targets=None,reference_rho=1.0,displacement=None,terminal_veto=True):
    if displacement is None: displacement=np.diff(positions,axis=0)
    return RapCoverCore(cfg.core,method_identity=cfg.method_identity,
        objective_count_mode=cfg.objective_count_mode,selection_mode=cfg.selection_mode,
        candidate_ranking_mode=cfg.candidate_ranking_mode,reversal_space=cfg.reversal_space,
        candidate_ranking_correction_penalty=ranking_correction_penalty,
        uncertainty_scale=cfg.uncertainty_scale,minimum_prequential_predictions=cfg.minimum_prequential_predictions,
        candidate_evaluation_policy=cfg.candidate_evaluation_policy,
        preprojection_objective_bound=cfg.preprojection_objective_bound,
        projector_context_cache_mode=cfg.projector_context_cache_mode,
        terminal_constraint_mode=cfg.terminal_constraint_mode,terminal_projection_role=cfg.terminal_projection_role,terminal_veto=terminal_veto
    ).apply(raw,stride=5,candidate_specs=list(cfg.candidate_specs),completed_actions=history,response=response,
        reference_targets=reference_targets,reference_rho=reference_rho,
        observed_displacements=displacement[-2:] if len(displacement)>=2 else None,
        risk_eligible=True,ablate_risk_trust_eligibility=True,
        reversal_context=dict(executed_action_history=history,completed_tcp_history=positions,
            direction_window=cfg.reversal.direction_window,reversal_cos_threshold=cfg.reversal.reversal_cos_threshold,
            min_consistent_steps=cfg.reversal.min_consistent_steps,velocity_epsilon=cfg.reversal.velocity_epsilon))

class WindowCorrector:
    def __init__(self,cfg,q_mode,rotation_arm,q01,q99,*,ranking_correction_penalty=None,memory_response=False,memory_state_ridge=1e-4,terminal_veto=True):
        if type(terminal_veto) is not bool or (not terminal_veto and q_mode!='Q6'):raise ValueError('Terminal veto ablation is Q6-only')
        self.terminal_veto=terminal_veto
        self.cfg,self.q_mode,self.rotation_arm=cfg,q_mode,rotation_arm
        from chunkamend.simulation.core.methods.rap_cover import _validate_ranking_penalty
        _validate_ranking_penalty(ranking_correction_penalty,cfg.candidate_ranking_mode)
        if type(memory_response) is not bool or (memory_response and q_mode!="Q3"): raise ValueError("Memory response is Q3-only")
        if memory_state_ridge not in (1e-4,1e-3) or (not memory_response and memory_state_ridge!=1e-4):raise ValueError('Unfrozen memory state ridge')
        self.memory_response=memory_response
        self.memory_state_ridge=memory_state_ridge
        self.ranking_correction_penalty=ranking_correction_penalty
        self.q01=np.asarray(q01,dtype=np.float64);self.q99=np.asarray(q99,dtype=np.float64)
        assert self.q01.shape==self.q99.shape==(7,) and np.all(self.q99>self.q01)
    def unnormalize(self,actions):
        return (actions+1.)/2.*(self.q99-self.q01+1e-6)+self.q01
    def rotation(self,physical):
        return 2.*(physical[:,3:6].astype(np.float64)-self.q01[3:6])/(self.q99[3:6]-self.q01[3:6]+1e-6)-1.
    def apply(self,raw_normalized,raw_physical,history,controller_history,positions,*,accelerated=True,infeasibility_shortcut=False,compile_bridge=False,compile_projection=True,reference_targets=None,reference_rho=1.0,feedback_window=None):
        raw=np.asarray(raw_normalized);physical=np.asarray(raw_physical)
        history=np.asarray(history);controllers=np.asarray(controller_history);positions=np.asarray(positions)
        if not valid_remaining_shape(raw,7) or physical.shape!=raw.shape: raise ValueError('Remaining raw shape')
        if history.ndim!=2 or history.shape[1]!=7 or controllers.shape!=history.shape or positions.shape!=(len(history)+1,3):
            raise ValueError('Completed feedback alignment')
        if not all(np.isfinite(x).all() for x in (raw,physical,history,controllers,positions)): raise ValueError('Nonfinite input')
        completed_pairs,window_start=(len(history),0) if feedback_window is None else validate_feedback_window(feedback_window,history,controllers,positions)
        expected=self.unnormalize(raw)
        if not np.array_equal(expected,physical): raise ValueError('Output transform mismatch')
        counterfactual,rotation_reason=rotate_window(physical,controllers[-1,3:6] if len(controllers) else None,self.rotation_arm)
        d=np.diff(positions,axis=0)
        with ExitStack() as scopes:
            if compile_bridge:
                scopes.enter_context(compiled_bridge.enabled())
            if compile_bridge and compile_projection: scopes.enter_context(compiled_projection.enabled())
            if infeasibility_shortcut: scopes.enter_context(box_certificate.enabled())
            if accelerated:
                scopes.enter_context(numeric_dispatch.enabled());scopes.enter_context(exact_cycle.enabled())
            six=None
            if self.q_mode=='Q6':
                x=np.concatenate((history[:,:3].astype(np.float64),self.rotation(controllers)),axis=1)
                six=fit_completed_six_d_response(x,d,action_representation=Q6D_ACTION_REPRESENTATION,
                    rotation_ridge=self.cfg.response_rotation_ridge)
                if window_start: six=replace(six,window_start=six.window_start+window_start,window_end=six.window_end+window_start)
                response=ActiveResponseState.adapt_response(six,raw_rotation=self.rotation(physical),selected_rotation=self.rotation(counterfactual))
            elif self.q_mode=='Q3':
                if self.memory_response:
                    from chunkamend.simulation.core.methods.memory_integration import fit_memory_adapter
                    preceding=validate_memory_context(feedback_window)
                    response=fit_memory_adapter(history[:,:3],d,preceding_displacement=preceding,state_ridge=self.memory_state_ridge)
                    if self.memory_state_ridge!=1e-4:
                        response=replace(response,memory_fit_diagnostics=dict(response.memory_fit_diagnostics,action_ridge=1e-4,state_ridge=self.memory_state_ridge))
                else:
                    response=fit_completed_translation_response(history[:,:3],d)
                if window_start: response=replace(response,window_start=response.window_start+window_start,window_end=response.window_end+window_start)
            else: raise ValueError('Unknown response mode')
            decision=core_decision(self.cfg,raw,history,positions,response,ranking_correction_penalty=self.ranking_correction_penalty,reference_targets=reference_targets,reference_rho=reference_rho,displacement=d,terminal_veto=self.terminal_veto)
            acceleration=dict(numeric=numeric_dispatch.implementation_receipt(),cycle=exact_cycle.implementation_receipt(),box_certificate=box_certificate.receipt(),compiled_bridge=compiled_bridge.receipt(),compiled_projection=compiled_projection.receipt())
        selected=decision.chunk.copy()
        output=physical.copy()
        if decision.reason.value=='SELECTED':
            output[:,:3]=self.unnormalize(selected)[:,:3]
            output[:,3:6]=counterfactual[:,3:6]
        assert selected.dtype==raw.dtype and np.array_equal(selected[5:],raw[5:])
        assert np.array_equal(selected[:,3:],raw[:,3:])
        assert np.array_equal(output[:,6:],physical[:,6:]) and np.array_equal(output[5:],physical[5:])
        if decision.reason.value!='SELECTED': assert np.array_equal(output,physical) and np.array_equal(selected,raw)
        response_receipt=response
        if self.memory_response:
            response_receipt={field.name:getattr(response,field.name) for field in fields(response) if field.name!='temporal_response'}
            response_receipt['temporal_response']=(dict(representation='RECONSTRUCT_FROM_MEMORY_FIT_DIAGNOSTICS',horizon=len(raw)) if response.temporal_response is not None else None)
        receipt=clean(dict(reason=decision.reason,real_remaining_horizon=len(raw),completed_pairs=completed_pairs,
            response=response_receipt,six_d_response=six,
            decision=decision,rotation_counterfactual_reason=rotation_reason,
            rotation_applied=decision.reason.value=='SELECTED' and rotation_reason=='APPLIED',
            acceleration=acceleration))
        if not self.memory_response:
            receipt['response'].pop('temporal_response',None)
            receipt['response'].pop('memory_fit_diagnostics',None)
        else:
            receipt['memory_response']=dict(identity='C146_Q3_MEMORY_REMAINING',active=response.memory_fit_diagnostics['available'],
                horizon=len(raw),supervised_pairs=response.sample_count,window_start=response.window_start,window_end=response.window_end)
        if self.ranking_correction_penalty is not None:
            receipt["ranking_correction_penalty"]=float(self.ranking_correction_penalty)
        return dict(actions=output,normalized_physical_actions=selected,receipt=receipt)
