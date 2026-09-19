"""C137: publish ready RAW before the next tick if correction is still running."""
import copy,threading,time
import numpy as np
from .rtc_state_portable import RtcAsyncState
from .old_raw_reference import RawFrame,make_bundle
from .core.shared.feedback_window import snapshot_feedback,completed_position

class CommonExecutor:
    def __init__(self,initial,infer,observation,position,*,correct=None,synchronous=False,reference_config=None):
        self.state=RtcAsyncState(initial['normalized_actions'],d_init=2)
        self.physical=np.asarray(initial['actions']).copy()
        assert self.physical.shape==(10,7) and np.isfinite(self.physical).all()
        self.infer,self.correct,self.synchronous=infer,correct,synchronous
        self.observation=copy.deepcopy(observation)
        self.steps=0;self.events=[];self.history=[];self.controller_history=[];self.positions=[completed_position(np.asarray(position,dtype=np.float64))]
        self.condition=threading.Condition();self.closed=False;self.error=None
        self.worker=None
        self.pending_raw=None
        self.reference_config=copy.deepcopy(reference_config)
        if reference_config is not None:
            if synchronous or correct is None:raise ValueError('Reference requires asynchronous corrector')
            if reference_config['length'] not in (0,1,2) or isinstance(reference_config['length'],bool) or reference_config['rho']!=1.0:
                raise ValueError('Unfrozen reference configuration')
        self.raw_frame=None if reference_config is None else RawFrame(0,0,reference_config['normalization_identity'],initial['normalized_actions'])
        if not synchronous:
            self.worker=threading.Thread(target=self._worker,name='policy-inference',daemon=True);self.worker.start()
    def _begin(self):
        request=self.state.begin_inference()
        assert int(request.observation[0])==self.steps
        observed_step=self.steps;obs=copy.deepcopy(self.observation)
        started=time.monotonic_ns()
        self.events.append(dict(event='inference_start',request_id=request.request_id,generation=request.generation,
            observation_step=observed_step,s=request.s,d_hat=request.d_hat,started_ns=started,previous_prior=request.previous_prior.tolist()))
        return request,obs,observed_step,started
    def _infer_adopt(self,item):
        if self.correct is not None and not self.synchronous:
            return self._infer_adopt_nonblocking(item)
        request,obs,observed_step,started=item
        result=self.infer(obs,request,observed_step) # expensive inference outside controller lock
        received=time.monotonic_ns()
        with self.condition:
            locked=time.monotonic_ns()
            if self.closed:
                self.events.append(dict(event='terminal_response_discarded',request_id=request.request_id,received_ns=received))
                return
            delay=self.steps-observed_step
            assert delay==self.state.cursor-request.s
            normalized=np.asarray(result['normalized_actions']).copy()
            physical=np.asarray(result['actions']).copy()
            if normalized.shape!=(10,32) or physical.shape!=(10,7) or not np.isfinite(physical).all():raise ValueError('Bad inference output')
            correction=None;correction_ms=0.
            if self.correct is not None:
                if 10-delay<5:raise RuntimeError('SHORT_REMAINING_HORIZON')
                t=time.monotonic_ns()
                correction=self.correct(normalized[delay:,:7].copy(),physical[delay:].copy(),
                    np.asarray(self.history).reshape(-1,7),np.asarray(self.controller_history).reshape(-1,7),np.asarray(self.positions))
                correction_ms=(time.monotonic_ns()-t)/1e6
                normalized[delay:,:7]=correction['normalized_physical_actions']
                physical[delay:]=correction['actions']
            swap=self.state.complete_inference(request.request_id,normalized)
            assert swap.d_observed==delay and swap.next_action_index==delay
            self.physical=physical
            published=time.monotonic_ns()
            self.events.append(dict(event='swap',**vars(swap),observation_step=observed_step,adoption_step=self.steps,
                received_ns=received,lock_acquired_ns=locked,published_ns=published,worker_envelope_ms=(received-started)/1e6,
                adoption_lock_ms=(published-locked)/1e6,publication_lock_wait_ms=(locked-received)/1e6,
                correction_rpc_ms=correction_ms,correction=correction,normalized_actions=normalized.tolist(),
                physical_actions=physical.tolist(),sampler=result.get('diagnostics',{})))
            self.condition.notify_all()

    def _infer_adopt_nonblocking(self,item):
        request,obs,observed_step,started=item
        result=self.infer(obs,request,observed_step)
        received=time.monotonic_ns()
        raw_normalized=np.asarray(result['normalized_actions']).copy()
        raw_physical=np.asarray(result['actions']).copy()
        if (raw_normalized.shape!=(10,32) or raw_physical.shape!=(10,7)
                or not all(np.issubdtype(x.dtype,np.floating) and np.isfinite(x).all()
                    for x in (raw_normalized,raw_physical))):
            raise ValueError('Bad inference output')
        next_raw_frame=None if self.reference_config is None else RawFrame(request.generation+1,observed_step,self.reference_config['normalization_identity'],raw_normalized)
        snapshot_requested=time.monotonic_ns()
        with self.condition:
            snapshot_locked=time.monotonic_ns()
            if self.closed:
                self.events.append(dict(event='terminal_response_discarded',request_id=request.request_id,received_ns=received))
                return
            self._require_current_request(request,observed_step)
            snapshot=(self.state.generation,self.steps,self.state.cursor)
            delay=self.steps-observed_step
            if 10-delay<5:raise RuntimeError('SHORT_REMAINING_HORIZON')
            # Only already-completed feedback is copied. The callback cannot mutate live history.
            feedback_window=None
            if self.reference_config is None:
                feedback=(np.asarray(self.history).reshape(-1,7).copy(),
                    np.asarray(self.controller_history).reshape(-1,7).copy(),np.asarray(self.positions).copy())
            else:
                feedback,feedback_window=snapshot_feedback(self.history,self.controller_history,self.positions,self.steps,include_memory_context=self.reference_config.get("memory_context",False))
            inputs=(raw_normalized[delay:,:7].copy(),raw_physical[delay:].copy(),*feedback)
            for x in inputs:x.flags.writeable=False
            reference_bundle=None
            if self.reference_config is not None:
                reference_bundle=make_bundle(self.raw_frame,generation=snapshot[0],snapshot_step=snapshot[1],cursor=snapshot[2],
                    length=self.reference_config['length'],rho=self.reference_config['rho'],
                    normalization_identity=self.reference_config['normalization_identity'],episode_identity=self.reference_config['episode_identity'],
                    request_id=request.request_id,current_observation_step=observed_step)
            snapshot_finished=time.monotonic_ns()
            # The worker keeps this token after controller publication, so its late
            # response cannot complete the same request a second time.
            raw_normalized.flags.writeable=False;raw_physical.flags.writeable=False
            pending=dict(request=request,observed_step=observed_step,started=started,
                received=received,normalized=raw_normalized,physical=raw_physical,
                sampler=copy.deepcopy(result.get('diagnostics',{})),snapshot_step=snapshot[1],published=False,new_raw_frame=next_raw_frame)
            if self.pending_raw is not None:raise RuntimeError('DUPLICATE_PENDING_RAW')
            self.pending_raw=pending
        correction_started=time.monotonic_ns()
        correction=self.correct(*inputs) if reference_bundle is None else self.correct(*inputs,reference_bundle=reference_bundle,feedback_window=feedback_window) # No controller lock spans RPC.
        correction_finished=time.monotonic_ns()
        selected=np.asarray(correction['normalized_physical_actions'])
        output=np.asarray(correction['actions'])
        if any(value.shape!=source.shape or value.dtype!=source.dtype or not np.isfinite(value).all()
               for value,source in zip((selected,output),inputs[:2])):
            raise ValueError('Correction must preserve exact remaining shape and dtype')
        normalized=raw_normalized.copy();physical=raw_physical.copy()
        normalized[delay:,:7]=selected;physical[delay:]=output
        publication_requested=time.monotonic_ns()
        with self.condition:
            locked=time.monotonic_ns()
            if pending['published']:
                self.events.append(dict(event='late_correction_discarded',request_id=request.request_id,
                    old_generation=request.generation,current_generation=self.state.generation,
                    snapshot_step=snapshot[1],final_step=self.steps,closed=self.closed,discarded_ns=locked,
                    correction_started_ns=correction_started,correction_finished_ns=correction_finished,
                    correction_rpc_ms=(correction_finished-correction_started)/1e6,
                    discarded_correction=correction,publication_status='RAW_ALREADY_PUBLISHED'))
                self.condition.notify_all()
                return
            if self.closed:
                self.pending_raw=None
                self.events.append(dict(event='terminal_response_discarded',request_id=request.request_id,
                    received_ns=received,correction_rpc_ms=(correction_finished-correction_started)/1e6,
                    discarded_correction=correction,snapshot_step=snapshot[1],final_step=self.steps))
                return
            self._require_current_request(request,observed_step)
            fresh=snapshot==(self.state.generation,self.steps,self.state.cursor)
            delay=self.steps-observed_step
            if 10-delay<5:raise RuntimeError('SHORT_REMAINING_HORIZON')
            if not fresh:
                # One bounded attempt. Do not run the old-boundary correction at a new boundary.
                normalized=raw_normalized;physical=raw_physical
            swap=self.state.complete_inference(request.request_id,normalized)
            assert swap.d_observed==delay and swap.next_action_index==delay
            self.physical=physical
            self.raw_frame=pending['new_raw_frame']
            published=time.monotonic_ns()
            event=dict(event='swap',**vars(swap),observation_step=observed_step,adoption_step=self.steps,
                received_ns=received,lock_acquired_ns=locked,published_ns=published,worker_envelope_ms=(received-started)/1e6,
                adoption_lock_ms=(published-locked)/1e6,publication_lock_wait_ms=(locked-publication_requested)/1e6,
                snapshot_lock_ms=(snapshot_finished-snapshot_locked)/1e6,
                snapshot_lock_wait_ms=(snapshot_locked-snapshot_requested)/1e6,
                correction_started_ns=correction_started,correction_finished_ns=correction_finished,
                correction_rpc_ms=(correction_finished-correction_started)/1e6,
                correction_attempted=True,correction_result_fresh=fresh,snapshot_step=snapshot[1],
                publication_status='FRESH_CORRECTION_RESULT' if fresh else 'STALE_CORRECTION_RAW_FALLBACK',
                correction=correction if fresh else None,discarded_correction=None if fresh else correction,
                normalized_actions=normalized.tolist(),physical_actions=physical.tolist(),sampler=result.get('diagnostics',{}))
            self.pending_raw=None
            self.events.append(event);self.condition.notify_all()

    def _publish_pending_raw(self):
        # Called under the controller condition, before issuing the next action.
        p=self.pending_raw
        if p is None:return
        locked=time.monotonic_ns()
        request=p['request'];self._require_current_request(request,p['observed_step'])
        delay=self.steps-p['observed_step']
        swap=self.state.complete_inference(request.request_id,p['normalized'])
        assert swap.d_observed==delay and swap.next_action_index==delay
        self.physical=p['physical'].copy()
        self.raw_frame=p['new_raw_frame']
        p['published']=True;self.pending_raw=None
        published=time.monotonic_ns()
        self.events.append(dict(event='swap',**vars(swap),observation_step=p['observed_step'],
            adoption_step=self.steps,received_ns=p['received'],lock_acquired_ns=locked,published_ns=published,
            worker_envelope_ms=(p['received']-p['started'])/1e6,adoption_lock_ms=(published-locked)/1e6,
            correction_rpc_ms=None,correction_attempted=True,correction_result_fresh=False,
            correction_pending=True,snapshot_step=p['snapshot_step'],
            publication_status='RAW_READY_BEFORE_CONTROL_TICK',correction=None,discarded_correction=None,
            normalized_actions=p['normalized'].tolist(),physical_actions=p['physical'].tolist(),sampler=p['sampler']))
        self.condition.notify_all()

    def _require_current_request(self,request,observed_step):
        if self.state.inflight is not request or self.state.generation!=request.generation:
            raise RuntimeError('STALE_INFERENCE_REQUEST')
        if self.state.stop_reason is not None:raise RuntimeError(self.state.stop_reason.value)
        if self.steps-observed_step!=self.state.cursor-request.s:
            raise RuntimeError('CONTROLLER_FEEDBACK_CURSOR_MISMATCH')

    def step(self,execute):
        with self.condition:
            if self.closed:raise RuntimeError('executor closed')
            if self.error is not None:raise RuntimeError('inference worker failed') from self.error
            self._publish_pending_raw()
            issued=time.monotonic_ns()
            tick=self.state.controller_tick(np.array([self.steps+1],dtype=np.int64))
            action=self.physical[tick.action_index].copy()
            before=self.positions[-1].copy()
            obs,terminal,position=execute(action)
            position=completed_position(np.asarray(position,dtype=np.float64))
            self.steps+=1
            self.history.append(tick.action[:7].copy());self.controller_history.append(action.copy());self.positions.append(np.asarray(position).copy())
            self.observation=copy.deepcopy(obs)
            event=dict(event='action',step=self.steps,generation=tick.generation,action_index=tick.action_index,
                issued_ns=issued,completed_ns=time.monotonic_ns(),action=action.tolist(),
                normalized_physical_action=tick.action[:7].tolist(),pre_tcp=before.tolist(),post_tcp=np.asarray(position).tolist(),
                inference_inflight=self.state.inflight is not None)
            self.events.append(event)
            if terminal:self.closed=True
            self.condition.notify_all()
            item=self._begin() if self.synchronous and not terminal and self.state.cursor>=5 else None
        if item is not None:self._infer_adopt(item)
        return event,terminal
    def _worker(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda:self.closed or self.state.cursor>=5)
                    if self.closed:return
                    item=self._begin()
                self._infer_adopt(item)
        except BaseException as error:
            with self.condition:
                self.error=error;self.events.append(dict(event='worker_error',reason=repr(error)));self.condition.notify_all()
    def close(self):
        with self.condition:self.closed=True;self.condition.notify_all()
        if self.worker is not None:
            self.worker.join(timeout=30)
            if self.worker.is_alive():raise RuntimeError('worker join timeout')
        if self.error is not None:raise RuntimeError('inference worker failed') from self.error
