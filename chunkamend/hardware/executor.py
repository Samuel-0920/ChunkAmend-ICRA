"""Bounded future-boundary commits; no hardware/model dependencies at import.

Q is anchored to the last OLD target planned before the reserved boundary.
Only already completed feedback fits the response model. The old buffer stays
immutable until that boundary. A missed deadline publishes the raw response;
a late correction cannot alter it. Controller I/O never holds the state lock.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
import threading
import time
import numpy as np
from .contracts import validate_result, FeedbackSnapshot


@dataclass
class Reservation:
    request: object
    boundary: int
    delay: int
    raw_normalized: np.ndarray
    raw_physical: np.ndarray
    query_state: np.ndarray
    anchor: np.ndarray
    feedback: FeedbackSnapshot
    correction: object = None
    correction_s: float = None
    expired: bool = False


class ReservedQExecutor:
    def __init__(self, variant_name, coordinates, infer, feedback_snapshot, *, q_correct, budget_steps=3):
        if variant_name not in ('Q3_SINGLE_R0','Q6_COMBO_PROJECT_ASYNC'):
            raise ValueError('This candidate only implements Q3/Q6')
        if type(budget_steps) is not int or not 1<=budget_steps<=5:raise ValueError('budget_steps must be 1..5')
        self.variant=variant_name;self.coordinates=coordinates;self.infer=infer
        self.feedback_snapshot=feedback_snapshot;self.q_correct=q_correct;self.budget_steps=budget_steps
        self.cv=threading.Condition(threading.RLock())
        self.infer_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='q-model')
        self.q_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='q-correction')
        self._events=[];self.closed=False;self.error=None;self.in_slot=False
        self.normalized=self.physical=self.query_state=None
        self.cursor=0;self.generation_start=0;self._generation=0;self.next_request=1
        self.request=None;self.pending=None;self.q_busy=False
        self.feedback=None;self.last_completed_target=None

    def emit(self,event,**data):
        self._events.append(dict(event=event,monotonic_ns=time.monotonic_ns(),**data))

    def check(self):
        if self.closed:raise RuntimeError('Candidate closed')
        if self.error is not None:raise RuntimeError('Candidate worker or control slot failed') from self.error
        if self.physical is None:raise RuntimeError('Candidate not initialized')

    def payload(self,request,initial=False):
        return dict(request_id=request.request_id,generation=request.generation,observation=request.observation,
                    variant=self.variant,kind='initial' if initial else 'raw')

    def validate_raw(self,result,request):
        n,a,q=validate_result(result,request)
        np.testing.assert_allclose(a,self.coordinates.absolute(n,q),rtol=3e-6,atol=3e-6)
        return n,a,q

    def initialize(self,observation):
        with self.cv:
            if self.closed or self.physical is not None:raise RuntimeError('Already initialized or closed')
        req=SimpleNamespace(request_id=0,generation=0,stride=0,observation=deepcopy(observation))
        n,a,q=self.validate_raw(self.infer(self.payload(req,True)),req)
        feedback=self.feedback_snapshot()
        if not isinstance(feedback,FeedbackSnapshot) or feedback.anchor_joint is None:
            raise ValueError('Completed feedback with an explicit anchor required')
        with self.cv:
            if self.closed:raise RuntimeError('Closed during initialization')
            self.normalized=n;self.physical=a;self.query_state=q;self.feedback=feedback
            self.last_completed_target=feedback.anchor_joint.copy()
            self.emit('initialized',budget_steps=self.budget_steps)

    def start_if_eligible(self,observation):
        with self.cv:
            self.check()
            if self.in_slot or self.request is not None or self.pending is not None or self.cursor<5:return False
            if self.cursor>=15:raise RuntimeError('No remaining horizon for inference')
            req=SimpleNamespace(request_id=self.next_request,generation=self._generation,
                                stride=self.cursor,observation=deepcopy(observation))
            self.next_request+=1;self.request=req
            self.emit('inference_started',request_id=req.request_id,generation=req.generation,stride=req.stride)
            self.infer_pool.submit(self._infer,req)
            return True

    def _infer(self,req):
        try:
            n,a,q=self.validate_raw(self.infer(self.payload(req)),req)
            wait_start=time.monotonic()
            with self.cv:
                # No worker reads a ledger while its control callback updates it.
                self.cv.wait_for(lambda:not self.in_slot or self.closed or self.error is not None)
                if self.closed:return
                self.check()
                if self.request is not req or req.generation!=self._generation:return
                raw_delay=self.cursor-req.stride
                if raw_delay<0 or raw_delay>10:raise RuntimeError('Raw model response exhausted H15/K5 support')
                prefix_end=self.generation_start+5
                boundary=max(prefix_end,min(self.cursor+self.budget_steps,15,req.stride+10))
                can_correct=not self.q_busy and boundary>self.cursor
                if not can_correct:boundary=max(self.cursor,prefix_end)
                delay=boundary-req.stride
                anchor=(self.physical[boundary-1] if boundary>self.cursor else self.last_completed_target).copy()
                reservation=Reservation(req,boundary,delay,n,a,q,anchor,self.feedback)
                self.pending=reservation;self.request=None
                self.emit('raw_ready',request_id=req.request_id,delay=raw_delay,
                          slot_wait_s=time.monotonic()-wait_start,feedback_pairs=len(self.feedback.normalized_actions))
                self.emit('commit_reserved',request_id=req.request_id,old_generation=self._generation,
                          cursor=self.cursor,boundary=boundary,delay=delay,
                          reserved_steps=boundary-self.cursor,minimum_prefix_end=prefix_end,anchor_joint=anchor.tolist(),
                          anchor_role='planned_old_target_before_commit_not_measured_feedback')
                if can_correct:
                    self.q_busy=True;self.q_pool.submit(self._correct,reservation)
                else:self.emit('q_skipped',request_id=req.request_id,reason='Q_BUSY' if self.q_busy else 'NO_BUDGET')
        except BaseException as exc:
            with self.cv:
                if not self.closed:self.error=exc;self.cv.notify_all()

    def validate_correction(self,r,result):
        n=np.asarray(result.normalized_model_actions);a=np.asarray(result.absolute_joint_actions)
        if n.shape!=(15,32) or a.shape!=(15,7) or not np.isfinite(n).all() or not np.isfinite(a).all():
            raise ValueError('Invalid Q correction')
        mask=np.ones(15,dtype=bool);mask[r.delay:r.delay+5]=False
        np.testing.assert_array_equal(a[mask],r.raw_physical[mask])
        np.testing.assert_array_equal(n[mask],r.raw_normalized[mask])
        np.testing.assert_array_equal(a[:,6],r.raw_physical[:,6])
        np.testing.assert_array_equal(n[:,6:],r.raw_normalized[:,6:])
        if result.receipt.get('reason')!='SELECTED':
            np.testing.assert_array_equal(a,r.raw_physical)
            np.testing.assert_array_equal(n,r.raw_normalized)
        np.testing.assert_allclose(a,self.coordinates.absolute(n,r.query_state),rtol=3e-6,atol=3e-6)
        return SimpleNamespace(normalized_model_actions=n.copy(),absolute_joint_actions=a.copy(),receipt=deepcopy(result.receipt))

    def _correct(self,r):
        started=time.monotonic()
        try:
            result=self.q_correct(r.raw_normalized.copy(),r.raw_physical.copy(),r.query_state.copy(),r.feedback,
                                  delay=r.delay,anchor_joint=r.anchor.copy())
            result=self.validate_correction(r,result)
            elapsed=time.monotonic()-started
            with self.cv:
                if self.closed:return
                r.correction_s=elapsed
                if r.expired or self.pending is not r:
                    self.emit('q_result',request_id=r.request.request_id,adopted=False,correction_s=elapsed,
                              receipt=result.receipt,reason='MISSED_RESERVED_BOUNDARY')
                else:
                    r.correction=result
                    self.emit('q_prepared',request_id=r.request.request_id,correction_s=elapsed,receipt=result.receipt)
        except BaseException as exc:
            with self.cv:
                if not self.closed:
                    self.error=exc;self.emit('q_fault',request_id=r.request.request_id,reason=repr(exc))
        finally:
            with self.cv:self.q_busy=False;self.cv.notify_all()

    def _commit_if_due(self):
        r=self.pending
        if r is None or self.cursor<r.boundary:return
        if self.cursor!=r.boundary:raise RuntimeError('Reserved boundary was crossed')
        # The target used to convert Q must have actually been accepted in order.
        np.testing.assert_allclose(self.last_completed_target[:6],r.anchor[:6],rtol=0,atol=1e-7)
        self._generation+=1;self.cursor=r.delay;self.generation_start=r.delay;self.query_state=r.query_state.copy()
        if r.correction is not None:
            self.normalized=r.correction.normalized_model_actions.copy()
            self.physical=r.correction.absolute_joint_actions.copy()
            self.emit('correction_adopted',request_id=r.request.request_id,generation=self._generation,cursor=self.cursor)
            self.emit('q_result',request_id=r.request.request_id,adopted=True,correction_s=r.correction_s,
                      receipt=r.correction.receipt,generation=self._generation,
                      corrected_start=r.delay,corrected_end=r.delay+5,
                      raw_prefix=r.raw_physical[r.delay:r.delay+5].tolist(),
                      selected_prefix=self.physical[r.delay:r.delay+5].tolist())
        else:
            self.normalized=r.raw_normalized.copy();self.physical=r.raw_physical.copy()
            self.emit('raw_deadline_fallback',request_id=r.request.request_id,generation=self._generation,cursor=self.cursor)
        r.expired=True;self.pending=None

    def step(self,complete):
        with self.cv:
            self.check()
            if self.in_slot:raise RuntimeError('Concurrent controller owners are forbidden')
            self._commit_if_due()
            if self.cursor>=15:raise RuntimeError('H15 buffer exhausted; stop instead of reusing stale targets')
            row=self.physical[self.cursor].copy();index=self.cursor;generation=self._generation
            self.in_slot=True
        # Hardware/observation I/O runs without holding cv. Completed feedback
        # remains unpublished until all of this callback succeeds.
        try:
            complete(row,index,generation)
            snapshot=self.feedback_snapshot()
            if not isinstance(snapshot,FeedbackSnapshot):raise ValueError('Invalid completed feedback')
        except BaseException as exc:
            with self.cv:self.in_slot=False;self.error=exc;self.cv.notify_all()
            raise
        with self.cv:
            self.feedback=snapshot;self.last_completed_target=row.copy();self.cursor=index+1
            self.in_slot=False;self.cv.notify_all()
        return row,index,generation

    @property
    def generation(self):
        with self.cv:return self._generation
    @property
    def events(self):return self.diagnostic_events()
    def diagnostic_events(self):
        with self.cv:return deepcopy(self._events)
    def close(self):
        with self.cv:self.closed=True;self.cv.notify_all()
        self.infer_pool.shutdown(wait=True,cancel_futures=True)
        self.q_pool.shutdown(wait=True,cancel_futures=True)
