"""Scoped exact-cycle skipping and actual work accounting for validator-only projection."""
from contextlib import contextmanager
from contextvars import ContextVar

_WORK = ContextVar("exact_validator_cycle_work_v1", default=None)

@contextmanager
def enabled():
    token = _WORK.set(dict(valid_projector_calls=0, executed_cycles=0,
                           skipped_cycles=0, fixed_points=[]))
    try:
        yield
    finally:
        _WORK.reset(token)

def current_work():
    return _WORK.get()

def implementation_receipt():
    work = _WORK.get()
    result = dict(identity="EXACT_VALIDATOR_CYCLE_V1", enabled_in_context=work is not None,
                  scope="VALIDATOR_ONLY_CYCLIC_PROJECTOR",
                  projection_iterations_semantics="EQUIVALENT_BASELINE_CYCLES_USE_EXECUTED_CYCLES_FOR_ACTUAL_WORK")
    if work is not None:
        result.update(valid_projector_calls=work["valid_projector_calls"],
                      executed_cycles=work["executed_cycles"], skipped_cycles=work["skipped_cycles"],
                      equivalent_baseline_cycles=work["executed_cycles"] + work["skipped_cycles"],
                      fixed_points=[dict(event) for event in work["fixed_points"]])
    return result
