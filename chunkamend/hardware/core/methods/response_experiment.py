"""Explicit Q3 window×ridge development variants; never an implicit default change."""
from dataclasses import dataclass, replace, asdict
import hashlib
import json
import numpy as np
from chunkamend.hardware.core.methods.m1_types import ResponseFit
from chunkamend.hardware.core.methods.rap_response import (
    fit_completed_translation_response, _unavailable, MIN_SAMPLES, CONDITION_LIMIT,
)


@dataclass(frozen=True)
class ResponseExperiment:
    window: int
    ridge: float

    def __post_init__(self):
        if (type(self.window) is not int or self.window not in {16, 64}
                or type(self.ridge) is not float or self.ridge not in {1e-4, 1e-3}):
            raise ValueError("only the explicit Q3 2×2 development grid is supported")

    @property
    def identity(self):
        return f"Q3_WINDOW_RIDGE_DEV_V1_W{self.window}_L{self.ridge:g}"

    def receipt(self):
        return dict(identity=self.identity, window=self.window, ridge=self.ridge,
                    min_samples=8, condition_limit=1e8, affine=False,
                    objective="SSE_PLUS_RIDGE_NOT_MEAN_SSE",
                    input_semantics="FROZEN_SELECTED_NORMALIZED_TRANSLATION_PRECLIP",
                    solver="NUMPY_NORMAL_EQUATION_SOLVE", rank_gate_changed=False,
                    physical_input_clipping_changed=False, research_status="UNTESTED_CLOSED_LOOP")


def fit_response_experiment(actions, tcp_displacements, experiment, *, fit_age=0):
    if not isinstance(experiment, ResponseExperiment):
        raise ValueError("explicit response experiment required")
    if type(fit_age) is not int or fit_age < 0:
        return _unavailable("NONPRIMARY_RESPONSE_CONFIG", fit_age=0)
    if experiment.window == 64 and experiment.ridge == 1e-4:
        return fit_completed_translation_response(actions, tcp_displacements, fit_age=fit_age)
    try:
        x = np.asarray(actions, dtype=np.float64)
        y = np.asarray(tcp_displacements, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return _unavailable("NONFINITE_OR_SHAPE", fit_age=fit_age)
    if (x.ndim != 2 or x.shape[1:] != (3,) or y.shape != x.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        return _unavailable("NONFINITE_OR_SHAPE", fit_age=fit_age)
    end = len(x)
    start = max(0, end-experiment.window)
    x, y = x[start:end], y[start:end]
    count = len(x)
    rank = int(np.linalg.matrix_rank(x)) if count else 0
    system = x.T @ x + experiment.ridge*np.eye(3)
    condition = float(np.linalg.cond(system))
    diagnostics = dict(sample_count=count, window_start=start, window_end=end,
                       condition_number=condition, rank=rank, fit_age=fit_age)
    if count < MIN_SAMPLES:
        return _unavailable("INSUFFICIENT_COMPLETED_PAIRS", **diagnostics)
    if not np.isfinite(condition) or condition > CONDITION_LIMIT:
        return _unavailable("ILL_CONDITIONED_NORMAL_SYSTEM", **diagnostics)
    try:
        matrix = np.linalg.solve(system, x.T @ y)
    except np.linalg.LinAlgError:
        return _unavailable("NORMAL_SOLVE_FAILURE", **diagnostics)
    residual = float(np.sqrt(np.mean((x@matrix-y)**2)))
    if not np.isfinite(matrix).all() or not np.isfinite(residual):
        return _unavailable("NONFINITE_FIT", **diagnostics)
    return ResponseFit(matrix=matrix, available=True, reason="AVAILABLE",
                       residual_rmse=residual, **diagnostics)


def build_experimental_runtime(anchor, experiment):
    """Caller must supply the separately verified historical Q3_T70_R_E25 runtime.

    No missing bank configuration, seeds or policy inputs are manufactured here.
    Only the public method identity changes; estimator override is held separately
    in ActiveResponseState and appears in each response receipt.
    """
    if not isinstance(experiment, ResponseExperiment):
        raise ValueError("explicit response experiment required")
    if (anchor.method_identity != "OVERNIGHT-26ARM__Q3_T70_R_E25"
            or anchor.method_id.value != "COVER_ALPHA_BANK_065_095_VALIDATOR_ONLY_CAP003"
            or anchor.response_mode != "PRIMARY" or anchor.risk is not None
            or anchor.response.min_samples != 8 or anchor.response.ridge != 1e-4
            or anchor.response.condition_limit != 1e8):
        raise ValueError("requires the verified historical Q3 anchor runtime")
    if (anchor.core.correction_cap != .03 or anchor.core.k_max != 5
            or tuple(anchor.core.objective_weights) != (1., .5, .25)
            or anchor.reversal_space != "RESPONSE_TCP"
            or anchor.reversal.reversal_cos_threshold != -.5
            or anchor.objective_count_mode != "EXECUTED_ONLY"
            or anchor.selection_mode != "ACTION_OBJECTIVE"
            or anchor.terminal_constraint_mode != "EXECUTED_INTERNAL"
            or anchor.terminal_projection_role != "VALIDATE_ONLY"
            or len(anchor.candidate_specs) != 1):
        raise ValueError("anchor treatment does not match frozen T70/R/E25 context")
    spec = anchor.candidate_specs[0]
    if (spec.k != 5 or spec.alpha != .7 or spec.relative_cap != 1.
            or not np.allclose(np.asarray(spec.transition_weights)*spec.alpha,
                               [.7,.65,.6,.55,.5], rtol=0, atol=1e-12)):
        raise ValueError("anchor translation strengths differ")
    runtime = replace(anchor, method_identity=experiment.identity)
    values = asdict(anchor)
    identity_hash = hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()
    return runtime, dict(response_experiment=experiment.receipt(),
                         anchor_runtime_sha256=identity_hash,
                         base_fields_preserved=True, seed_registry_qualified=False)
