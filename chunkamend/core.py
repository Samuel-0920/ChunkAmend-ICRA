"""Factories for the paper's simulation and hardware correction kernels."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
EXECUTE_STEPS=5

def _namespace(runtime: dict, domain) -> SimpleNamespace:
    types = importlib.import_module(f'chunkamend.{domain}.core.methods.m1_types')
    RapCandidateSpec, RapConfig = (types.RapCandidateSpec, types.RapConfig)
    core = dict(runtime['core'])
    core['objective_weights'] = tuple(core['objective_weights'])
    candidate_specs = tuple((RapCandidateSpec(k=item['k'], alpha=item['alpha'], transition_weights=tuple(item['transition_weights']), relative_cap=item['relative_cap']) for item in runtime['candidate_specs']))
    if not candidate_specs or any((item.k > EXECUTE_STEPS for item in candidate_specs)):
        raise ValueError('UF850 Q candidates must operate wholly in the K=5 prefix')
    if int(core['k_max']) != EXECUTE_STEPS:
        raise ValueError('Source configuration does not preserve the UF850 K=5 contract')
    return SimpleNamespace(core=RapConfig(**core), candidate_specs=candidate_specs, response_rotation_ridge=float(runtime['response_rotation_ridge']), reversal=SimpleNamespace(**runtime['reversal']), method_identity=runtime['method_identity'], objective_count_mode=runtime['objective_count_mode'], selection_mode=runtime['selection_mode'], candidate_ranking_mode=runtime['candidate_ranking_mode'], reversal_space=runtime['reversal_space'], uncertainty_scale=runtime['uncertainty_scale'], minimum_prequential_predictions=runtime['minimum_prequential_predictions'], candidate_evaluation_policy=runtime['candidate_evaluation_policy'], preprojection_objective_bound=runtime['preprojection_objective_bound'], projector_context_cache_mode=runtime['projector_context_cache_mode'], terminal_constraint_mode=runtime['terminal_constraint_mode'], terminal_projection_role=runtime['terminal_projection_role'])

def make_corrector(domain, q01, q99):
    if domain not in ('simulation', 'hardware'):
        raise ValueError('Expected simulation or hardware')
    settings=json.loads((Path(__file__).resolve().parents[1]/'configs/chunkamend.json').read_text())
    module=importlib.import_module(f'chunkamend.{domain}.core.methods.remaining_window')
    return module.WindowCorrector(_namespace(settings['runtime'], domain), settings['q_mode'],
        settings['rotation_arm'], q01, q99,
        ranking_correction_penalty=settings['ranking_correction_penalty'], terminal_veto=True)
