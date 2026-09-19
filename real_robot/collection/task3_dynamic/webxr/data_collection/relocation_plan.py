"""Resolve the next retained demonstration slot; no hardware access."""
import hashlib
import json
from pathlib import Path


def next_assignment(plan_path, captures):
    path = Path(plan_path)
    plan = json.loads(path.read_text())
    entries = plan['episodes']
    used = set()
    for manifest in Path(captures).glob('session_*/episodes/episode_*/manifest.json'):
        row = json.loads(manifest.read_text())
        if row.get('relocation_plan_id') != plan['plan_id']:
            raise ValueError('Unexpected episode in dynamic-only capture directory')
        index = row['relocation_plan_index']
        if type(index) is not int or index in used or not 1 <= index <= len(entries):
            raise ValueError('Duplicate or invalid relocation plan index')
        if row.get('planned_destination_direction') != entries[index-1]['destination']:
            raise ValueError('Relocation direction differs from saved plan')
        used.add(index)
    for entry in entries:
        index = entry['planned_index']
        if index not in used:
            return {
                'collection_condition': 'single_relocation_manual_cue',
                'relocation_plan_id': plan['plan_id'],
                'relocation_plan_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'relocation_plan_index': index,
                'planned_destination_direction': entry['destination'],
                'relocation_trigger': 'human_judged_after_grasp_before_descent',
                'destination_markers_present': False,
                'actual_destination_position': None,
                'relocation_event_verified': False,
            }
    return None
