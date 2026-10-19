"""Current dataset identity policy; compatible with Python 3.5 and TensorFlow 1.12."""
import json
from pathlib import Path
POLICY_PATH = Path(__file__).resolve().parents[1] / 'dataset/current/identity_policy.json'

def policy():
    with POLICY_PATH.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def canonical_position(value):
    rules = policy()
    value = str(value)
    return rules['aliases'].get(value, value)

def require_active_position(value):
    result = canonical_position(value)
    if result in policy()['excluded_positions']:
        raise ValueError('Position excluded from the active dataset: ' + result)
    return result

def tube_id(value):
    return require_active_position(value).rsplit('-', 1)[0]

def identify_position(path):
    rules = policy()
    for part in reversed(Path(path).parts):
        if part in rules['aliases'] or part in rules['aliases'].values() or part in rules['excluded_positions']:
            return require_active_position(part)
        bits = part.split('-')
        if len(bits) == 3 and bits[0] in ['0353', 'A017', 'Q1']:
            return require_active_position(part)
    return None

def normalize_prediction_io(input_path, output_path):
    position = identify_position(input_path)
    rules = policy()
    parts = []
    for part in Path(output_path).parts:
        if part in rules['excluded_positions']:
            raise ValueError('Output refers to an excluded position: ' + part)
        parts.append(rules['aliases'].get(part, part))
    return Path(*parts), position
