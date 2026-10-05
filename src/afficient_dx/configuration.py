import argparse
import json
from pathlib import Path

def parse_configured(parser, argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config', type=Path)
    selected, _ = preliminary.parse_known_args(argv)
    parser.add_argument('--config', type=Path)
    if selected.config:
        values = json.loads(selected.config.read_text())
        if not isinstance(values, dict):
            raise ValueError('Configuration must be a JSON object')
        actions = {a.dest: a for a in parser._actions}
        unknown = set(values) - set(actions)
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        for key, value in values.items():
            action = actions[key]
            if action.required and value is not None:
                action.required = False
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                if not isinstance(value, bool):
                    raise ValueError(f'{key} must be boolean')
            elif value is not None and action.type:
                if action.nargs in ('+', '*'):
                    if not isinstance(value, list) or action.nargs == '+' and not value:
                        raise ValueError(f'{key} must be a nonempty list')
                    values[key] = [action.type(v) for v in value]
                else:
                    values[key] = action.type(value)
            check = values[key] if action.nargs in ('+', '*') else [values[key]]
            if action.choices and any((v not in action.choices for v in check)):
                raise ValueError(f'Invalid value for {key}')
        parser.set_defaults(**values)
    return parser.parse_args(argv)
