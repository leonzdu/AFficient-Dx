import json
from pathlib import Path
import numpy as np
import pandas as pd
from afficient_dx.metrics import scores


def repository_root():
    source = Path(__file__).resolve().parents[2]
    return source if (source / 'artifacts').is_dir() else Path.cwd()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_frame(path):
    return pd.read_csv(path, float_precision='round_trip')


def load_runs(matched_dir, benchmark_dir):
    runs = []
    for scope, directory, pattern in (
        ('matched_100hz', Path(matched_dir), '*/seed_*/result.json'),
        ('benchmark_100hz', Path(benchmark_dir), 'runs/*/seed_*/result.json'),
    ):
        for path in sorted(directory.glob(pattern)):
            result = read_json(path)
            if result.get('status') != 'complete' or not result.get('test'):
                raise ValueError(f'Incomplete result: {path}')
            run = path.parent
            model = result.get('model', run.parent.name)
            runs.append({'scope': scope, 'model': model, 'seed': result['seed'],
                         'directory': run, 'result': result,
                         'predictions': read_frame(run / 'test_predictions.csv')})
    if not runs:
        raise FileNotFoundError('No completed experiment results found')
    keys = [(run['scope'], run['model'], run['seed']) for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate model/seed results')
    return runs


def rescore(run):
    frame = run['predictions']
    result = run['result']['test']['primary']
    return scores(frame.y.to_numpy(), frame.probability.to_numpy(),
                  result['threshold'], result.get('ece_bins', 10),
                  frame.logit.to_numpy())


def summarize_runs(runs):
    rows = []
    labels = {'tiny_100hz': 'AFficient-Dx', 'reference_100hz': 'Reference'}
    groups = {}
    metrics = ['auroc', 'auprc', 'sensitivity', 'specificity', 'ppv', 'f1',
               'brier', 'ece', 'tn', 'fp', 'fn', 'tp']
    for run in runs:
        groups.setdefault((run['scope'], run['model']), []).append(run)
    for (scope, model), group in groups.items():
        first = group[0]['result']
        parameters = {run['result']['parameters'] for run in group}
        if len(parameters) != 1:
            raise ValueError(f'Parameter counts differ across seeds: {model}')
        row = {'scope': scope, 'model': model,
               'model_display': first.get('model_display', labels.get(model, model)),
               'frequency_hz': 100, 'parameters': first['parameters'],
               'n_seeds': len(group)}
        values = [rescore(run) for run in group]
        for metric in metrics:
            array = np.asarray([value[metric] for value in values], dtype=float)
            row[f'{metric}_mean'] = float(array.mean())
            row[f'{metric}_seed_sd'] = float(array.std(ddof=1)) if len(array) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows)
