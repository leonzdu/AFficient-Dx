import argparse
import json
from pathlib import Path
import numpy as np
import torch
from afficient_dx.artifacts import load_runs, read_frame, read_json, repository_root, rescore
from afficient_dx.metrics import apply_calibrator
from afficient_dx.prediction import load_checkpoint
from afficient_dx.utils import sha256


def verify_manifest(root):
    path = Path(root) / 'artifacts/manifest.json'
    manifest = read_json(path)
    for relative, expected in manifest['files'].items():
        actual = Path(root) / relative
        if not actual.is_file() or sha256(actual) != expected:
            raise ValueError(f'Artifact hash mismatch: {relative}')
    included = {str(p.relative_to(root)) for p in (Path(root) / 'artifacts').rglob('*')
                if p.is_file() and p != path}
    if included != set(manifest['files']):
        raise ValueError('Artifact file inventory differs from manifest')
    return len(included)


def verify_runs(runs):
    identities = {}
    for run in runs:
        result, frame, directory = run['result'], run['predictions'], run['directory']
        if not frame.ecg_id.is_unique or frame[['ecg_id', 'patient_id', 'y']].isna().any().any():
            raise ValueError(f'Invalid test identities: {directory}')
        identity = frame[['ecg_id', 'patient_id', 'y']].sort_values('ecg_id').reset_index(drop=True)
        scope = run['scope']
        if scope in identities and not identity.equals(identities[scope]):
            raise ValueError(f'Test identities differ between models/seeds: {directory}')
        identities[scope] = identity
        primary = result['test']['primary']
        recalculated = rescore(run)
        for key in ['n', 'positives', 'auroc', 'auprc', 'sensitivity', 'specificity', 'ppv',
                    'brier', 'ece', 'tn', 'fp', 'fn', 'tp']:
            if not np.isclose(recalculated[key], primary[key], rtol=1e-8, atol=2e-8):
                raise ValueError(f'Result/prediction mismatch for {key}: {directory}')
        model, frozen = load_checkpoint(directory)
        if sum(p.numel() for p in model.parameters()) != result['parameters']:
            raise ValueError(f'Model parameter count mismatch: {directory}')
        table = read_frame(directory / 'parameter_tensors.csv')
        if int(table.parameters.sum()) != result['parameters']:
            raise ValueError(f'Parameter table mismatch: {directory}')
        probability = apply_calibrator(frame.logit.to_numpy(), frozen['calibrator'])
        if not np.allclose(probability, frame.probability.to_numpy(), rtol=1e-8, atol=1e-10):
            raise ValueError(f'Calibration mismatch: {directory}')
        predictions = (frame.probability.to_numpy() >= frozen['threshold']).astype(int)
        if not np.array_equal(predictions, frame.prediction.to_numpy()):
            raise ValueError(f'Frozen threshold mismatch: {directory}')
        roles = []
        if (directory / 'split_manifest.csv').is_file():
            split = read_frame(directory / 'split_manifest.csv')
            roles = [(name, set(part.patient_id)) for name, part in split.groupby('role')]
        else:
            for name in ['selection', 'calibration', 'threshold']:
                path = directory / f'{name}_predictions.csv'
                if path.is_file():
                    roles.append((name, set(read_frame(path).patient_id)))
            roles.append(('test', set(frame.patient_id)))
        for index, (name, patients) in enumerate(roles):
            for other, other_patients in roles[:index]:
                if patients & other_patients:
                    raise ValueError(f'Patients overlap in {name}/{other}: {directory}')
        if scope == 'matched_100hz':
            args = result['configuration']['arguments']
            if args['frequency'] != 100 or len(args['leads']) != 12 or args['epochs'] != 40:
                raise ValueError(f'Matched protocol mismatch: {directory}')
        elif result['epochs_completed'] > 6:
            raise ValueError(f'Benchmark exceeds six epochs: {directory}')
        x = torch.zeros(1, len(frozen['leads']), 10 * frozen['frequency_hz'])
        aux = {'record_index': torch.zeros(1, dtype=torch.int64),
               'peak_position': torch.tensor([500.]), 'rr': torch.zeros(1, 2)}
        from afficient_dx.models.published import forward_record
        with torch.inference_mode():
            if not torch.isfinite(forward_record(model, x, aux)).all():
                raise ValueError(f'Nonfinite checkpoint output: {directory}')
    return len(runs)


def verify(root):
    root = Path(root).resolve()
    files = verify_manifest(root)
    runs = load_runs(root / 'artifacts/matched_100hz', root / 'artifacts/benchmark_100hz')
    count = verify_runs(runs)
    expected = {('matched_100hz', model, seed) for model in ['tiny_100hz', 'reference_100hz'] for seed in [42, 43, 44]}
    expected |= {('benchmark_100hz', model, seed) for model in ['afficient_419', 'smaller_238', 'basso_ph_multiscopic', 'busia_transformer'] for seed in [42, 43, 44]}
    actual = {(run['scope'], run['model'], run['seed']) for run in runs}
    if actual != expected:
        raise ValueError('Archived run inventory differs from the paper experiments')
    benchmark = root / 'artifacts/benchmark_100hz'
    split = read_frame(benchmark / 'split_manifest.csv')
    sets = [set(part.patient_id) for _, part in split.groupby('role')]
    if any(left & right for index, left in enumerate(sets) for right in sets[:index]):
        raise ValueError('Benchmark patient leakage')
    if read_json(benchmark / 'execution_plan.json')['epochs'] != 6:
        raise ValueError('Archived benchmark budget mismatch')
    return {'status': 'verified', 'artifact_files': files, 'runs': count,
            'checks': ['hashes', 'weights', 'parameter_counts', 'test_identities',
                       'metrics', 'calibration', 'thresholds', 'patient_separation', 'epoch_budgets']}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=repository_root())
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(args.threads)
    print(json.dumps(verify(args.root), indent=2))


if __name__ == '__main__':
    main()
