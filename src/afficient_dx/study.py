from afficient_dx.configuration import parse_configured
from afficient_dx.utils import source_fingerprint
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

def hash_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def write_rows(path, rows, empty_columns=None):
    columns = list(dict.fromkeys((key for row in rows for key in row))) or empty_columns or ['status']
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def find_trainer(explicit):
    if explicit:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    from afficient_dx import training
    return Path(training.__file__).resolve()

def experiments(suites):
    matched = 'matched' in suites
    if matched and len(suites) != 1:
        raise ValueError('The matched suite must be selected alone')
    selected = {'core', 'width', 'ablation', 'wearable'} if 'all' in suites else {'core'} if matched else set(suites)
    items = {}

    def add(name, suite, base=1, frequency=50, extra=None, leads=None, benchmark=False):
        if suite in selected:
            items[name] = {'name': name, 'suite': suite, 'base': base, 'frequency': frequency, 'extra': extra or [], 'leads': leads or ['all'], 'benchmark': benchmark}
    for frequency in [50, 100]:
        add(f'tiny_{frequency}hz', 'core', frequency=frequency, benchmark=True)
        add(f'reference_{frequency}hz', 'core', base=32, frequency=frequency, benchmark=True)
    for base in [2, 3, 4, 8, 16]:
        add(f'width_b{base}_50hz', 'width', base=base)
    for widths in [[1, 1, 1], [1, 1, 2], [1, 2, 2], [1], [1, 2]]:
        add('channels_' + '_'.join(map(str, widths)) + '_50hz', 'width', extra=['--channels', *map(str, widths)])
    for name, extra in [('no_residual', ['--no-residual']), ('no_batchnorm', ['--no-batchnorm']), ('unweighted_bce', ['--no-class-weight']), ('no_dropout', ['--dropout', '0']), ('shuffled_training_labels', ['--shuffle-labels'])]:
        add(name + '_50hz', 'ablation', extra=extra)
    for lead in ['I', 'II']:
        add(f'single_lead_{lead}_50hz', 'wearable', leads=[lead], benchmark=True)
        add(f'single_lead_{lead}_reference_50hz', 'wearable', base=32, leads=[lead], benchmark=True)
    return [item for item in items.values() if not matched or item['frequency'] == 100]

def external_experiments(path):
    if not path:
        return []
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError('Baseline configuration must be a JSON list')
    items = []
    for record in records:
        if not isinstance(record, dict) or not all((record.get(k) for k in ['name', 'factory', 'source', 'adaptation'])):
            raise ValueError('Every baseline needs name, factory, source/revision, and adaptation')
        if not re.fullmatch('[a-zA-Z0-9_-]+', record['name']):
            raise ValueError('Baseline name must be a simple filename-safe identifier')
        module_name = record['factory'].rsplit(':', 1)[0]
        if ':' not in record['factory']:
            raise ValueError('Factory must be module:function')
        if not isinstance(record.get('kwargs', {}), dict):
            raise ValueError('Baseline kwargs must be a JSON object')
        spec = importlib.util.find_spec(module_name)
        if spec is None or not spec.origin or (not Path(spec.origin).is_file()):
            raise ValueError(f'Missing importable baseline adapter: {module_name}')
        for frequency in record.get('frequencies', [50, 100]):
            if frequency not in [50, 100]:
                raise ValueError('External baselines must use 50/100-Hz study inputs')
            items.append({'name': f"{record['name']}_{frequency}hz", 'suite': 'published_adapter', 'base': 1, 'frequency': frequency, 'leads': record.get('leads', ['all']), 'benchmark': True, 'adapter_sha256': hash_file(Path(spec.origin)), 'extra': ['--model-factory', record['factory'], '--model-name', record['name'], '--model-source', record['source'], '--model-adaptation', record['adaptation'], '--model-kwargs', json.dumps(record.get('kwargs', {}), sort_keys=True)]})
    return items

def common_arguments(args):
    values = ['--epochs', args.epochs, '--patience', args.patience, '--batch-size', args.batch_size, '--lr', args.lr, '--weight-decay', args.weight_decay, '--dropout', args.dropout, '--bootstrap', args.bootstrap, '--split-seed', args.split_seed, '--calibration', args.calibration, '--threshold-method', args.threshold_method, '--target-sensitivity', args.target_sensitivity, '--threshold-bootstrap', args.threshold_bootstrap, '--noise-seed', args.noise_seed, '--noise-repeats', args.noise_repeats, '--latency-repeats', args.latency_repeats, '--latency-warmup', args.latency_warmup, '--latency-threads', args.latency_threads, '--train-threads', args.train_threads, '--device', args.device, '--noise-snr', *args.noise_snr]
    if args.skip_test:
        values.append('--skip-test')
    if args.plots:
        values.append('--plots')
    return list(map(str, values))

def build_plan(args, script):
    grid = experiments(args.suite) + external_experiments(args.baseline_config)
    if len(set((item['name'] for item in grid))) != len(grid):
        raise ValueError('Duplicate experiment names')
    plan = []
    for item in grid:
        for seed in args.seeds:
            out = args.output_dir / item['name'] / f'seed_{seed}'
            command = [sys.executable, str(script), '--data-dir', str(args.data_dir), '--output-dir', str(out), '--base', str(item['base']), '--frequency', str(item['frequency']), '--seed', str(seed), '--leads', *item['leads'], *common_arguments(args), *item['extra']]
            if item['benchmark'] and (not args.no_robustness):
                command.append('--robustness')
            if item['benchmark'] and (not args.no_quantization):
                command.append('--quantize-int8')
            plan.append({'experiment': item['name'], 'suite': item['suite'], 'seed': seed, 'run_dir': str(out), 'command': command, 'adapter_sha256': item.get('adapter_sha256')})
    return plan

def literature_context(out, has_adapters):
    write_json(out / 'model_provenance.json', {'documentation': 'docs/PROVENANCE.md', 'adapters_configured': bool(has_adapters)})

def architecture_benchmark(args, script):
    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location('afficient_training', script)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    if args.latency_repeats <= 0 or args.benchmark_sessions <= 0:
        raise ValueError('Architecture timing requires positive repeats and sessions')
    out = args.output_dir
    report = {}
    for label, base in [('tiny', 1), ('reference', 32)]:
        for frequency in [50, 100]:
            trainer.set_seed(args.seeds[0], torch.device('cpu'))
            model = trainer.ECGResNet(base).eval()
            sample = np.random.default_rng(0).normal(size=(1, 12, 10 * frequency)).astype(np.float32)
            sessions = [trainer.benchmark_cpu(model, sample, args.latency_warmup, args.latency_repeats, args.latency_threads) for _ in range(args.benchmark_sessions)]
            name = f'{label}_{frequency}hz'
            layers = trainer.layer_report(model, 10 * frequency)
            write_rows(out / f'{name}_layers.csv', layers)
            torch.save(model.state_dict(), out / f'{name}_state.pt')
            report[name] = {'median_of_session_medians_ms': float(np.median([s['median_ms'] for s in sessions])), 'sessions': sessions, 'parameters': sum((p.numel() for p in model.parameters())), 'macs': sum((row['conv_linear_macs'] for row in layers)), 'serialized_state_dict_bytes': (out / f'{name}_state.pt').stat().st_size, 'scope': 'initialized architecture, fixed random normalized input; no clinical performance claim'}
            print(f"{name}: median {report[name]['median_of_session_medians_ms']:.6f} ms", flush=True)
    write_json(out / 'architecture_benchmarks.json', report)
    write_json(out / 'architecture_benchmark_config.json', {'train_source_sha256': hash_file(script), 'runner_source_sha256': hash_file(Path(__file__)), 'weight_seed': args.seeds[0], 'input_seed': 0, 'environment': trainer.runtime_metadata(), 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})

def aggregate(out, plan, paired_bootstrap):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score
    records, results, robustness = ([], {}, [])
    metrics = ['auroc', 'auprc', 'pr_trapezoid_auc', 'balanced_accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1', 'brier', 'log_loss', 'ece', 'false_positives_per_1000_ecgs', 'missed_af_per_1000_ecgs', 'referral_fraction']
    for item in plan:
        path = Path(item['run_dir'])
        result_file = path / 'result.json'
        if not result_file.is_file():
            continue
        result = json.loads(result_file.read_text())
        if result.get('status') != 'complete':
            continue
        results[item['experiment'], item['seed']] = result
        row = {'experiment': item['experiment'], 'suite': item['suite'], 'seed': item['seed'], 'frequency_hz': result['frequency_hz'], 'parameters': result['parameters'], 'leads': ','.join(result['leads']), 'macs_estimated': result['macs'], 'parameter_fp32_kB': result['resources']['parameter_fp32_kB'], 'serialized_state_dict_bytes': result['resources']['serialized_state_dict_bytes'], 'cpu_latency_median_ms': result['resources']['cpu_latency'].get('median_ms'), 'quantization_status': result['quantization']['status'], 'int8_serialized_bytes': result['quantization'].get('serialized_torchscript_bytes'), 'int8_cpu_latency_median_ms': result['quantization'].get('cpu_latency', {}).get('median_ms'), 'threshold': result['validation']['threshold']['primary']['threshold'], 'test_evaluated': result['test'] is not None}
        for split, source in [('selection', result['validation']['selection']), ('test', result['test'])]:
            for key in metrics:
                row[f'{split}_{key}'] = source['primary'].get(key) if source else None
        records.append(row)
        if (path / 'robustness.csv').is_file():
            for entry in pd.read_csv(path / 'robustness.csv').to_dict('records'):
                robustness.append({'experiment': item['experiment'], 'seed': item['seed'], **entry})
    write_rows(out / 'runs.csv', records)
    write_rows(out / 'robustness_all_runs.csv', robustness)
    grouped = []
    for name in sorted({r['experiment'] for r in records}):
        group = [r for r in records if r['experiment'] == name]
        for key in [f'{split}_{metric}' for split in ['selection', 'test'] for metric in metrics] + ['cpu_latency_median_ms']:
            values = [r[key] for r in group if r.get(key) is not None]
            grouped.append({'experiment': name, 'metric': key, 'n_seeds': len(values), 'mean': float(np.mean(values)) if values else None, 'sample_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None, 'note': 'seed variation, not a confidence interval'})
    write_rows(out / 'summary.csv', grouped)
    width_rows = []
    for name in sorted({r['experiment'] for r in records if r['suite'] in ['core', 'width'] and r['frequency_hz'] == 50}):
        group = [r for r in records if r['experiment'] == name]
        width_rows.append({'experiment': name, 'parameters': group[0]['parameters'], 'selection_auprc_mean': float(np.mean([r['selection_auprc'] for r in group]))})
    for row in width_rows:
        row['selection_pareto'] = not any((other['parameters'] <= row['parameters'] and other['selection_auprc_mean'] >= row['selection_auprc_mean'] and (other['parameters'] < row['parameters'] or other['selection_auprc_mean'] > row['selection_auprc_mean']) for other in width_rows))
        row['scope'] = 'explored grid only; descriptive selection-data frontier; no optimality claim'
    write_rows(out / 'width_frontier.csv', width_rows)
    contrasts = []
    comparisons = [('reference_50hz', 'tiny_50hz', 'width_at_50hz'), ('reference_100hz', 'tiny_100hz', 'width_at_100hz'), ('tiny_100hz', 'tiny_50hz', 'sampling_rate_tiny'), ('reference_100hz', 'reference_50hz', 'sampling_rate_reference')]
    for left, right, label in comparisons:
        seeds = sorted({seed for name, seed in results if name == left} & {seed for name, seed in results if name == right})
        for seed in seeds:
            a, b = (results[left, seed], results[right, seed])
            for split in ['selection', 'test']:
                ma = a['validation'][split] if split != 'test' else a['test']
                mb = b['validation'][split] if split != 'test' else b['test']
                if ma is None or mb is None:
                    continue
                intervals = {}
                if split == 'test' and paired_bootstrap:
                    pa = pd.read_csv(out / left / f'seed_{seed}' / 'test_predictions.csv')
                    pb = pd.read_csv(out / right / f'seed_{seed}' / 'test_predictions.csv')
                    merged = pa.merge(pb, on=['ecg_id', 'patient_id', 'y'], validate='one_to_one', suffixes=('_a', '_b'))
                    if len(merged) != len(pa) or len(merged) != len(pb):
                        raise ValueError('Paired comparison contains unmatched test ECGs')
                    y = merged.y.to_numpy()
                    p_a, p_b = (merged.logit_a.to_numpy(), merged.logit_b.to_numpy())
                    lookup = [np.flatnonzero(merged.patient_id.to_numpy() == patient) for patient in merged.patient_id.unique()]
                    rng = np.random.default_rng(seed + 9001)
                    draws = {'auroc': [], 'auprc': []}
                    for _ in range(20 * paired_bootstrap):
                        if len(draws['auroc']) == paired_bootstrap:
                            break
                        idx = np.concatenate([lookup[i] for i in rng.integers(0, len(lookup), len(lookup))])
                        if np.unique(y[idx]).size < 2:
                            continue
                        draws['auroc'].append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
                        draws['auprc'].append(average_precision_score(y[idx], p_a[idx]) - average_precision_score(y[idx], p_b[idx]))
                    for key, values in draws.items():
                        intervals[key] = np.percentile(values, [2.5, 97.5]).tolist() if values else None
                for metric in ['auroc', 'auprc']:
                    ci = intervals.get(metric)
                    contrasts.append({'comparison': label, 'left': left, 'right': right, 'seed': seed, 'split': split, 'metric': metric, 'left_minus_right': ma['primary'][metric] - mb['primary'][metric], 'paired_patient_ci_low': float(ci[0]) if ci else None, 'paired_patient_ci_high': float(ci[1]) if ci else None, 'scope': 'fixed fitted models/seed; descriptive, no multiple-testing correction'})
    write_rows(out / 'matched_comparisons.csv', contrasts)
    interaction = []
    for seed in sorted({seed for _, seed in results}):
        names = ['reference_100hz', 'tiny_100hz', 'reference_50hz', 'tiny_50hz']
        if not all(((name, seed) in results for name in names)):
            continue
        for metric in ['auroc', 'auprc']:
            if all((results[name, seed]['test'] is not None for name in names)):
                values = [results[name, seed]['test']['primary'][metric] for name in names]
                interaction.append({'seed': seed, 'metric': metric, 'width_gap_at_100hz_minus_width_gap_at_50hz': values[0] - values[1] - values[2] + values[3], 'scope': 'descriptive factorial interaction'})
    write_rows(out / 'factorial_interaction.csv', interaction)
    write_json(out / 'report_scope.json', {'completed_runs': len(records), 'planned_runs': len(plan), 'test_evaluated': any((r['test_evaluated'] for r in records)), 'parameter_selection': 'selection subset only; no claim that 419 parameters is optimal', 'calibration': 'disjoint probability-calibration subset; not prospective clinical validation', 'wearable': 'single-lead clinical retraining and synthetic stress tests only', 'hardware': 'CPU forward latency and optional host INT8 serialization, no MCU power or peak RAM', 'literature': 'separate from matched-protocol experiments'})

def parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--train-script', type=Path)
    p.add_argument('--suite', nargs='+', choices=['matched', 'all', 'core', 'width', 'ablation', 'wearable'], default=['matched'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--baseline-config', type=Path)
    p.add_argument('--architecture-only', action='store_true')
    p.add_argument('--benchmark-sessions', type=int, default=5)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--keep-going', action='store_true')
    p.add_argument('--skip-test', action='store_true')
    p.add_argument('--no-robustness', action='store_true')
    p.add_argument('--no-quantization', action='store_true')
    p.add_argument('--plots', action='store_true')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--dropout', type=float, default=0.15)
    p.add_argument('--split-seed', type=int, default=2026)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--paired-bootstrap', type=int, default=1000)
    p.add_argument('--calibration', choices=['platt', 'none'], default='platt')
    p.add_argument('--threshold-method', choices=['youden', 'sensitivity', 'fixed'], default='youden')
    p.add_argument('--target-sensitivity', type=float, default=0.9)
    p.add_argument('--threshold-bootstrap', type=int, default=200)
    p.add_argument('--noise-snr', nargs='+', type=float, default=[20.0, 10.0, 0.0])
    p.add_argument('--noise-repeats', type=int, default=3)
    p.add_argument('--noise-seed', type=int, default=7919)
    p.add_argument('--latency-warmup', type=int, default=50)
    p.add_argument('--latency-repeats', type=int, default=200)
    p.add_argument('--latency-threads', type=int, default=1)
    p.add_argument('--train-threads', type=int, default=0)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    return p

def main(argv=None):
    args = parse_configured(parser(), argv)
    if len(set(args.seeds)) != len(args.seeds) or any((seed < 0 for seed in args.seeds)):
        raise ValueError('Training seeds must be distinct nonnegative integers')
    if args.paired_bootstrap < 0:
        raise ValueError('Paired bootstrap count must be nonnegative')
    args.output_dir = args.output_dir.resolve()
    script = find_trainer(args.train_script)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.architecture_only:
        architecture_benchmark(args, script)
        return
    if args.data_dir is None:
        raise ValueError('--data-dir is required unless --architecture-only is selected')
    args.data_dir = args.data_dir.resolve()
    plan = build_plan(args, script)
    metadata_path = args.data_dir / 'ptbxl_database.csv'
    manifest = {'schema_version': 3, 'train_source_sha256': hash_file(script), 'runner_source_sha256': hash_file(Path(__file__)), 'package_sha256': source_fingerprint(), 'metadata_sha256': hash_file(metadata_path) if metadata_path.is_file() else None, 'baseline_config_sha256': hash_file(args.baseline_config) if args.baseline_config else None, 'paired_bootstrap': args.paired_bootstrap, 'plan': plan}
    if args.dry_run:
        write_json(args.output_dir / 'study_plan.json', manifest)
        literature_context(args.output_dir, bool(args.baseline_config))
        print(f'Planned {len(plan)} runs; no training or test evaluation performed.')
        return
    if not metadata_path.is_file():
        raise FileNotFoundError(f'PTB-XL metadata not found: {metadata_path}')
    manifest_path = args.output_dir / 'study_manifest.json'
    if manifest_path.is_file():
        if not args.resume:
            raise FileExistsError('Study already exists; use --resume for an identical plan')
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Study configuration, code, adapter or dataset metadata changed; use a new directory')
    else:
        write_json(manifest_path, manifest)
    literature_context(args.output_dir, bool(args.baseline_config))
    statuses, failed = ([], False)
    for index, item in enumerate(plan, 1):
        run_dir = Path(item['run_dir'])
        run_dir.mkdir(parents=True, exist_ok=True)
        request_path = run_dir / 'run_request.json'
        if request_path.is_file() and json.loads(request_path.read_text()) != item:
            raise ValueError(f'Run request changed: {run_dir}')
        write_json(request_path, item)
        result_path = run_dir / 'result.json'
        if args.resume and result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get('status') == 'complete':
                statuses.append({'experiment': item['experiment'], 'seed': item['seed'], 'status': 'resumed_complete'})
                write_json(args.output_dir / 'study_status.json', statuses)
                continue
        print(f"[{index}/{len(plan)}] {item['experiment']} seed {item['seed']}", flush=True)
        try:
            with open(run_dir / 'run.log', 'w') as log:
                subprocess.run(item['command'], check=True, stdout=log, stderr=subprocess.STDOUT)
            if not result_path.is_file():
                raise RuntimeError('Trainer exited without result.json')
            statuses.append({'experiment': item['experiment'], 'seed': item['seed'], 'status': 'complete'})
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            failed = True
            statuses.append({'experiment': item['experiment'], 'seed': item['seed'], 'status': 'failed', 'error': str(exc), 'log': str(run_dir / 'run.log')})
            write_json(args.output_dir / 'study_status.json', statuses)
            if not args.keep_going:
                aggregate(args.output_dir, plan, args.paired_bootstrap)
                raise
        write_json(args.output_dir / 'study_status.json', statuses)
    aggregate(args.output_dir, plan, args.paired_bootstrap)
    print(f'Reports saved in {args.output_dir}; {len(statuses)} runs recorded.')
    if failed:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
