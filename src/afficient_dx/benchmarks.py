from afficient_dx.configuration import parse_configured
from afficient_dx.utils import source_fingerprint
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
try:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from afficient_dx.cache import CachedECGs, fit_rr_normalization, make_loader, normalize_rr, prepare_role, to_device
    from afficient_dx.models.published import DISPLAY_NAMES, MODEL_NAMES, forward_record, make_model, parameter_table
    import afficient_dx.api as study
except ModuleNotFoundError as exc:
    raise SystemExit(f'Missing {exc.name}. Use the same Python environment as your earlier run.\nIf needed: python3 -m pip install -r requirements.txt') from exc
ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 2

def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)

def json_read(path):
    return json.loads(Path(path).read_text())

def acquire_lock(path):
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            saved = path.read_text().strip()
            try:
                pid = int(saved)
                if pid < 1:
                    raise ValueError
            except ValueError:
                raise ValueError('The run lock is incomplete. Remove RUNNING.lock from the results folder only after closing any other copy of this script.')
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                path.unlink(missing_ok=True)
                continue
            except PermissionError:
                pass
            raise ValueError(f'Another run is active (process {pid}); keep its Terminal open.')
        else:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
    raise RuntimeError('Could not acquire the run lock')

def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def dataset_signature(data_dir, frames):
    digest = hashlib.sha256()
    for name in ('ptbxl_database.csv',):
        digest.update(study.sha256(data_dir / name).encode())
    for frame in frames:
        for filename in frame.filename_lr:
            base = data_dir / filename
            for suffix in ('.hea', '.dat'):
                path = base.with_suffix(suffix)
                if not path.is_file():
                    raise FileNotFoundError(f'Missing ECG file: {path}\nUse the folder containing ptbxl_database.csv and records100.')
                stat = path.stat()
                digest.update(f'{filename}{suffix}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    return digest.hexdigest()

def find_data_dir(explicit):
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        candidates = [ROOT.parent / 'data', ROOT / 'data', Path.cwd() / 'data']
    found = []
    for candidate in candidates:
        if (candidate / 'ptbxl_database.csv').is_file():
            found.append(candidate.resolve())
        if candidate.is_dir():
            found.extend((p.parent.resolve() for p in candidate.glob('*/ptbxl_database.csv')))
            found.extend((p.parent.resolve() for p in candidate.glob('*/*/ptbxl_database.csv')))
    found = list(dict.fromkeys(found))
    if len(found) == 1:
        return found[0]
    if not found:
        raise ValueError('PTB-XL not found; use --data-dir with the folder containing ptbxl_database.csv and records100')
    raise ValueError("Found multiple datasets. Choose one using --data-dir '/path/to/PTB-XL'.")

def build_protocol(args, data_dir, signature):
    return {'schema_version': SCHEMA, 'purpose': 'same-protocol short-budget architecture comparison', 'data_dir': str(data_dir), 'dataset_signature': signature, 'metadata_sha256': study.sha256(data_dir / 'ptbxl_database.csv'), 'source_sha256': source_fingerprint(), 'model_names': list(MODEL_NAMES), 'seeds': args.seeds, 'sampling_rate_hz': 100, 'leads': study.LEADS, 'record_length': 1000, 'train_folds': list(range(1, 9)), 'validation_fold': 9, 'test_fold': 10, 'split_seed': args.split_seed, 'validation_roles': {'selection': 0.5, 'calibration': 0.25, 'threshold': 0.25}, 'epochs_max': args.epochs, 'early_stopping_patience': args.patience, 'checkpoint_metric': 'selection AUROC', 'batch_size': args.batch_size, 'optimizer': 'AdamW', 'learning_rate': 0.001, 'weight_decay': 0.0001, 'positive_class_weight': 'training negatives / training positives', 'gradient_clip_norm': 5, 'lr_schedule': 'ReduceLROnPlateau: factor .5, patience 2, min 1e-5', 'calibration': 'unweighted positive-slope Platt scaling on calibration patients only', 'threshold': 'Youden J on separate threshold patients, highest-threshold tie break', 'threshold_bootstrap': args.threshold_bootstrap, 'test_bootstrap': args.bootstrap, 'bootstrap_method': 'paired patient-cluster percentile; average metric across paired seeds', 'calibration_bins': 10, 'device_policy': args.device, 'threads': args.threads, 'target_training_minutes': args.minutes, 'timing_budget_rule': 'fixed prespecified epoch limit' if args.fixed_budget else 'effective epochs = min(maximum, max(4, floor(target/suite_epoch_estimate)))', 'comparison_scope': 'adapted architectures under the common budget; not original-task reproduction', 'smaller_model_scope': 'prespecified (1,1,1) versus (1,2,4) channels; not a global optimum search', 'original_study_results': 'preserved; not pooled with this short-budget comparison', 'fixed_epoch_budget': args.fixed_budget}

def guard_protocol(out, protocol):
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'protocol.json'
    if path.exists():
        if json_read(path) != protocol:
            raise ValueError('The dataset, code, or settings changed since this run began. Use a new --output-dir; completed/partial results will not be mixed.')
    elif any(out.iterdir()):
        raise ValueError('Output folder is nonempty without a protocol.json. Use a new --output-dir.')
    else:
        study.write_json(path, protocol)

def predict(model, loader, device):
    model.eval()
    labels, logits = ([], [])
    with torch.inference_mode():
        for x, y, aux in loader:
            x, y, aux = to_device(x, y, aux, device)
            z = forward_record(model, x, aux)
            if z.shape != y.shape or not bool(torch.isfinite(z).all()):
                raise ValueError('Expected one finite logit per ECG')
            labels.append(y.cpu().numpy())
            logits.append(z.cpu().numpy())
    return (np.concatenate(labels), np.concatenate(logits).astype(np.float64))

def rng_state(loader, device):
    state = np.random.get_state()
    result = {'python': random.getstate(), 'numpy_name': state[0], 'numpy_keys': torch.from_numpy(state[1].astype(np.int64)), 'numpy_position': int(state[2]), 'numpy_has_gauss': int(state[3]), 'numpy_cached_gaussian': float(state[4]), 'torch': torch.get_rng_state(), 'loader': loader.generator.get_state()}
    if device.type == 'mps':
        result['mps'] = torch.mps.get_rng_state()
    return result

def restore_rng(state, loader, device):
    random.setstate(state['python'])
    np.random.set_state((state['numpy_name'], state['numpy_keys'].numpy().astype(np.uint32), state['numpy_position'], state['numpy_has_gauss'], state['numpy_cached_gaussian']))
    torch.set_rng_state(state['torch'].cpu())
    loader.generator.set_state(state['loader'].cpu())
    if device.type == 'mps':
        torch.mps.set_rng_state(state['mps'].cpu())

def weights_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

def export_parameters(model, out):
    table = pd.DataFrame(parameter_table(model))
    table.to_csv(out / 'parameter_tensors.csv', index=False)
    layers = table.groupby(['layer', 'type'], sort=False, as_index=False).agg(parameters=('parameters', 'sum'))
    layers.to_csv(out / 'layers.csv', index=False)
    return {'parameters': int(table.parameters.sum()), 'parameters_used_in_forward': int(table.loc[table.used_in_forward, 'parameters'].sum()), 'fp32_parameter_kB_decimal': float(table.parameters.sum() * 4 / 1000)}

def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()

def plan_execution(datasets, args, out, base_hash):
    path = out / 'execution_plan.json'
    if path.exists():
        plan = json_read(path)
        if plan['base_protocol_hash'] != base_hash:
            raise ValueError('Execution plan differs from the declared protocol')
        device = torch.device(plan['device'])
        if device.type == 'mps' and (not torch.backends.mps.is_available()):
            raise ValueError('This saved run requires MPS. Resume on the original Mac.')
        return (plan, device)
    preferred = 'mps' if args.device == 'auto' and torch.backends.mps.is_available() else args.device
    device = torch.device('cpu' if preferred == 'auto' else preferred)
    if args.skip_runtime_estimate:
        plan = {'base_protocol_hash': base_hash, 'device': str(device), 'epochs': args.epochs, 'speed_measurement_skipped': True, 'target_training_minutes': args.minutes}
        study.write_json(path, plan)
        return (plan, device)
    batch = next(iter(make_loader(datasets['train'], args.batch_size, shuffle=True, seed=731)))
    fallback_reason = None

    def measure(chosen):
        x, y, aux = to_device(*batch, chosen)
        estimates = []
        print(f"Measuring this computer's training speed on {chosen}...", flush=True)
        for name in MODEL_NAMES:
            study.set_seed(731, chosen)
            model = make_model(name).to(chosen)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            criterion = torch.nn.BCEWithLogitsLoss()
            elapsed = []
            for attempt in range(4):
                sync(chosen)
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                criterion(forward_record(model, x, aux), y).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                sync(chosen)
                elapsed.append(time.perf_counter() - start)
            seconds = float(np.median(elapsed[1:]))
            epoch_suite = seconds * math.ceil(len(datasets['train']) / args.batch_size) * len(args.seeds) * 1.3
            estimates.append({'model': name, 'seconds_per_training_batch': seconds, 'estimated_one_epoch_all_seeds_seconds': epoch_suite})
            print(f'  {DISPLAY_NAMES[name]}: {seconds:.3f}s per training batch', flush=True)
        return estimates
    try:
        estimates = measure(device)
    except (RuntimeError, NotImplementedError) as exc:
        if args.device != 'auto' or device.type != 'mps':
            raise
        fallback_reason = str(exc)
        device = torch.device('cpu')
        print('An acceleration operation was unsupported; using CPU for all four models.', flush=True)
        estimates = measure(device)
    epoch_seconds = sum((row['estimated_one_epoch_all_seeds_seconds'] for row in estimates))
    epochs = args.epochs if args.fixed_budget else min(args.epochs, max(4, math.floor(args.minutes * 60 / max(epoch_seconds, 1e-09))))
    total = epochs * epoch_seconds / 60
    plan = {'base_protocol_hash': base_hash, 'device': str(device), 'epochs': epochs, 'target_training_minutes': args.minutes, 'estimated_training_minutes': total, 'measured': estimates, 'mps_fallback_reason': fallback_reason, 'rule_uses': 'hardware timing only; no validation/test performance', 'scope': 'rough estimate including validation allowance; excludes initial cache and final report generation'}
    study.write_json(path, plan)
    print(f'Shared budget fixed: {epochs} epochs maximum for every model/seed.\nRough training estimate: {total:.0f} minutes (target {args.minutes:g}). This is not a time guarantee.\nCtrl+C safely stops; running the same command resumes from the last completed epoch.\n', flush=True)
    return (plan, device)

def train_and_freeze(name, seed, datasets, frames, normalization, rr_normalization, out, args, device, protocol_hash):
    run = out / 'runs' / name / f'seed_{seed}'
    run.mkdir(parents=True, exist_ok=True)
    frozen_path = run / 'frozen.json'
    if frozen_path.exists():
        frozen = json_read(frozen_path)
        if frozen['protocol_hash'] != protocol_hash:
            raise ValueError('Frozen run protocol mismatch')
        print(f'Already trained: {DISPLAY_NAMES[name]}, seed {seed}', flush=True)
        return
    study.set_seed(seed, device)
    model = make_model(name).to(device)
    parameters = export_parameters(model, run)
    train_loader = make_loader(datasets['train'], args.batch_size, True, seed)
    selection_loader = make_loader(datasets['selection'], args.batch_size)
    train_y = datasets['train'].y
    positive_weight = float((len(train_y) - train_y.sum()) / train_y.sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-05)
    checkpoint = run / 'last_training.pt'
    best_weights = None
    best_auc, best_epoch, stale, last_epoch = (-1.0, 0, 0, 0)
    history, training_seconds = ([], 0.0)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['protocol_hash'] != protocol_hash:
            raise ValueError('Training checkpoint protocol mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        best_weights, best_auc, best_epoch = (saved['best_weights'], saved['best_auc'], saved['best_epoch'])
        stale, last_epoch, history = (saved['stale'], saved['epoch'], saved['history'])
        training_seconds = saved['training_seconds']
        restore_rng(saved['rng'], train_loader, device)
        print(f'Resuming {DISPLAY_NAMES[name]}, seed {seed}, after epoch {last_epoch}', flush=True)
    for epoch in range(last_epoch + 1, args.epochs + 1):
        if stale >= args.patience:
            break
        start = time.perf_counter()
        model.train()
        total = 0.0
        for x, y, aux in train_loader:
            x, y, aux = to_device(x, y, aux, device)
            optimizer.zero_grad(set_to_none=True)
            z = forward_record(model, x, aux)
            if z.shape != y.shape:
                raise ValueError('Model did not return one logit per record')
            loss = loss_fn(z, y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite loss in {name}, seed {seed}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(y)
        ys, zs = predict(model, selection_loader, device)
        selection_auc = float(roc_auc_score(ys, zs))
        selection_ap = float(average_precision_score(ys, zs))
        scheduler.step(selection_auc)
        epoch_seconds = time.perf_counter() - start
        training_seconds += epoch_seconds
        history.append({'epoch': epoch, 'training_loss': total / len(train_y), 'selection_auroc': selection_auc, 'selection_auprc': selection_ap, 'learning_rate': optimizer.param_groups[0]['lr'], 'seconds': epoch_seconds})
        if selection_auc > best_auc:
            best_weights, best_auc, best_epoch, stale = (weights_cpu(model), selection_auc, epoch, 0)
        else:
            stale += 1
        atomic_save({'protocol_hash': protocol_hash, 'model': weights_cpu(model), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'stale': stale, 'best_weights': best_weights, 'best_auc': best_auc, 'best_epoch': best_epoch, 'history': history, 'training_seconds': training_seconds, 'rng': rng_state(train_loader, device)}, checkpoint)
        pd.DataFrame(history).to_csv(run / 'history.csv', index=False)
        print(f'{DISPLAY_NAMES[name]} | seed {seed} | epoch {epoch}/{args.epochs} | validation AUROC {selection_auc:.4f}, AUPRC {selection_ap:.4f} | {epoch_seconds:.1f}s', flush=True)
    if best_weights is None:
        raise RuntimeError('No completed training epoch')
    model.load_state_dict(best_weights)
    atomic_save(best_weights, run / 'weights.pt')
    yc, zc = predict(model, make_loader(datasets['calibration'], args.batch_size), device)
    calibrator = study.fit_calibrator(yc, zc, 'platt')
    yh, zh = predict(model, make_loader(datasets['threshold'], args.batch_size), device)
    ph = study.apply_calibrator(zh, calibrator)
    threshold = study.threshold_from_validation(yh, ph)
    options = SimpleNamespace(threshold_bootstrap=args.threshold_bootstrap, split_seed=args.split_seed, threshold_method='youden', target_sensitivity=0.9)
    stability = study.threshold_stability(yh, ph, frames['threshold'].patient_id.to_numpy(), options)
    study.write_json(run / 'threshold_stability.json', stability)
    study.prediction_frame(frames['calibration'], yc, zc, study.apply_calibrator(zc, calibrator), threshold).to_csv(run / 'calibration_predictions.csv', index=False)
    study.prediction_frame(frames['threshold'], yh, zh, ph, threshold).to_csv(run / 'threshold_predictions.csv', index=False)
    frozen = {'protocol_hash': protocol_hash, 'model': name, 'model_display': DISPLAY_NAMES[name], 'seed': seed, **parameters, 'best_epoch': best_epoch, 'epochs_completed': len(history), 'best_selection_auroc': best_auc, 'training_seconds': training_seconds, 'positive_weight': positive_weight, 'calibrator': calibrator, 'threshold': threshold, 'normalization': normalization, 'rr_normalization': rr_normalization, 'weights_sha256': study.sha256(run / 'weights.pt'), 'test_used_for_fit': False}
    study.write_json(frozen_path, frozen)

def assert_frozen(out, args, protocol_hash):
    runs = []
    for name in MODEL_NAMES:
        for seed in args.seeds:
            run = out / 'runs' / name / f'seed_{seed}'
            if not (run / 'frozen.json').exists():
                raise RuntimeError(f'Test access blocked: {name}, seed {seed} is not frozen')
            frozen = json_read(run / 'frozen.json')
            if frozen['protocol_hash'] != protocol_hash or study.sha256(run / 'weights.pt') != frozen['weights_sha256']:
                raise ValueError('Frozen protocol/weights integrity check failed')
            runs.append((run, frozen))
    return runs

def evaluate_test(out, dataset, test_frame, runs, args, device):
    options = SimpleNamespace(calibration_bins=10, plots=False)
    for run, frozen in runs:
        if (run / 'result.json').exists():
            print(f"Already evaluated: {frozen['model_display']}, seed {frozen['seed']}", flush=True)
            continue
        model = make_model(frozen['model']).to(device)
        model.load_state_dict(torch.load(run / 'weights.pt', map_location='cpu', weights_only=True))
        y, z = predict(model, make_loader(dataset, args.batch_size), device)
        metrics = study.evaluate_predictions(run, 'test', test_frame, y, z, frozen['calibrator'], frozen['threshold'], options, bootstrap=False)
        p = study.apply_calibrator(z, frozen['calibrator'])
        errors = study.prediction_frame(test_frame, y, z, p, frozen['threshold'])
        errors = errors[errors.outcome.isin(['FP', 'FN'])].sort_values('error_confidence', ascending=False)
        errors.to_csv(run / 'test_failure_cases.csv', index=False)
        study.write_json(run / 'result.json', {**frozen, 'status': 'complete', 'test': metrics, 'test_prevalence': float(y.mean())})
        print(f"Test: {frozen['model_display']}, seed {frozen['seed']}: AUROC={metrics['primary']['auroc']:.4f}, AUPRC={metrics['primary']['auprc']:.4f}", flush=True)

def summarize(out, test_frame, runs, args):
    raw_rows, predictions = ([], {name: [] for name in MODEL_NAMES})
    for run, frozen in runs:
        result = json_read(run / 'result.json')
        raw_rows.append({'model': frozen['model'], 'model_display': frozen['model_display'], 'seed': frozen['seed'], 'parameters': frozen['parameters'], 'best_epoch': frozen['best_epoch'], 'epochs_completed': frozen['epochs_completed'], 'training_seconds': frozen['training_seconds'], **result['test']['primary']})
        predicted = pd.read_csv(run / 'test_predictions.csv')
        if not np.array_equal(predicted.ecg_id.to_numpy(), test_frame.index.to_numpy()):
            raise ValueError('Test prediction IDs are not aligned across models')
        predictions[frozen['model']].append(predicted.logit.to_numpy())
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out / 'run_metrics.csv', index=False)
    metrics = ('auroc', 'auprc', 'sensitivity', 'specificity', 'ppv', 'f1', 'brier', 'ece')
    summary_rows = []
    for name in MODEL_NAMES:
        group = raw[raw.model == name]
        row = {'model': name, 'model_display': DISPLAY_NAMES[name], 'parameters': int(group.parameters.iloc[0]), 'frequency_hz': 100, 'n_seeds': len(group), 'epochs_max': args.epochs, 'training_seconds_total': float(group.training_seconds.sum())}
        for metric in metrics:
            row[f'{metric}_mean'] = float(group[metric].mean())
            row[f'{metric}_seed_sd'] = float(group[metric].std(ddof=1)) if len(group) > 1 else None
        summary_rows.append(row)
    y, patients = (test_frame.target.to_numpy(), test_frame.patient_id.to_numpy())
    samples = {name: {metric: [] for metric in ('auroc', 'auprc')} for name in MODEL_NAMES}
    difference_samples = {name: {metric: [] for metric in ('auroc', 'auprc')} for name in MODEL_NAMES[1:]}
    print(f'Calculating {args.bootstrap} paired patient-bootstrap replicates...', flush=True)
    for replicate, idx in enumerate(study.cluster_resamples(y, patients, args.bootstrap, args.split_seed + 4019), 1):
        values = {}
        for name in MODEL_NAMES:
            values[name] = {'auroc': float(np.mean([roc_auc_score(y[idx], z[idx]) for z in predictions[name]])), 'auprc': float(np.mean([average_precision_score(y[idx], z[idx]) for z in predictions[name]]))}
            for metric in ('auroc', 'auprc'):
                samples[name][metric].append(values[name][metric])
        for name in MODEL_NAMES[1:]:
            for metric in ('auroc', 'auprc'):
                difference_samples[name][metric].append(values[MODEL_NAMES[0]][metric] - values[name][metric])
        if replicate % 100 == 0:
            print(f'  bootstrap {replicate}/{args.bootstrap}', flush=True)
    for row in summary_rows:
        for metric in ('auroc', 'auprc'):
            values = samples[row['model']][metric]
            if len(values) != args.bootstrap:
                raise RuntimeError('Too few valid bootstrap replicates to produce the requested intervals')
            row[f'{metric}_patient_ci_lower'], row[f'{metric}_patient_ci_upper'] = np.percentile(values, [2.5, 97.5])
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / 'comparison.csv', index=False)
    difference_rows = []
    for name in MODEL_NAMES[1:]:
        for metric in ('auroc', 'auprc'):
            reference_mean = summary.loc[summary.model == MODEL_NAMES[0], f'{metric}_mean'].iloc[0]
            other_mean = summary.loc[summary.model == name, f'{metric}_mean'].iloc[0]
            low, high = np.percentile(difference_samples[name][metric], [2.5, 97.5])
            difference_rows.append({'reference': MODEL_NAMES[0], 'comparison': name, 'metric': metric, 'reference_minus_comparison': reference_mean - other_mean, 'patient_ci_lower': low, 'patient_ci_upper': high, 'paired_bootstrap_replicates': args.bootstrap, 'scope': 'test-patient sampling conditional on these fitted models; seed variation reported separately'})
    pd.DataFrame(difference_rows).to_csv(out / 'paired_differences.csv', index=False)
    summary[summary.model.isin(MODEL_NAMES[:2])].to_csv(out / 'smaller_model_comparison.csv', index=False)
    create_report(out, summary, args, len(test_frame), int(y.sum()), difference_rows)

def create_report(out, summary, args, n_test, positives, differences):
    lines = ['# Model comparison', '', f'PTB-XL; 12 leads; 100 Hz; up to {args.epochs} epochs; seeds {args.seeds}.', f'Test ECGs: {n_test}; AF-positive ECGs: {positives}.', '', '| Model | Parameters | AUROC, mean (SD) | AUPRC, mean (SD) |', '|---|---:|---:|---:|']
    def value(mean, sd):
        return f'{mean:.4f} ({sd:.4f})' if sd is not None and np.isfinite(sd) else f'{mean:.4f}'
    for row in summary.to_dict('records'):
        lines.append(f"| {row['model_display']} | {row['parameters']:,} | {value(row['auroc_mean'], row['auroc_seed_sd'])} | {value(row['auprc_mean'], row['auprc_seed_sd'])} |")
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def zip_results(out):
    target = out / 'run_bundle.zip'
    temporary = out / 'run_bundle.partial.zip'
    package = Path(__file__).resolve().parent
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if not path.is_file() or path in (target, temporary) or path.name == 'RUNNING.lock':
                continue
            if 'cache' in path.relative_to(out).parts or path.name.startswith('last_training.') or path.suffix == '.tmp':
                continue
            archive.write(path, str(Path('results') / path.relative_to(out)))
        for path in sorted(package.rglob('*.py')):
            archive.write(path, str(Path('src/afficient_dx') / path.relative_to(package)))
        for name in ('README.md', 'pyproject.toml', 'requirements.txt', 'requirements-recorded.txt', 'CITATION.cff', 'LICENSE'):
            if (ROOT / name).is_file():
                archive.write(ROOT / name, name)
        for folder in ('docs', 'configs', 'LICENSES', 'scripts'):
            if (ROOT / folder).is_dir():
                for path in sorted((ROOT / folder).rglob('*')):
                    if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                        archive.write(path, str(path.relative_to(ROOT)))
    temporary.replace(target)

def parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-dir')
    p.add_argument('--output-dir', type=Path, default=Path.cwd() / 'outputs/benchmark_100hz')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--split-seed', type=int, default=2026)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    p.add_argument('--minutes', type=float, default=20.0)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--threshold-bootstrap', type=int, default=200)
    p.add_argument('--skip-runtime-estimate', action='store_true')
    p.add_argument('--auto-budget', dest='fixed_budget', action='store_false')
    p.set_defaults(fixed_budget=True)
    return p

def main(argv=None):
    args = parse_configured(parser(), argv)
    if any((value < 1 for value in (args.epochs, args.patience, args.batch_size, args.threads, args.bootstrap, args.threshold_bootstrap, args.minutes))) or not args.seeds or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Epochs/patience/batch/threads/bootstrap must be positive; seeds must be unique')
    if args.device == 'mps' and (not torch.backends.mps.is_available()):
        raise ValueError('MPS is unavailable. Omit --device mps to use CPU.')
    torch.set_num_threads(args.threads)
    data_dir, out = (find_data_dir(args.data_dir), args.output_dir.expanduser().resolve())
    train_frame, validation_frame, test_frame = study.load_metadata(data_dir)
    frames = {'train': train_frame, **study.partition_validation(validation_frame, args.split_seed)}
    signature = dataset_signature(data_dir, (train_frame, validation_frame, test_frame))
    protocol = build_protocol(args, data_dir, signature)
    base_hash = hash_json(protocol)
    guard_protocol(out, protocol)
    lock = out / 'RUNNING.lock'
    acquire_lock(lock)
    try:
        print(f'AFficient-Dx: requested fast checks\nDataset: {data_dir}\nResults: {out}\nFour models × {len(args.seeds)} seeds; maximum {args.epochs} epochs each.\n', flush=True)
        study.write_json(out / 'environment.json', study.runtime_metadata())
        study.write_json(out / 'command.json', [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
        rows = [frame.assign(role=role).reset_index()[['ecg_id', 'patient_id', 'strat_fold', 'role', 'target']] for role, frame in {**frames, 'test': test_frame}.items()]
        pd.concat(rows).to_csv(out / 'split_manifest.csv', index=False)
        cache_dir = out / 'cache'
        train_x, train_beats, train_manifest = prepare_role(cache_dir, 'train', train_frame, data_dir)
        normalization = train_manifest['fitted_normalization']
        rr_normalization = fit_rr_normalization(train_beats)
        np.savez(out / 'normalization.npz', mean=np.asarray(normalization['mean']), std=np.asarray(normalization['std']), leads=np.asarray(study.LEADS))
        study.write_json(out / 'rr_normalization.json', rr_normalization)
        datasets = {'train': CachedECGs(train_x, train_frame, normalize_rr(train_beats, rr_normalization))}
        detector_reports = {'train': {key: value for key, value in train_manifest.items() if key != 'ecg_ids'}}
        for role in ('selection', 'calibration', 'threshold'):
            x, beats, manifest = prepare_role(cache_dir, role, frames[role], data_dir, normalization)
            datasets[role] = CachedECGs(x, frames[role], normalize_rr(beats, rr_normalization))
            detector_reports[role] = {key: value for key, value in manifest.items() if key != 'ecg_ids'}
        execution, device = plan_execution(datasets, args, out, base_hash)
        protocol_hash = hash_json({'protocol': protocol, 'execution_plan': execution})
        args = copy.copy(args)
        args.epochs = execution['epochs']
        for name in MODEL_NAMES:
            for seed in args.seeds:
                train_and_freeze(name, seed, datasets, frames, normalization, rr_normalization, out, args, device, protocol_hash)
        runs = assert_frozen(out, args, protocol_hash)
        print('All models and thresholds frozen. Opening held-out test signals now.', flush=True)
        test_x, test_beats, test_manifest = prepare_role(cache_dir, 'test', test_frame, data_dir, normalization)
        detector_reports['test'] = {key: value for key, value in test_manifest.items() if key != 'ecg_ids'}
        study.write_json(out / 'beat_preprocessing.json', detector_reports)
        test_dataset = CachedECGs(test_x, test_frame, normalize_rr(test_beats, rr_normalization))
        evaluate_test(out, test_dataset, test_frame, runs, args, device)
        summarize(out, test_frame, runs, args)
        study.write_json(out / 'study_status.json', {'status': 'complete', 'protocol_hash': protocol_hash, 'runs_completed': len(runs), 'all_expected_runs_completed': True})
        zip_results(out)
        print(f"\nDONE: {out / 'run_bundle.zip'}", flush=True)
    finally:
        lock.unlink(missing_ok=True)
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped safely. Run the same command to resume from the last completed epoch.', flush=True)
        sys.exit(130)
    except (ValueError, FileNotFoundError, RuntimeError, FloatingPointError) as exc:
        print(f'\nRun stopped: {exc}', file=sys.stderr)
        sys.exit(1)
