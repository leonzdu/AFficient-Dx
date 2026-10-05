import argparse
import copy
import importlib
import importlib.metadata
import inspect
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from afficient_dx.configuration import parse_configured
from afficient_dx.constants import SCHEMA_VERSION
from afficient_dx.data import canonical_leads, load_metadata, load_split, partition_validation
from afficient_dx.inference import logits_from_model, make_loader, predict_logits
from afficient_dx.metrics import apply_calibrator, fit_calibrator, threshold_from_validation, threshold_stability
from afficient_dx.models.residual import ECGResNet, make_model
from afficient_dx.profiling import benchmark_cpu, layer_report, runtime_metadata
from afficient_dx.quantization import quantize_and_evaluate
from afficient_dx.reports import evaluate_predictions
from afficient_dx.robustness import robustness_evaluation
from afficient_dx.utils import get_device, jsonable, set_seed, sha256, source_fingerprint, write_json

def parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--base', type=int, default=1)
    p.add_argument('--channels', nargs='+', type=int)
    p.add_argument('--frequency', type=int, choices=[50, 100], required=True)
    p.add_argument('--leads', nargs='+', default=['all'])
    p.add_argument('--no-residual', action='store_true')
    p.add_argument('--no-batchnorm', action='store_true')
    p.add_argument('--no-class-weight', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split-seed', type=int, default=2026)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--dropout', type=float, default=0.15)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--calibration', choices=['platt', 'none'], default='platt')
    p.add_argument('--calibration-bins', type=int, default=10)
    p.add_argument('--threshold-method', choices=['youden', 'sensitivity', 'fixed'], default='youden')
    p.add_argument('--target-sensitivity', type=float, default=0.9)
    p.add_argument('--threshold-bootstrap', type=int, default=200)
    p.add_argument('--shuffle-labels', action='store_true')
    p.add_argument('--skip-test', action='store_true')
    p.add_argument('--robustness', action='store_true')
    p.add_argument('--noise-snr', nargs='+', type=float, default=[20.0, 10.0, 0.0])
    p.add_argument('--noise-repeats', type=int, default=3)
    p.add_argument('--noise-seed', type=int, default=7919)
    p.add_argument('--latency-warmup', type=int, default=50)
    p.add_argument('--latency-repeats', type=int, default=200)
    p.add_argument('--latency-threads', type=int, default=1)
    p.add_argument('--train-threads', type=int, default=0)
    p.add_argument('--quantize-int8', action='store_true')
    p.add_argument('--quantized-engine', choices=['auto', 'x86', 'fbgemm', 'qnnpack', 'onednn'], default='auto')
    p.add_argument('--quantization-records', type=int, default=256)
    p.add_argument('--plots', action='store_true')
    p.add_argument('--model-factory')
    p.add_argument('--model-name', default='AFficient-Dx residual CNN')
    p.add_argument('--model-source')
    p.add_argument('--model-adaptation')
    p.add_argument('--model-kwargs', default='{}')
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--validation-protocol', choices=['split', 'original'], default='split')
    return p

def validate_args(args):
    args.leads = canonical_leads(args.leads)
    if args.validation_protocol == 'original' and args.calibration != 'none':
        raise ValueError('Original validation requires --calibration none')
    for name in ['base', 'epochs', 'patience', 'batch_size', 'calibration_bins', 'noise_repeats', 'latency_threads', 'quantization_records']:
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    for name in ['bootstrap', 'threshold_bootstrap', 'latency_warmup', 'latency_repeats', 'train_threads']:
        if getattr(args, name) < 0:
            raise ValueError(f'{name} must be nonnegative')
    if args.channels and any((c <= 0 for c in args.channels)):
        raise ValueError('Channel widths must be positive integers')
    if not 0 <= args.dropout < 1 or not 0 < args.target_sensitivity <= 1:
        raise ValueError('Invalid dropout or target sensitivity')
    if args.lr <= 0 or args.weight_decay < 0 or (not all(np.isfinite(args.noise_snr))):
        raise ValueError('Invalid optimizer/noise settings')
    if not isinstance(json.loads(args.model_kwargs), dict):
        raise ValueError('--model-kwargs must be a JSON object')
    if args.model_factory and (not (args.model_source and args.model_adaptation and (':' in args.model_factory))):
        raise ValueError('External models require factory, source/revision, and adaptation notes')
    if args.model_factory and (args.channels or args.no_residual or args.no_batchnorm or (args.base != 1)):
        raise ValueError('Built-in architecture switches cannot be applied to external factories')

def main(argv=None):
    args = parse_configured(parser(), argv)
    validate_args(args)
    args.data_dir, args.output_dir = (args.data_dir.resolve(), args.output_dir.resolve())
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise FileExistsError(f'Completed run already exists: {out}; use a new directory')
    device = get_device(args.device)
    if args.train_threads:
        torch.set_num_threads(args.train_threads)
    set_seed(args.seed, device)
    metadata_hash = sha256(args.data_dir / 'ptbxl_database.csv')
    configuration = {'schema_version': SCHEMA_VERSION, 'arguments': jsonable(vars(args)), 'source_sha256': sha256(Path(__file__)), 'package_sha256': source_fingerprint(), 'metadata_sha256': metadata_hash}
    if args.model_factory:
        module = importlib.import_module(args.model_factory.rsplit(':', 1)[0])
        source_file = inspect.getsourcefile(module)
        configuration['adapter_sha256'] = sha256(source_file) if source_file else None
    if (out / 'config.json').exists():
        if json.loads((out / 'config.json').read_text()) != configuration:
            raise ValueError('Partial run configuration changed; use a new output directory')
    write_json(out / 'config.json', configuration)
    write_json(out / 'environment.json', runtime_metadata())
    (out / 'command.json').write_text(json.dumps([sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)], indent=2))
    train_df, validation_df, test_df = load_metadata(args.data_dir)
    frames = {'train': train_df, **(partition_validation(validation_df, args.split_seed) if args.validation_protocol == 'split' else {'selection': validation_df, 'calibration': validation_df, 'threshold': validation_df})}
    split_rows = [part.assign(role=role).reset_index()[['ecg_id', 'patient_id', 'strat_fold', 'role']] for role, part in {**frames, 'test': test_df}.items()]
    pd.concat(split_rows).to_csv(out / 'split_manifest.csv', index=False)
    arrays = {}
    train_x, train_y = load_split(train_df, args.data_dir, args.frequency, args.leads)
    mean = train_x.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.maximum(train_x.std(axis=(0, 2), keepdims=True, dtype=np.float64), 1e-06).astype(np.float32)
    train_x -= mean
    train_x /= std
    if args.shuffle_labels:
        train_y = np.random.default_rng(args.seed).permutation(train_y)
    arrays['train'] = (train_x, train_y)
    for role in ['selection', 'calibration', 'threshold']:
        x, y = load_split(frames[role], args.data_dir, args.frequency, args.leads)
        x -= mean
        x /= std
        arrays[role] = (x, y)
    np.savez(out / 'normalization.npz', mean=mean, std=std, leads=np.asarray(args.leads))
    model = make_model(args, train_x.shape[-1])
    layers = layer_report(model, train_x.shape[-1], len(args.leads))
    pd.DataFrame(layers).to_csv(out / 'layers.csv', index=False)
    pd.DataFrame([{'tensor': name, 'shape': list(p.shape), 'parameters': p.numel(), 'trainable': p.requires_grad, 'bytes': p.numel() * p.element_size()} for name, p in model.named_parameters()]).to_csv(out / 'parameter_tensors.csv', index=False)
    params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    all_params = sum((p.numel() for p in model.parameters()))
    macs = int(sum((row['conv_linear_macs'] for row in layers)))
    model = model.to(device)
    positive_weight = float((len(train_y) - train_y.sum()) / train_y.sum()) if not args.no_class_weight else 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=min(1e-05, args.lr))
    train_loader = make_loader(train_x, train_y, args.batch_size, True, args.seed)
    selection_loader = make_loader(*arrays['selection'], args.batch_size)
    best_auc, best_epoch, stale = (-1.0, 0, 0)
    checkpoint = out / 'weights.pt'
    history = []
    start_training = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = (x.to(device), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(logits_from_model(model, x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.item() * len(x)
        ys, zs = predict_logits(model, selection_loader, device)
        selection_auc = float(roc_auc_score(ys, zs))
        selection_ap = float(average_precision_score(ys, zs))
        scheduler.step(selection_auc)
        history.append({'epoch': epoch, 'training_loss': total / len(train_y), 'selection_auroc': selection_auc, 'selection_auprc': selection_ap, 'lr': optimizer.param_groups[0]['lr']})
        print(f'seed={args.seed} epoch={epoch:02d} selection AUROC={selection_auc:.4f} AP={selection_ap:.4f}', flush=True)
        if selection_auc > best_auc:
            best_auc, best_epoch, stale = (selection_auc, epoch, 0)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            break
    training_seconds = time.perf_counter() - start_training
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    yc, zc = predict_logits(model, make_loader(*arrays['calibration'], args.batch_size), device)
    calibration = fit_calibrator(yc, zc, args.calibration)
    yh, zh = predict_logits(model, make_loader(*arrays['threshold'], args.batch_size), device)
    ph = apply_calibrator(zh, calibration)
    threshold = threshold_from_validation(yh, ph, args.threshold_method, args.target_sensitivity)
    frozen = {'configuration': configuration, 'calibration': calibration, 'threshold': threshold, 'threshold_method': args.threshold_method, 'normalization_mean': mean.flatten().tolist(), 'normalization_std': std.flatten().tolist(), 'leads': args.leads, 'frequency_hz': args.frequency, 'input_length': train_x.shape[-1]}
    write_json(out / 'inference.json', frozen)
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'inference': frozen}, out / 'checkpoint.pt')
    result = {'schema_version': SCHEMA_VERSION, 'status': 'complete', 'configuration': configuration, 'model_name': args.model_name, 'base_channels': args.base if not args.model_factory else None, 'channels': args.channels or [args.base, 2 * args.base, 4 * args.base] if not args.model_factory else None, 'frequency_hz': args.frequency, 'leads': args.leads, 'seed': args.seed, 'shuffle_labels': args.shuffle_labels, 'parameters': int(params), 'macs': macs if not args.model_factory else None, 'counted_conv_linear_macs': macs, 'mac_scope': 'Conv1d and Linear only; excludes BN, nonlinearities, pooling, additions and custom ops', 'best_epoch': best_epoch, 'training_seconds': training_seconds, 'positive_class_weight': positive_weight, 'calibration': calibration, 'threshold_stability': threshold_stability(yh, ph, frames['threshold'].patient_id.to_numpy(), args), 'test': None, 'robustness': {'status': 'not_requested'}, 'quantization': {'status': 'not_requested'}, 'limitations': ['Single clinical dataset; no established wearable generalization', 'Explored widths do not establish a global optimum', 'Test results must not guide architecture or threshold selection']}
    result['validation'] = {}
    for role in ['selection', 'calibration', 'threshold']:
        y, z = predict_logits(model, make_loader(*arrays[role], args.batch_size), device)
        result['validation'][role] = evaluate_predictions(out, role, frames[role], y, z, calibration, threshold, args)
    if not args.skip_test:
        frames['test'] = test_df
        x_test, y_test = load_split(test_df, args.data_dir, args.frequency, args.leads)
        x_test -= mean
        x_test /= std
        arrays['test'] = (x_test, y_test)
        yt, zt = predict_logits(model, make_loader(x_test, y_test, args.batch_size), device)
        result['test'] = evaluate_predictions(out, 'test', test_df, yt, zt, calibration, threshold, args, bootstrap=True)
        if args.robustness:
            result['robustness'] = robustness_evaluation(model, test_df, args.data_dir, mean, std, calibration, threshold, device, args, out)
    elif args.robustness:
        result['robustness'] = {'status': 'skipped_test_withheld'}
    cpu_model = copy.deepcopy(model).cpu().eval()
    result['resources'] = {'units': 'bytes; kB = 1000 bytes', 'parameter_fp32_bytes': all_params * 4, 'parameter_fp32_kB': all_params * 4 / 1000, 'parameter_int8_bytes_theoretical': all_params, 'parameter_int8_estimate_scope': 'weights only; excludes scale/zero-point, buffers and runtime', 'state_tensor_bytes': sum((t.numel() * t.element_size() for t in cpu_model.state_dict().values())), 'serialized_state_dict_bytes': checkpoint.stat().st_size, 'serialized_reproducible_checkpoint_bytes': (out / 'checkpoint.pt').stat().st_size, 'input_fp32_bytes': int(train_x[:1].nbytes), 'peak_runtime_ram_bytes': None, 'microcontroller_latency_ms': None, 'power_mw': None, 'energy_per_inference_mj': None, 'cpu_latency': benchmark_cpu(cpu_model, train_x[:1], args.latency_warmup, args.latency_repeats, args.latency_threads), 'scope': 'Measured host serialization/timing and theoretical parameter storage are separate'}
    if args.quantize_int8:
        try:
            result['quantization'] = quantize_and_evaluate(cpu_model, arrays, frames, mean, std, args, out)
        except Exception as exc:
            result['quantization'] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'note': 'No INT8 performance or size claim is made'}
            print(f'INT8 evaluation failed: {exc}', file=sys.stderr, flush=True)
    write_json(out / 'result.json', result)
    print(json.dumps(jsonable(result['test']['primary'] if result['test'] else result['validation']['selection']['primary']), indent=2))
if __name__ == '__main__':
    main()
