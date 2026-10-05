import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from afficient_dx.artifacts import load_runs, read_frame, repository_root, rescore, summarize_runs


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(str(value) for value in row) + ' |')
    return '\n'.join(lines)


def formatted_results(summary):
    rows = []
    for row in summary.to_dict('records'):
        values = {'Model': row['model_display'], 'Parameters': f"{row['parameters']:,}"}
        for metric, display in [('auroc', 'AUROC'), ('auprc', 'AUPRC')]:
            values[display] = f"{row[metric + '_mean']:.4f} ± {row[metric + '_seed_sd']:.4f}" if row['n_seeds'] > 1 else f"{row[metric + '_mean']:.4f}"
        rows.append(values)
    return pd.DataFrame(rows)


def generate(matched_dir, benchmark_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(matched_dir, benchmark_dir)
    summary = summarize_runs(runs)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    metrics = []
    resources = []
    calibration = []
    thresholds = []
    failures = []
    robustness = []
    layer_source = None
    for run in runs:
        result, directory = run['result'], run['directory']
        identity = {'scope': run['scope'], 'model': run['model'], 'seed': run['seed']}
        metrics.append({**identity, **rescore(run)})
        for name in ('primary', 'raw_fixed_0_5', 'calibrated_fixed_0_5'):
            if name in result['test']:
                value = result['test'][name]
                calibration.append({**identity, 'evaluation': name,
                                    'brier': value['brier'], 'ece': value['ece'],
                                    'ece_bins': value['ece_bins'], 'threshold': value['threshold']})
        stability = result.get('threshold_stability')
        if stability is None and (directory / 'threshold_stability.json').is_file():
            from afficient_dx.artifacts import read_json
            stability = read_json(directory / 'threshold_stability.json')
        if stability:
            interval = stability['percentile_95_interval'] or [None, None]
            thresholds.append({**identity, 'fitted_threshold': result['test']['primary']['threshold'],
                               'bootstrap_median': stability['median'], 'patient_ci_lower': interval[0],
                               'patient_ci_upper': interval[1], 'valid_replicates': stability['valid']})
        resource = result.get('resources', {})
        resources.append({**identity, 'parameters': result['parameters'],
                          'parameter_fp32_bytes': result['parameters'] * 4,
                          'parameter_fp32_kB': result['parameters'] * 4 / 1000,
                          'macs_conv_linear': result.get('macs'),
                          'serialized_weights_bytes': (directory / 'weights.pt').stat().st_size,
                          'cpu_batch1_median_ms': resource.get('cpu_latency', {}).get('median_ms'),
                          'input_fp32_bytes': resource.get('input_fp32_bytes'),
                          'quantization_run_status': result.get('quantization', {}).get('status', 'not_requested'),
                          'serialized_int8_bytes': (directory / 'model_int8.ts').stat().st_size if (directory / 'model_int8.ts').is_file() else None})
        frame = run['predictions']
        errors = frame[frame.outcome.isin(['FP', 'FN'])].copy()
        errors['error_confidence'] = np.where(errors.y == 1, 1 - errors.probability, errors.probability)
        failures.append(errors.assign(**identity))
        if (directory / 'robustness.csv').is_file():
            robustness.append(read_frame(directory / 'robustness.csv').assign(**identity))
        if run['scope'] == 'matched_100hz' and run['model'] == 'tiny_100hz' and layer_source is None:
            layer_source = directory / 'layers.csv'
    pd.DataFrame(metrics).to_csv(output_dir / 'per_seed_metrics.csv', index=False)
    pd.DataFrame(resources).to_csv(output_dir / 'resources.csv', index=False)
    pd.DataFrame(calibration).to_csv(output_dir / 'calibration.csv', index=False)
    pd.DataFrame(thresholds).to_csv(output_dir / 'threshold_stability.csv', index=False)
    pd.concat(failures, ignore_index=True).sort_values('error_confidence', ascending=False).to_csv(output_dir / 'failure_cases.csv', index=False)
    sections = ['# Reproduced study results', '', 'AUROC and AUPRC are recalculated from the archived test logits. Values are means ± sample SD across training seeds. AUPRC uses average precision.', '']
    if layer_source:
        layers = read_frame(layer_source)
        layers.to_csv(output_dir / 'table_iii_layers.csv', index=False)
        table = layers[['layer', 'type', 'parameters']].copy()
        table.columns = ['Layer', 'Type', 'Parameters']
        table.loc[len(table)] = ['Total', '', int(layers.parameters.sum())]
        sections.extend(['## Table III. AFficient-Dx layer parameters', '', markdown_table(table), ''])
    for scope, filename, title in (
        ('matched_100hz', 'table_iv_matched_100hz.csv', 'Table IV. AFficient-Dx and reference at 100 Hz'),
        ('benchmark_100hz', 'table_v_benchmark_100hz.csv', 'Table V. Six-epoch comparison at 100 Hz'),
    ):
        group = summary[summary.scope == scope]
        if not group.empty:
            group.to_csv(output_dir / filename, index=False)
            sections.extend(['## ' + title, '', markdown_table(formatted_results(group)), ''])
    confusion = summary[['scope', 'model', 'n_seeds', 'tn_mean', 'fp_mean', 'fn_mean', 'tp_mean']]
    confusion.to_csv(output_dir / 'confusion_matrices.csv', index=False)
    if robustness:
        raw = pd.concat(robustness, ignore_index=True)
        raw.to_csv(output_dir / 'robustness_per_seed.csv', index=False)
        keys = ['scope', 'model', 'seed', 'condition', 'snr_db_at_100hz', 'kind']
        per_seed = raw.groupby(keys, dropna=False, as_index=False)[['auroc', 'auprc']].mean()
        combined = per_seed.groupby([key for key in keys if key != 'seed'], dropna=False).agg(
            auroc_mean=('auroc', 'mean'), auroc_seed_sd=('auroc', 'std'),
            auprc_mean=('auprc', 'mean'), auprc_seed_sd=('auprc', 'std'), n_seeds=('seed', 'nunique')).reset_index()
        combined.to_csv(output_dir / 'robustness_summary.csv', index=False)
    paired = Path(benchmark_dir) / 'paired_differences.csv'
    if paired.is_file():
        read_frame(paired).to_csv(output_dir / 'paired_differences.csv', index=False)
    sections.extend(['## Resource measurements', '', 'Weight storage uses decimal units: 1 kB = 1,000 bytes. CPU latency is the batch-1 forward pass measured in the archived environment. Serialized INT8 file sizes include export overhead; they do not measure runtime RAM or power. The original quantization result status is retained in resources.csv.', ''])
    (output_dir / 'TABLES.md').write_text('\n'.join(sections), encoding='utf-8')
    return summary


def main(argv=None):
    root = repository_root()
    parser = argparse.ArgumentParser()
    parser.add_argument('--matched-dir', type=Path, default=root / 'artifacts/matched_100hz')
    parser.add_argument('--benchmark-dir', type=Path, default=root / 'artifacts/benchmark_100hz')
    parser.add_argument('--output-dir', type=Path, default=root / 'results/reproduced')
    args = parser.parse_args(argv)
    summary = generate(args.matched_dir, args.benchmark_dir, args.output_dir)
    print(f'{len(summary)} models: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
