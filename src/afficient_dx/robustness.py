import hashlib
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from afficient_dx.data import load_split
from afficient_dx.inference import make_loader, predict_logits
from afficient_dx.metrics import apply_calibrator, scores

def synthetic_noise(shape, kind, rng, frequency=100):
    n, channels, length = shape
    t = np.arange(length)[None, None, :] / frequency
    if kind == 'gaussian':
        return rng.standard_normal(shape).astype(np.float32)
    if kind == 'baseline':
        freq = rng.uniform(0.1, 0.5, (n, channels, 1))
        phase = rng.uniform(0, 2 * np.pi, (n, channels, 1))
        return np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
    if kind == 'motion':
        value = np.zeros(shape, dtype=np.float32)
        for _ in range(4):
            center = rng.uniform(0, length / frequency, (n, channels, 1))
            width = rng.uniform(0.04, 0.3, (n, channels, 1))
            amplitude = rng.normal(size=(n, channels, 1))
            value += (amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)).astype(np.float32)
        return value
    raise ValueError(kind)

def add_noise_at_snr(signal, noise, snr_db):
    signal_ac = signal - signal.mean(axis=-1, keepdims=True)
    noise = noise - noise.mean(axis=-1, keepdims=True)
    signal_power = np.mean(signal_ac.astype(np.float64) ** 2, axis=-1, keepdims=True)
    noise_power = np.mean(noise.astype(np.float64) ** 2, axis=-1, keepdims=True)
    scale = np.sqrt(signal_power / np.maximum(noise_power, 1e-20) / 10 ** (snr_db / 10))
    return (signal + scale * noise).astype(np.float32)

def robustness_evaluation(model, df, data_dir, mean, std, calibration, threshold, device, args, out):
    raw100, y = load_split(df, data_dir, 100, args.leads)
    rows = []
    for kind in ['gaussian', 'baseline', 'motion']:
        for snr in args.noise_snr:
            for repeat in range(args.noise_repeats):
                key = f'{kind}:{snr}:{repeat}:{args.noise_seed}'
                rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))
                noisy = add_noise_at_snr(raw100, synthetic_noise(raw100.shape, kind, rng), snr)
                if args.frequency == 50:
                    noisy = resample_poly(noisy, 1, 2, axis=-1, padtype='line').astype(np.float32)
                normalized = ((noisy - mean) / std).astype(np.float32)
                _, z = predict_logits(model, make_loader(normalized, y, args.batch_size), device)
                probability = apply_calibrator(z, calibration)
                rows.append({'condition': kind, 'snr_db_at_100hz': snr, 'repeat': repeat, 'noise_seed': args.noise_seed, 'kind': 'synthetic_stress_test', **scores(y, probability, threshold, args.calibration_bins, z)})
    if len(args.leads) > 1:
        clean = raw100 if args.frequency == 100 else resample_poly(raw100, 1, 2, axis=-1, padtype='line')
        clean = ((clean - mean) / std).astype(np.float32)
        for lead in [name for name in ['I', 'II'] if name in args.leads]:
            masked = np.zeros_like(clean)
            position = args.leads.index(lead)
            masked[:, position] = clean[:, position]
            _, z = predict_logits(model, make_loader(masked, y, args.batch_size), device)
            rows.append({'condition': f'retain_{lead}_only', 'repeat': 0, 'kind': 'test_time_lead_ablation_not_single_lead_retraining', **scores(y, apply_calibrator(z, calibration), threshold, args.calibration_bins, z)})
    pd.DataFrame(rows).to_csv(out / 'robustness.csv', index=False)
    return {'conditions': len(rows), 'threshold': threshold, 'threshold_policy': 'clean-data calibrator and threshold remain fixed', 'scope': 'synthetic PTB-XL stress tests; not wearable/external-dataset validation', 'snr_definition': 'per ECG/lead AC signal power divided by noise power at 100 Hz', 'noise_seed': args.noise_seed, 'repeats': args.noise_repeats}
