import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from afficient_dx.artifacts import read_json
from afficient_dx.cache import detect_beats, normalize_rr
from afficient_dx.constants import LEADS
from afficient_dx.data import load_record
from afficient_dx.metrics import apply_calibrator
from afficient_dx.models.published import forward_record, make_model as published_model
from afficient_dx.models.residual import make_model as residual_model
from afficient_dx.utils import sha256, write_json


def load_checkpoint(directory):
    directory = Path(directory)
    weights = directory / 'weights.pt'
    if (directory / 'inference.json').is_file():
        metadata = read_json(directory / 'inference.json')
        arguments = SimpleNamespace(**metadata['configuration']['arguments'])
        model = residual_model(arguments, metadata['input_length'])
        metadata = {**metadata, 'model': metadata['configuration']['arguments'].get('model_name', 'AFficient-Dx'),
                    'calibrator': metadata['calibration'],
                    'normalization': {'mean': metadata['normalization_mean'],
                                      'std': metadata['normalization_std']}}
    elif (directory / 'frozen.json').is_file():
        metadata = read_json(directory / 'frozen.json')
        if sha256(weights) != metadata['weights_sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {weights}')
        model = published_model(metadata['model'])
        metadata = {**metadata, 'frequency_hz': 100, 'leads': LEADS}
    else:
        raise FileNotFoundError(f'Inference metadata missing: {directory}')
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    return model.eval(), metadata


def predict_record(checkpoint_dir, record):
    model, metadata = load_checkpoint(checkpoint_dir)
    raw = load_record(Path(record).expanduser(), metadata['frequency_hz'], metadata['leads'])
    mean = np.asarray(metadata['normalization']['mean'], dtype=np.float32)[:, None]
    std = np.asarray(metadata['normalization']['std'], dtype=np.float32)[:, None]
    if mean.shape != (len(metadata['leads']), 1) or std.shape != mean.shape or (std <= 0).any():
        raise ValueError('Invalid checkpoint normalization')
    x = torch.from_numpy(np.ascontiguousarray((raw - mean) / std))[None, ...]
    aux = {}
    if metadata['model'] == 'busia_transformer':
        peaks, intervals, fallback = detect_beats(raw[metadata['leads'].index('II')])
        normalized = normalize_rr({'rr_seconds': intervals}, metadata['rr_normalization'])
        aux = {'record_index': torch.zeros(len(peaks), dtype=torch.int64),
               'peak_position': torch.from_numpy(peaks), 'rr': torch.from_numpy(normalized['rr'])}
    with torch.inference_mode():
        logit = forward_record(model, x, aux).detach().cpu().numpy().reshape(-1)
    if logit.shape != (1,) or not np.isfinite(logit).all():
        raise ValueError('Expected one finite record logit')
    probability = float(apply_calibrator(logit, metadata['calibrator'])[0])
    threshold = float(metadata['threshold'])
    result = {'record': str(record), 'model': metadata['model'],
              'frequency_hz': metadata['frequency_hz'], 'logit': float(logit[0]),
              'af_probability': probability, 'threshold': threshold,
              'af_prediction': int(probability >= threshold)}
    if metadata['model'] == 'busia_transformer':
        result.update(detected_beats=len(peaks), beat_detector_fallback=bool(fallback))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(args.threads)
    result = predict_record(args.checkpoint_dir, args.record)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
