import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import butter, find_peaks, sosfiltfilt
from torch.utils.data import DataLoader, Dataset
from afficient_dx.constants import LEADS
from afficient_dx.data import load_record
from afficient_dx.utils import write_json
_FILTER = butter(2, [5, 18], btype='bandpass', fs=100, output='sos')

def detect_beats(lead):
    lead = np.asarray(lead, dtype=np.float64)
    filtered = sosfiltfilt(_FILTER, lead)
    energy = np.convolve(np.diff(filtered, prepend=filtered[0]) ** 2, np.ones(12) / 12, mode='same')
    scale = float(np.quantile(energy, 0.95) - np.median(energy))
    if scale <= 1e-14:
        peaks = np.empty(0, dtype=int)
    else:
        candidates, _ = find_peaks(energy, distance=25, prominence=0.12 * scale, height=np.median(energy) + 0.12 * scale)
        refined = []
        for candidate in candidates:
            lo, hi = (max(0, candidate - 8), min(len(lead), candidate + 9))
            refined.append(lo + int(np.argmax(np.abs(filtered[lo:hi]))))
        selected = []
        for peak in sorted(set(refined)):
            if selected and peak - selected[-1] < 25:
                if abs(filtered[peak]) > abs(filtered[selected[-1]]):
                    selected[-1] = peak
            else:
                selected.append(peak)
        peaks = np.asarray(selected, dtype=int)
    fallback = len(peaks) == 0
    if fallback:
        peaks = np.array([len(lead) // 2])
    intervals = np.diff(peaks) / 100.0
    typical = float(np.median(intervals)) if len(intervals) else 1.0
    previous = np.r_[typical, intervals]
    following = np.r_[intervals, typical]
    return (peaks.astype(np.float32), np.column_stack([previous, following]).astype(np.float32), fallback)

def prepare_role(cache_dir, role, frame, data_dir, normalization=None):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    done = cache_dir / f'{role}_complete.json'
    array_path, beats_path = (cache_dir / f'{role}.npy', cache_dir / f'{role}_beats.npz')
    ids = frame.index.to_list()
    if done.exists():
        manifest = json.loads(done.read_text())
        if manifest['ecg_ids'] != ids or manifest['normalization'] != normalization:
            raise ValueError(f'{role} cache differs from the declared split/normalization')
        x = np.load(array_path, mmap_mode='r')
        with np.load(beats_path) as saved:
            beats = {key: saved[key].copy() for key in saved.files}
        if x.shape != (len(frame), 12, 1000) or x.dtype != np.float32:
            raise ValueError(f'Invalid {role} signal cache; remove its *_complete.json and rerun')
        return (x, beats, manifest)
    t = time.perf_counter()
    print(f'Caching {role}: {len(frame):,} ECGs (one-time step)...', flush=True)
    temporary = cache_dir / f'{role}.partial.npy'
    x = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(len(frame), 12, 1000))
    mean, std = (None, None)
    if normalization is not None:
        mean = np.asarray(normalization['mean'], dtype=np.float32)[:, None]
        std = np.asarray(normalization['std'], dtype=np.float32)[:, None]
    offsets, positions, rr, fallback_count = ([0], [], [], 0)
    for i, filename in enumerate(frame.filename_lr):
        signal = load_record(data_dir / filename, 100, LEADS)
        peaks, intervals, fallback = detect_beats(signal[1])
        positions.append(peaks)
        rr.append(intervals)
        offsets.append(offsets[-1] + len(peaks))
        fallback_count += int(fallback)
        x[i] = signal if mean is None else (signal - mean) / std
        if (i + 1) % 1000 == 0:
            print(f'  {role}: {i + 1:,}/{len(frame):,}', flush=True)
    fitted_normalization = normalization
    if normalization is None:
        total, squares = (np.zeros(12), np.zeros(12))
        for start in range(0, len(x), 128):
            block = np.asarray(x[start:start + 128], dtype=np.float64)
            total += block.sum(axis=(0, 2))
            squares += (block * block).sum(axis=(0, 2))
        n = len(x) * 1000
        mean = (total / n).astype(np.float32)[:, None]
        std = np.maximum(np.sqrt(np.maximum(squares / n - (total / n) ** 2, 0)), 1e-06)
        std = std.astype(np.float32)[:, None]
        for start in range(0, len(x), 128):
            x[start:start + 128] = (x[start:start + 128] - mean[None, ...]) / std[None, ...]
        fitted_normalization = {'mean': mean.ravel().tolist(), 'std': std.ravel().tolist()}
    x.flush()
    del x
    temporary.replace(array_path)
    beats = {'indptr': np.asarray(offsets, dtype=np.int64), 'peak_position': np.concatenate(positions), 'rr_seconds': np.concatenate(rr)}
    with open(beats_path.with_suffix('.tmp'), 'wb') as handle:
        np.savez(handle, **beats)
    beats_path.with_suffix('.tmp').replace(beats_path)
    counts = np.diff(beats['indptr'])
    manifest = {'role': role, 'ecg_ids': ids, 'normalization': normalization, 'fitted_normalization': fitted_normalization, 'beat_detector': 'fixed_energy_detector_v1_lead_II_100Hz', 'n_records': len(frame), 'n_beats': int(counts.sum()), 'median_beats_per_record': float(np.median(counts)), 'minimum_beats_per_record': int(counts.min()), 'maximum_beats_per_record': int(counts.max()), 'fallback_records': fallback_count, 'seconds': time.perf_counter() - t}
    write_json(done, manifest)
    return (np.load(array_path, mmap_mode='r'), beats, manifest)

def fit_rr_normalization(train_beats):
    values = train_beats['rr_seconds'].astype(np.float64)
    low, high = (values.min(axis=0), values.max(axis=0))
    return {'minimum_seconds': low.tolist(), 'maximum_seconds': high.tolist(), 'method': 'train-only min-max to [-2,2], evaluation clipped to training range'}

def normalize_rr(beats, normalization):
    low = np.asarray(normalization['minimum_seconds'], dtype=np.float32)
    high = np.asarray(normalization['maximum_seconds'], dtype=np.float32)
    span = high - low
    result = np.where(span > 1e-08, 4 * (beats['rr_seconds'] - low) / np.maximum(span, 1e-08) - 2, 0)
    return {**beats, 'rr': np.clip(result, -2, 2).astype(np.float32)}

class CachedECGs(Dataset):

    def __init__(self, signals, frame, beats):
        self.signals, self.beats = (signals, beats)
        self.y = frame.target.to_numpy(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (torch.from_numpy(self.signals[index].copy()), float(self.y[index]), index)

    def collate(self, records):
        signals, labels, indices = zip(*records)
        record_indices, peaks, rr = ([], [], [])
        for i, index in enumerate(indices):
            start, end = self.beats['indptr'][index:index + 2]
            record_indices.append(np.full(end - start, i, dtype=np.int64))
            peaks.append(self.beats['peak_position'][start:end])
            rr.append(self.beats['rr'][start:end])
        aux = {'record_index': torch.from_numpy(np.concatenate(record_indices)), 'peak_position': torch.from_numpy(np.concatenate(peaks)), 'rr': torch.from_numpy(np.concatenate(rr))}
        return (torch.stack(signals), torch.tensor(labels, dtype=torch.float32), aux)

def make_loader(dataset, batch_size, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=dataset.collate, generator=generator, drop_last=False)

def to_device(x, y, aux, device):
    return (x.to(device), y.to(device), {key: value.to(device) for key, value in aux.items()})
