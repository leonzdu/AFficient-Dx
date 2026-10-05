import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import wfdb
import afficient_dx.cache as data_cache
import afficient_dx.benchmarks as runner
from afficient_dx.models.published import BassoPHMultiScopic, BusiaBeatTransformer, BusiaRecordAdapter, MODEL_NAMES, PHConv1d, PHLinear, make_model
from afficient_dx.constants import LEADS

def architectural_checks():
    assert sum((p.numel() for p in make_model('afficient_419').parameters())) == 419
    assert sum((p.numel() for p in make_model('smaller_238').parameters())) == 238
    assert sum((p.numel() for p in BassoPHMultiScopic(4, 1).parameters())) == 61403
    assert sum((p.numel() for p in BusiaBeatTransformer(1, 5).parameters())) == 6643
    conv, linear = (PHConv1d(4, 12, 16, 3), PHLinear(4, 16, 24))
    expected_conv = sum((torch.stack([torch.kron(conv.A[i], conv.F[i, :, :, k]) for k in range(3)], dim=-1) for i in range(4)))
    expected_linear = sum((torch.kron(linear.A[i], linear.S[i]) for i in range(4)))
    torch.testing.assert_close(conv.effective_weight(), expected_conv)
    torch.testing.assert_close(linear.effective_weight(), expected_linear)
    conv(torch.randn(2, 12, 100)).square().mean().backward()
    assert conv.A.grad is not None and conv.F.grad is not None and (conv.bias.grad is None)
    model = BusiaRecordAdapter().eval()
    x = torch.arange(3 * 12 * 1000).float().reshape(3, 12, 1000) / 1000
    peaks = torch.tensor([0.0, 500.0, 999.0, 300.0, 700.0])
    records = torch.tensor([0, 0, 1, 2, 2])
    rr = torch.zeros(5, 2)
    expected_windows = []
    for index, peak in zip(records.tolist(), peaks.tolist()):
        positions = peak + np.arange(-99, 99) * 100 / 360
        window = np.stack([np.interp(positions, np.arange(-1, 1001), np.r_[0.0, channel.numpy(), 0.0], left=0, right=0) for channel in x[index]])
        expected_windows.append(window)
    with torch.inference_mode():
        logits = model.core(torch.tensor(np.stack(expected_windows), dtype=torch.float32), rr).flatten()
        expected = torch.stack([logits[records == i].mean() for i in range(3)])
        actual = model(x, {'record_index': records, 'peak_position': peaks, 'rr': rr})
    torch.testing.assert_close(actual, expected, atol=2e-05, rtol=2e-05)
    regular = np.zeros(1000)
    for peak in range(80, 1000, 80):
        regular += np.exp(-0.5 * ((np.arange(1000) - peak) / 2.0) ** 2)
    detected, intervals, fallback = data_cache.detect_beats(regular)
    assert not fallback and 10 <= len(detected) <= 14
    assert np.allclose(np.median(intervals), 0.8, atol=0.03)
    _, _, fallback = data_cache.detect_beats(np.zeros(1000))
    assert fallback

def synthetic_dataset(root):
    root.mkdir()
    rng = np.random.default_rng(2026)
    rows, ecg_id = ([], 0)
    groups = [('train', 48, 0), ('validation', 40, 100), ('test', 12, 200)]
    for role, patient_count, offset in groups:
        for i in range(patient_count):
            for repetition in range(2 if role == 'test' else 1):
                ecg_id += 1
                positive = int(i % 2)
                positions = [70]
                while positions[-1] < 950:
                    gap = int(rng.integers(45, 100)) if positive else 80
                    positions.append(positions[-1] + gap)
                signal = 0.01 * rng.normal(size=(1000, 12))
                t = np.arange(1000)
                for peak in positions:
                    signal += np.exp(-0.5 * ((t - peak) / 2) ** 2)[:, None] * np.linspace(0.6, 1.3, 12)
                signal += 0.02 * np.sin(t / 70)[:, None]
                name = f'{role}_{ecg_id:05d}'
                wfdb.wrsamp(name, fs=100, units=['mV'] * 12, sig_name=LEADS, p_signal=signal, fmt=['16'] * 12, write_dir=str(root))
                rows.append({'ecg_id': ecg_id, 'patient_id': offset + i, 'strat_fold': i % 8 + 1 if role == 'train' else 9 if role == 'validation' else 10, 'scp_codes': "{'AFIB': 100.0}" if positive else "{'SR': 100.0}", 'filename_lr': name})
    pd.DataFrame(rows).to_csv(root / 'ptbxl_database.csv', index=False)

def integration_checks():
    with tempfile.TemporaryDirectory(prefix='afficient_checks_') as directory:
        root = Path(directory)
        data = root / 'data'
        synthetic_dataset(data)
        baseline, resumed = (root / 'complete', root / 'resumed')
        current = [None]
        opened_test = []
        original_load = data_cache.load_record

        def guarded_load(path, frequency, leads):
            if path.name.startswith('test_'):
                for name in MODEL_NAMES:
                    assert (current[0] / 'runs' / name / 'seed_42' / 'frozen.json').exists()
                opened_test.append(path.name)
            return original_load(path, frequency, leads)
        data_cache.load_record = guarded_load
        common = ['--data-dir', str(data), '--epochs', '2', '--seeds', '42', '--batch-size', '16', '--threads', '2', '--bootstrap', '20', '--threshold-bootstrap', '20', '--device', 'cpu', '--skip-runtime-estimate']
        try:
            current[0] = baseline
            runner.main([*common, '--output-dir', str(baseline)])
            assert len(opened_test) == 24
            current[0] = resumed
            original_save = runner.atomic_save
            interrupted = [False]

            def interrupt_after_saved_epoch(value, path):
                original_save(value, path)
                if Path(path).name == 'last_training.pt' and value['epoch'] == 1 and (not interrupted[0]):
                    interrupted[0] = True
                    raise KeyboardInterrupt
            runner.atomic_save = interrupt_after_saved_epoch
            try:
                runner.main([*common, '--output-dir', str(resumed)])
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError('Interrupt injection was not reached')
            finally:
                runner.atomic_save = original_save
            assert interrupted[0] and (not (resumed / 'RUNNING.lock').exists())
            assert not (resumed / 'cache/test.npy').exists()
            assert not (resumed / 'study_status.json').exists()
            assert len(opened_test) == 24
            runner.main([*common, '--output-dir', str(resumed)])
            for name in MODEL_NAMES:
                a = baseline / 'runs' / name / 'seed_42'
                b = resumed / 'runs' / name / 'seed_42'
                wa = torch.load(a / 'weights.pt', map_location='cpu', weights_only=True)
                wb = torch.load(b / 'weights.pt', map_location='cpu', weights_only=True)
                for key in wa:
                    torch.testing.assert_close(wa[key], wb[key], rtol=0, atol=0)
                pd.testing.assert_frame_equal(pd.read_csv(a / 'test_predictions.csv'), pd.read_csv(b / 'test_predictions.csv'))
                table = pd.read_csv(a / 'parameter_tensors.csv')
                assert int(table.parameters.sum()) == sum((p.numel() for p in make_model(name).parameters()))
            assert json.loads((resumed / 'study_status.json').read_text())['status'] == 'complete'
            before = len(opened_test)
            saved_hash = runner.study.sha256(resumed / 'runs/afficient_419/seed_42/weights.pt')
            runner.main([*common, '--output-dir', str(resumed)])
            assert len(opened_test) == before
            assert runner.study.sha256(resumed / 'runs/afficient_419/seed_42/weights.pt') == saved_hash
            changed = common.copy()
            changed[changed.index('--epochs') + 1] = '3'
            try:
                runner.main([*changed, '--output-dir', str(resumed)])
            except ValueError as exc:
                assert 'changed' in str(exc)
            else:
                raise AssertionError('Configuration mismatch was not rejected')
            assert (resumed / 'run_bundle.zip').is_file()
            lock = root / 'test.lock'
            runner.acquire_lock(lock)
            try:
                runner.acquire_lock(lock)
            except ValueError as exc:
                assert 'active' in str(exc)
            else:
                raise AssertionError('Concurrent run lock was not enforced')
            lock.unlink()
            splits = pd.read_csv(resumed / 'split_manifest.csv')
            patient_sets = [set(part.patient_id) for _, part in splits.groupby('role')]
            assert all((not a.intersection(b) for i, a in enumerate(patient_sets) for b in patient_sets[:i]))
            planned = root / 'planned'
            current[0] = planned
            runner.main(['--data-dir', str(data), '--output-dir', str(planned), '--epochs', '1', '--seeds', '42', '43', '--batch-size', '16', '--threads', '2', '--bootstrap', '20', '--threshold-bootstrap', '20', '--device', 'auto'])
            execution = json.loads((planned / 'execution_plan.json').read_text())
            assert execution['epochs'] == 1 and len(execution['measured']) == 4
            summary = pd.read_csv(planned / 'comparison.csv')
            raw = pd.read_csv(planned / 'run_metrics.csv')
            for name in MODEL_NAMES:
                row = summary[summary.model == name].iloc[0]
                group = raw[raw.model == name]
                assert row.n_seeds == 2
                assert np.isclose(row.auroc_mean, group.auroc.mean())
                assert np.isclose(row.auprc_seed_sd, group.auprc.std(ddof=1))
        finally:
            data_cache.load_record = original_load

class SyntheticChecks(unittest.TestCase):

    def test_model_operations(self):
        torch.set_num_threads(2)
        architectural_checks()

    def test_training_resume_and_isolation(self):
        torch.set_num_threads(2)
        integration_checks()
