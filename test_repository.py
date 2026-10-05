import argparse
import ast
import hashlib
import importlib
import io
import json
import tempfile
import tokenize
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from afficient_dx import benchmarks, study, training
from afficient_dx.artifacts import load_runs, repository_root
from afficient_dx.configuration import parse_configured
from afficient_dx.metrics import scores
from afficient_dx.models.residual import ECGResNet
from afficient_dx.paper_tables import generate
from afficient_dx.prediction import load_checkpoint, predict_record
from afficient_dx.profiling import count_macs
from afficient_dx.quantization import quantize_and_evaluate
from afficient_dx.verification import verify
from test_integration import synthetic_dataset


ROOT = repository_root()


class RepositoryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_scripts_have_no_comments_or_descriptions(self):
        for folder in ['src', 'scripts', 'tests']:
            for path in (ROOT / folder).rglob('*.py'):
                content = path.read_text()
                tokens = tokenize.generate_tokens(io.StringIO(content).readline)
                self.assertFalse(any(token.type == tokenize.COMMENT for token in tokens), str(path))
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.assertIsNone(ast.get_docstring(node), str(path))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ['ArgumentParser', 'add_argument']:
                        self.assertFalse({'help', 'description', 'epilog'} & {key.arg for key in node.keywords}, str(path))

    def test_numerical_source_preserved(self):
        manifest = json.loads((ROOT / 'docs/source_map.json').read_text())
        for entry in manifest['symbols']:
            if entry['changed']:
                continue
            path = ROOT / 'src/afficient_dx' / (entry['destination'].replace('.', '/') + '.py')
            nodes = {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            digest = hashlib.sha256(ast.unparse(nodes[entry['symbol']]).encode()).hexdigest()
            self.assertEqual(digest, entry['original_semantic_sha256'], entry['symbol'])

    def test_model_counts_and_macs(self):
        self.assertEqual(sum(p.numel() for p in ECGResNet(1).parameters()), 419)
        self.assertEqual(sum(p.numel() for p in ECGResNet(32).parameters()), 203841)
        self.assertEqual(count_macs(ECGResNet(1), 1000), 123004)
        self.assertEqual(count_macs(ECGResNet(1), 500), 61576)

    def test_configs_and_matched_plan(self):
        common = ['--data-dir', 'data', '--output-dir', 'outputs/example']
        args = parse_configured(training.parser(), ['--config', str(ROOT / 'configs/original_50hz.json'), *common])
        self.assertEqual(args.frequency, 50)
        self.assertEqual(args.validation_protocol, 'original')
        self.assertEqual(args.calibration, 'none')
        args = parse_configured(study.parser(), ['--config', str(ROOT / 'configs/matched_100hz.json'), *common])
        plan = study.build_plan(args, study.find_trainer(None))
        self.assertEqual(len(plan), 6)
        self.assertEqual({item['experiment'] for item in plan}, {'tiny_100hz', 'reference_100hz'})
        self.assertEqual({item['seed'] for item in plan}, {42, 43, 44})
        for item in plan:
            command = item['command']
            self.assertEqual(command[command.index('--frequency') + 1], '100')
            self.assertEqual(command[command.index('--epochs') + 1], '40')
            self.assertIn('--robustness', command)
        args = parse_configured(benchmarks.parser(), ['--config', str(ROOT / 'configs/benchmark_100hz.json')])
        self.assertEqual(args.epochs, 6)
        self.assertTrue(args.fixed_budget)
        args = parse_configured(benchmarks.parser(), ['--config', str(ROOT / 'configs/benchmark_100hz.json'), '--epochs', '2', '--auto-budget'])
        self.assertEqual(args.epochs, 2)
        self.assertFalse(args.fixed_budget)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text('{"epoch": 6}')
            with self.assertRaisesRegex(ValueError, 'Unknown configuration'):
                parse_configured(benchmarks.parser(), ['--config', str(path)])
            path.write_text('{"seeds": 42}')
            with self.assertRaisesRegex(ValueError, 'list'):
                parse_configured(benchmarks.parser(), ['--config', str(path)])

    def test_archived_results(self):
        result = verify(ROOT)
        self.assertEqual(result['runs'], 18)
        self.assertEqual(result['status'], 'verified')

    def test_report_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = generate(ROOT / 'artifacts/matched_100hz', ROOT / 'artifacts/benchmark_100hz', output)
            self.assertEqual(len(summary), 6)
            row = summary[(summary.scope == 'matched_100hz') & (summary.model == 'tiny_100hz')].iloc[0]
            self.assertAlmostEqual(row.auroc_mean, .9726, places=4)
            self.assertAlmostEqual(row.auprc_mean, .7370, places=4)
            table = pd.read_csv(output / 'table_iii_layers.csv')
            self.assertEqual(int(table.parameters.sum()), 419)
            confusion = pd.read_csv(output / 'confusion_matrices.csv')
            np.testing.assert_allclose(confusion[['tn_mean', 'fp_mean', 'fn_mean', 'tp_mean']].sum(axis=1), 2198)
            failures = pd.read_csv(output / 'failure_cases.csv')
            self.assertEqual(set(failures.outcome), {'FP', 'FN'})
            self.assertTrue((output / 'robustness_summary.csv').is_file())
            row = summary[(summary.scope == 'benchmark_100hz') & (summary.model == 'afficient_419')].iloc[0]
            self.assertAlmostEqual(row.auroc_mean, .9325, places=4)

    def test_single_record_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data'
            synthetic_dataset(data)
            record = data / 'train_00001'
            for checkpoint in [ROOT / 'artifacts/matched_100hz/tiny_100hz/seed_42',
                               ROOT / 'artifacts/benchmark_100hz/runs/busia_transformer/seed_42']:
                result = predict_record(checkpoint, record)
                self.assertTrue(0 <= result['af_probability'] <= 1)
                self.assertEqual(result['af_prediction'], int(result['af_probability'] >= result['threshold']))
                model, frozen = load_checkpoint(checkpoint)
                self.assertTrue(all(torch.isfinite(value).all() for value in model.state_dict().values()))

    def test_quantization_cleanup(self):
        engine = {'value': 'none', 'changes': []}
        class EngineProxy:
            supported_engines = ['none', 'x86']
            @property
            def engine(self):
                return engine['value']
            @engine.setter
            def engine(self, value):
                if value == 'none':
                    raise RuntimeError('NoQEngine')
                engine['changes'].append(value)
                engine['value'] = value
        before = torch.get_num_threads()
        args = argparse.Namespace(quantized_engine='auto', latency_threads=1)
        with patch.object(torch.backends, 'quantized', EngineProxy()), patch('afficient_dx.quantization.copy.deepcopy', side_effect=ValueError('injected')):
            with self.assertRaisesRegex(ValueError, 'injected'):
                quantize_and_evaluate(ECGResNet(1), {}, {}, None, None, args, Path('.'))
        self.assertEqual(engine['changes'], ['x86'])
        self.assertEqual(torch.get_num_threads(), before)

    def test_original_training_and_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            data, output = Path(directory) / 'data', Path(directory) / 'run'
            synthetic_dataset(data)
            args = ['--config', str(ROOT / 'configs/original_50hz.json'),
                    '--data-dir', str(data), '--output-dir', str(output),
                    '--epochs', '1', '--batch-size', '16', '--device', 'cpu', '--train-threads', '2']
            training.main(args)
            result = json.loads((output / 'result.json').read_text())
            self.assertEqual(result['parameters'], 419)
            self.assertEqual(result['frequency_hz'], 50)
            self.assertEqual(result['calibration']['method'], 'none')
            split = pd.read_csv(output / 'split_manifest.csv')
            roles = {name: set(part.patient_id) for name, part in split.groupby('role')}
            self.assertEqual(roles['selection'], roles['threshold'])
            self.assertFalse(roles['train'] & roles['test'])
            prediction = predict_record(output, data / 'test_00089')
            self.assertEqual(prediction['frequency_hz'], 50)
            with self.assertRaises(FileExistsError):
                training.main(args)

    def test_matched_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            data, output = Path(directory) / 'data', Path(directory) / 'matched'
            synthetic_dataset(data)
            args = ['--data-dir', str(data), '--output-dir', str(output), '--suite', 'matched',
                    '--seeds', '42', '--epochs', '1', '--batch-size', '16', '--device', 'cpu',
                    '--train-threads', '2', '--bootstrap', '0', '--threshold-bootstrap', '0',
                    '--paired-bootstrap', '0', '--latency-repeats', '0', '--no-robustness', '--no-quantization']
            study.main(args)
            for model, parameters in [('tiny_100hz', 419), ('reference_100hz', 203841)]:
                run = output / model / 'seed_42'
                result = json.loads((run / 'result.json').read_text())
                self.assertEqual(result['parameters'], parameters)
                self.assertEqual(result['frequency_hz'], 100)
                splits = pd.read_csv(run / 'split_manifest.csv')
                roles = [set(part.patient_id) for _, part in splits.groupby('role')]
                self.assertFalse(any(a & b for i, a in enumerate(roles) for b in roles[:i]))
            self.assertFalse((output / 'tiny_50hz').exists())
            study.main([*args, '--resume'])
            statuses = json.loads((output / 'study_status.json').read_text())
            self.assertTrue(all(row['status'] == 'resumed_complete' for row in statuses))
            with patch('afficient_dx.study.source_fingerprint', return_value={'changed': 'source'}):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    study.main([*args, '--resume'])
