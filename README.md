# AFficient-Dx

This is the official repository for the paper "AFficient-Dx: A Lightweight ECG-Based Residual Convolutional Neural Network for Resource-Constrained Atrial Fibrillation Screening" by Leon Du, accepted in the 2026 _IEEE International Conference on Data Mining Workshops (ICDM Workshops)_.

AFficient-Dx classifies atrial fibrillation in 10-second ECGs with 419 trainable parameters. Its three residual blocks use 1, 2, and 4 channels. The original 50-Hz model achieved 0.9716 AUROC** and 0.7154 AUPRC on PTB-XL, with 1.676 kB of FP32 parameter storage.

The repository includes the model implementations, experiment configurations, trained 100-Hz weights, predictions, calibration, confusion matrices, failure cases, robustness tests, and measured CPU timings.

## Recorded results

Values are means ± sample SD across seeds 42, 43, and 44. AUPRC uses average precision.

| Experiment | Model | Input rate | Parameters | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|
| Original study | AFficient-Dx | 50 Hz | 419 | 0.9716 ± 0.0020 | 0.7154 ± 0.0102 |
| Original study | Reference | 100 Hz | 203,841 | 0.9781 ± 0.0022 | 0.8554 ± 0.0299 |
| Matched comparison | AFficient-Dx | 100 Hz | 419 | 0.9726 ± 0.0039 | 0.7370 ± 0.0467 |
| Matched comparison | Reference | 100 Hz | 203,841 | 0.9784 ± 0.0018 | 0.8526 ± 0.0067 |
| Six-epoch comparison | AFficient-Dx | 100 Hz | 419 | 0.9325 ± 0.0220 | 0.5280 ± 0.0555 |
| Six-epoch comparison | Reduced channels | 100 Hz | 238 | 0.9287 ± 0.0338 | 0.5382 ± 0.2041 |
| Six-epoch comparison | Basso, adapted | 100 Hz | 61,947 | 0.9801 ± 0.0039 | 0.8879 ± 0.0228 |
| Six-epoch comparison | Busia, adapted | 100 Hz | 7,095 | 0.9694 ± 0.0006 | 0.7534 ± 0.0227 |

The original comparison changes both model size and sampling rate. The matched comparison holds the input rate and training protocol constant, with a 40-epoch cap. The six-epoch comparison holds the shorter training budget constant across all four models; its scores belong to that comparison. Basso and Busia are adaptations to the common PTB-XL record task; their original published scores are not substituted here.

Original-study values are transcribed from the manuscript in [results/original_reported.csv](results/original_reported.csv). The included checkpoints and predictions cover the matched 100-Hz and six-epoch experiments. See [protocols](docs/PROTOCOLS.md) and [baseline provenance](docs/PROVENANCE.md).

## Install

Use Python 3.10 or newer; Python 3.12 matches the recorded major/minor version. Run these commands from this repository folder. On macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The same environment can be activated later with `source .venv/bin/activate`. [requirements-recorded.txt](requirements-recorded.txt) lists the package versions reported by the training environment; [docs/recorded_environment.json](docs/recorded_environment.json) preserves the complete record.

## Use the included results

These commands do not load PTB-XL or train models:

```bash
python scripts/verify_results.py
python scripts/make_tables.py
```

The first command checks all archived file hashes, loads the weights, checks parameter counts and patient separation, and recalculates test metrics from predictions. The second writes [results/reproduced/TABLES.md](results/reproduced/TABLES.md) and CSVs for layer parameters, model comparisons, calibration, confusion matrices, thresholds, resources, robustness, and failure cases.

## Dataset

Download [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) or use your existing copy. `--data-dir` must point to the folder containing `ptbxl_database.csv` and `records100/`. The scripts read the original 100-Hz files and downsample them when 50-Hz input is requested; `records500/` is unnecessary.

ECG waveform files are not included. Metadata and signal normalization are fitted using the declared patient splits. AF is positive when `AFIB` appears in `scp_codes`; all other records form the negative class.

## Reproduce the experiments

### Matched AFficient-Dx versus reference at 100 Hz

```bash
python scripts/run_study.py --config configs/matched_100hz.json --data-dir data --output-dir outputs/matched_100hz
```

This runs both models across three seeds with a 40-epoch cap and patience 8. It also exports the requested robustness, calibration, failure-case, latency, and host INT8 checks. Add `--dry-run` to inspect the six planned runs without training. Add `--resume` to skip completed runs after an interruption; an unfinished run starts again from its seed.

### Published-model comparison and channel ablation

```bash
python scripts/run_requested_checks.py --config configs/benchmark_100hz.json --data-dir data --output-dir outputs/benchmark_100hz
```

This runs AFficient-Dx, the 238-parameter channel reduction, and the adapted Basso and Busia models for six epochs maximum, with patience 4. The epoch cap is fixed across computers. A timing preflight checks the selected backend; automatic mode can fall back from MPS to CPU if a required operation is unsupported.

Rerun the identical command to resume from the last completed epoch. Code, dataset, or configuration changes require a fresh output folder. The completed folder contains `run_bundle.zip`, weights, predictions, training histories, patient splits, and comparison tables. Training time depends on the machine; the six-epoch setting is a training budget, not a guaranteed time limit.

To rebuild tables from newly completed experiments:

```bash
python scripts/make_tables.py --matched-dir outputs/matched_100hz --benchmark-dir outputs/benchmark_100hz --output-dir outputs/paper_tables
```

### Original-study configurations

```bash
python scripts/train_afib.py --config configs/original_50hz.json --data-dir data --output-dir outputs/original_50hz/seed_42 --seed 42
python scripts/train_afib.py --config configs/original_reference_100hz.json --data-dir data --output-dir outputs/original_reference_100hz/seed_42 --seed 42
```

Repeat with seeds 43 and 44 and corresponding output folders. These configurations reconstruct the manuscript's original full-validation threshold procedure. The supplied result archives do not contain that original run's checkpoints, so the transcribed original scores are kept separately from the verifiable recorded 100-Hz experiments.

## Predict one record

Pass a WFDB record prefix without `.hea` or `.dat`:

```bash
python scripts/predict.py --checkpoint-dir artifacts/matched_100hz/tiny_100hz/seed_42 --record data/records100/00000/00001_lr
```

The command applies the checkpoint's training normalization, calibration, and frozen threshold. It prints the AF probability and binary prediction. Any included reference or benchmark checkpoint can be supplied in the same way.

## Repository layout

| Path | Contents |
|---|---|
| `src/afficient_dx/models/` | Residual CNN, Basso adaptation, Busia adaptation |
| `src/afficient_dx/` | Data loading, training, evaluation, calibration, robustness, profiling, inference |
| `scripts/` | Short commands for training, comparisons, tables, verification, prediction |
| `configs/` | Settings for each experiment protocol |
| `artifacts/` | Recorded checkpoints, predictions, metrics, and SHA-256 manifest |
| `results/` | Original reported values and rebuilt study tables |
| `docs/` | Protocols, provenance, environment, and reproducibility details |
| `tests/` | Synthetic ECG checks and archived-result verification |
| `LICENSES/` | Retained third-party license notices |

Generated training outputs, datasets, caches, and virtual environments are ignored by Git. The small included weight files can be stored directly in Git.

## Verification

```bash
python -m unittest discover -s tests -v
```

The tests use temporary synthetic WFDB records. They check model operations and counts, configuration handling, patient separation, test isolation, prediction, report generation, and exact CPU equivalence between uninterrupted and resumed training. GitHub Actions runs these checks after a push or pull request. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the archived artifact and source checks.

## Citation and licenses

[CITATION.cff](CITATION.cff) provides citation metadata. The code is released under the [MIT license](LICENSE). The Basso and torch_ecg adaptations retain their upstream MIT notices. PTB-XL-derived metadata is covered by the dataset's CC BY 4.0 license; see [THIRD_PARTY.md](docs/THIRD_PARTY.md) for attribution and sources.
