# Reproducibility

## Included evidence

The repository includes all 18 supplied revision checkpoints: six matched 100-Hz runs and twelve six-epoch benchmark runs. Each run retains weights, predictions, calibration or frozen inference metadata, parameters, metrics, and training history. Matched runs also retain robustness and subgroup reports, patient splits, curve data, diagnostic figures, and available host INT8 exports. The benchmark retains its shared split manifest, execution plan, preprocessing normalization, environment, and paired comparisons.

`artifacts/manifest.json` gives a SHA-256 digest for every archived file. `python scripts/verify_results.py` checks these digests, loads all weights, verifies parameter counts, recalculates test metrics and calibrated probabilities, checks frozen decisions, and checks patient separation. It does not need the ECG dataset and does not independently rerun waveform inference for every archived record.

`python scripts/make_tables.py` reconstructs tables from the saved predictions and result metadata. It preserves the distinction between original manuscript values, the matched 100-Hz comparison, and the six-epoch model comparison. Robustness means first average repeated noise realizations within each seed, then average the seed-level values; reported seed SD is calculated across seeds. Failure-case outputs identify recorded errors and associated PTB-XL annotations.

## Source organization

The earlier training script and benchmark utility module contained identical numerical implementations after removing documentation. Shared functions are now in `src/afficient_dx/`; the model adapters and cached preprocessing have dedicated modules. `docs/source_map.json` records original source digests, archive digests, function destinations, and normalized AST digests for numerical functions. Tests compare the unchanged functions to those recorded digests.

The intentional operational changes are package imports, JSON configuration support, source guards covering the entire package, matched-100-Hz defaults, an explicit six-epoch benchmark cap, and compact result packaging. The original full-validation procedure is available as a separately labeled reconstruction. Comments, docstrings, and argument descriptions have been removed from Python files. Scientific explanations are in the Markdown documentation.

Local filesystem paths in archived JSON metadata have been redacted. The original recorded source hashes and protocol identifiers are retained; they identify the earlier flat-script implementation, not the reorganized package. New runs record the current package's hashes and must use new output folders.

## Quantization record

The uploaded matched runs wrote host INT8 exports and inference metadata, then reported a quantization failure when cleanup attempted to restore the unsupported `NoQEngine` backend. Those archived statuses are retained. Available serialized export sizes can be inspected independently; they do not establish a successful microcontroller deployment or measured peak RAM.

The reorganized quantization routine skips restoration to an unsupported empty backend and always restores the thread count. This cleanup fix does not change calibration data, quantization mathematics, or test predictions. New runs record their own success or failure.

## Testing and determinism

The packaged version passed all 12 CPU tests. All 18 archived runs passed the
independent artifact and metric checks. [verification.json](verification.json)
records the checks, test environment, and package source hashes.

The test suite creates temporary synthetic 100-Hz WFDB records with patient-separated train, validation, and test groups. It checks the hypercomplex operations, Busia window interpolation and beat aggregation, end-to-end outputs, freezing before test signal loading, configuration guards, and exact CPU agreement between interrupted/resumed and uninterrupted training. Temporary data and training outputs are deleted afterward.

Random seeds, data partitions, model definitions, normalization, calibrators, thresholds, and software versions are recorded. Exact values can vary with hardware backends and package versions; the archived predictions and weights preserve the supplied experimental outcomes. The test suite validates execution and result consistency, not clinical generalization.
