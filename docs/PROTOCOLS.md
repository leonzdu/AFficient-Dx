# Experiment protocols

All recorded revision experiments use PTB-XL's 10-second, 12-lead, 100-Hz records. Folds 1–8 are used for training, fold 9 for validation, and fold 10 for testing. AF labels depend only on the presence of `AFIB` in `scp_codes`. The held-out test set contains 2,198 records, including 152 AF-positive records.

## Validation and evaluation

Fold 9 is split by patient using split seed 2026 into selection, calibration, and threshold subsets with 1,093, 537, and 553 records. Selection AUROC determines the best checkpoint. Positive-slope, unweighted Platt scaling is fitted on calibration patients. Youden's J selects a threshold on separate threshold patients. These choices are frozen before test waveform evaluation.

Normalization uses training records only. Both revision comparisons use weighted binary cross-entropy with training negative/positive class ratio, AdamW with learning rate 0.001 and weight decay 0.0001, batch size 64, gradient clipping at norm 5, and ReduceLROnPlateau with factor 0.5, patience 2, and minimum learning rate 0.00001. Seeds are 42, 43, and 44. AFficient-Dx dropout is 0.15.

AUROC and AUPRC are calculated from raw logits. AUPRC is average precision. Calibration uses Brier score and 10-bin expected calibration error. Classification metrics and confusion matrices use calibrated probabilities at the frozen threshold. Fixed-0.5 results and raw calibration results are exported separately. Threshold stability uses 200 patient bootstrap replicates from the threshold subset.

Means and sample SD describe variation across the three training seeds. Patient bootstrap intervals describe test-patient sampling conditional on the fitted models. The six-epoch comparisons resample the same patients for all models, calculate each seed's metric, and average the seed metrics within a replicate. They use 500 replicates. Matched-model per-run intervals use 1,000 replicates; the study runner also exports paired per-seed differences.

## Matched 100-Hz comparison

`configs/matched_100hz.json` runs the 419-parameter AFficient-Dx and the 203,841-parameter reference with the same preprocessing, rate, patient roles, optimizer, stopping criteria, calibration, and threshold selection. Maximum epochs are 40, and early stopping patience is 8. The architecture's stage widths are (1,2,4) and (32,64,128), respectively.

The archived artifacts are in `artifacts/matched_100hz/`. Additional controlled 50-Hz runs are omitted from this repository's recorded study results.

## Six-epoch model comparison

`configs/benchmark_100hz.json` uses a common six-epoch cap and patience 4 for AFficient-Dx (419 parameters), its prespecified (1,1,1)-channel reduction (238 parameters), adapted Basso PH-Multi-Scopic (61,947 allocated parameters), and adapted Busia transformer (7,095 parameters). The remaining training and evaluation settings are shared. The third-party architectures retain their documented model-specific layers and dropout settings.

The uploaded benchmark selected six epochs from a hardware timing preflight with a declared 12-epoch maximum. All 12 model/seed runs completed six epochs. This repository fixes six epochs explicitly so the paper comparison does not depend on another computer's speed. Optional `--auto-budget` reproduces the earlier timing-based policy for separate exploratory runs.

Basso and Busia are adaptations to the common record-level AF task. [PROVENANCE.md](PROVENANCE.md) specifies the upstream sources, preserved operations, parameter counts, and preprocessing changes. The channel reduction is a prespecified ablation; the available results do not establish a global optimal parameter count.

## Stress tests and resource measurements

Matched-model stress tests add Gaussian noise, baseline drift, and synthetic motion artifacts at 20, 10, and 0 dB signal-to-noise ratio, with three noise realizations per condition and noise seed 7919. Noise is added to the original 100-Hz signals. The clean-data normalization, calibrator, and threshold remain fixed. Lead-I and lead-II tests retain that normalized channel and zero all other channels of the trained 12-lead model. They are lead-removal ablations, not separately trained single-lead models.

CPU latency is measured at batch size 1, with one CPU thread, 50 warmup passes, and 200 timed passes. It includes the eager forward pass and Python overhead and excludes preprocessing and file input. The recorded runs were supplied from an Apple M2 Mac; the environment files identify macOS, ARM64, package versions, and timing settings. The 419-parameter model's three seed-specific median latencies average 0.260 ms; the reference averages 0.749 ms.

Storage uses decimal kB: 1 kB = 1,000 bytes. FP32 parameter bytes, serialized state dictionaries, input buffers, and serialized host INT8 models are reported separately. Host latency and export size do not measure microcontroller peak RAM, latency, or power.

## Original study

The manuscript's original AFficient-Dx result uses 50-Hz input, while its reference uses 100 Hz. `results/original_reported.csv` preserves those reported values. `configs/original_50hz.json` and `configs/original_reference_100hz.json` reconstruct the original described training setup and full-fold-9 threshold procedure without probability calibration. The original run's weights and predictions were not present in the supplied archives. These settings and transcribed values are identified separately from the recorded revision experiments.
