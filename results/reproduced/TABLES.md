# Reproduced study results

AUROC and AUPRC are recalculated from the archived test logits. Values are means ± sample SD across training seeds. AUPRC uses average precision.

## Table III. AFficient-Dx layer parameters

| Layer | Type | Parameters |
| --- | --- | --- |
| features.0 | Conv1d | 180 |
| features.1 | BatchNorm1d | 2 |
| features.2 | ReLU | 0 |
| features.3.main.0 | Conv1d | 7 |
| features.3.main.1 | BatchNorm1d | 2 |
| features.3.main.2 | ReLU | 0 |
| features.3.main.3 | Conv1d | 5 |
| features.3.main.4 | BatchNorm1d | 2 |
| features.3.skip | Identity | 0 |
| features.3.relu | ReLU | 0 |
| features.4.main.0 | Conv1d | 14 |
| features.4.main.1 | BatchNorm1d | 4 |
| features.4.main.2 | ReLU | 0 |
| features.4.main.3 | Conv1d | 20 |
| features.4.main.4 | BatchNorm1d | 4 |
| features.4.skip.0 | Conv1d | 2 |
| features.4.skip.1 | BatchNorm1d | 4 |
| features.4.relu | ReLU | 0 |
| features.5.main.0 | Conv1d | 56 |
| features.5.main.1 | BatchNorm1d | 8 |
| features.5.main.2 | ReLU | 0 |
| features.5.main.3 | Conv1d | 80 |
| features.5.main.4 | BatchNorm1d | 8 |
| features.5.skip.0 | Conv1d | 8 |
| features.5.skip.1 | BatchNorm1d | 8 |
| features.5.relu | ReLU | 0 |
| features.6 | AdaptiveAvgPool1d | 0 |
| dropout | Dropout | 0 |
| fc | Linear | 5 |
| Total |  | 419 |

## Table IV. AFficient-Dx and reference at 100 Hz

| Model | Parameters | AUROC | AUPRC |
| --- | --- | --- | --- |
| Reference | 203,841 | 0.9784 ± 0.0018 | 0.8526 ± 0.0067 |
| AFficient-Dx | 419 | 0.9726 ± 0.0039 | 0.7370 ± 0.0467 |

## Table V. Six-epoch comparison at 100 Hz

| Model | Parameters | AUROC | AUPRC |
| --- | --- | --- | --- |
| AFficient-Dx (419) | 419 | 0.9325 ± 0.0220 | 0.5280 ± 0.0555 |
| Basso PH-Multi-Scopic n=4 (adapted) | 61,947 | 0.9801 ± 0.0039 | 0.8879 ± 0.0228 |
| Busia tiny transformer (adapted) | 7,095 | 0.9694 ± 0.0006 | 0.7534 ± 0.0227 |
| AFficient-Dx, reduced channels (238) | 238 | 0.9287 ± 0.0338 | 0.5382 ± 0.2041 |

## Resource measurements

Weight storage uses decimal units: 1 kB = 1,000 bytes. CPU latency is the batch-1 forward pass measured in the archived environment. Serialized INT8 file sizes include export overhead; they do not measure runtime RAM or power. The original quantization result status is retained in resources.csv.
