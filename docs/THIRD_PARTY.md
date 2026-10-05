# Third-party sources and attribution

## Basso and torch_ecg

The hypercomplex adaptation follows [HypercomplexECG](https://github.com/leibniz-future-lab/HypercomplexECG) at commit `5e35954fda3f765d7f68d05873d4637dcb6d3893`, with documented architecture configuration references from [torch_ecg](https://github.com/DeepPSP/torch_ecg) at commit `a40c65f4fefa83ba7d3d184072a4c05627b7e226`.

The complete supplied license notices are retained in [Basso_MIT_LICENSE.txt](../LICENSES/Basso_MIT_LICENSE.txt) and [torch_ecg_MIT_LICENSE.txt](../LICENSES/torch_ecg_MIT_LICENSE.txt). Model and task adaptations are documented in [PROVENANCE.md](PROVENANCE.md).

Associated paper: Basso et al., *Efficient ECG-based Atrial Fibrillation Detection via Parameterised Hypercomplex Neural Networks*, [DOI: 10.23919/EUSIPCO58844.2023.10289763](https://doi.org/10.23919/EUSIPCO58844.2023.10289763), [preprint](https://arxiv.org/abs/2211.02678).

## Busia

The transformer is a paper-based reimplementation of Busia et al., *A Tiny Transformer for Low-Power Arrhythmia Classification on Microcontrollers*, [DOI: 10.1109/TBCAS.2024.3401858](https://doi.org/10.1109/TBCAS.2024.3401858), [preprint](https://arxiv.org/abs/2402.10748). No author pretrained weights are included. The declared 12-lead record adaptation is documented in [PROVENANCE.md](PROVENANCE.md).

## PTB-XL

Data source: Wagner et al., *PTB-XL, a large publicly available electrocardiography dataset*, [Scientific Data, DOI: 10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6). The scripts target [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), [dataset DOI: 10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45).

The dataset is licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). The archived prediction, split, failure-case, and subgroup tables include PTB-XL record identifiers and metadata with model-derived outputs added. These dataset-derived contents retain the dataset's license and attribution. ECG waveforms are not redistributed here. The repository's MIT license applies to the code and does not replace the dataset license.
