# Published architectures and record-task adaptations

These are adapted baselines trained from scratch on the same PTB-XL folds,
12-lead 100-Hz records, loss, optimizer, seed set, calibration/threshold protocol
and maximum epoch budget as the two AFficient-Dx configurations. The comparison
estimates performance within this common budget. Published original-task metrics
are not substituted for measured AF AUROC/AUPRC.

## Basso: PH-Multi-Scopic, hypercomplex dimension n=4

Paper: *Efficient ECG-based Atrial Fibrillation Detection via Parameterised Hypercomplex Neural Networks*, EUSIPCO 2023, DOI
[10.23919/EUSIPCO58844.2023.10289763](https://doi.org/10.23919/EUSIPCO58844.2023.10289763).
[Preprint](https://arxiv.org/abs/2211.02678).

Author code: [HypercomplexECG](https://github.com/leibniz-future-lab/HypercomplexECG),
commit `5e35954fda3f765d7f68d05873d4637dcb6d3893`.
Reimplemented subset: `models/phc_layers.py`, `models/phc_nets.py`,
`models/multi_scopic_phc.py`, `cpsc2021/cfg.py`, `cpsc2021/phc_model.py`.
See `../LICENSES/Basso_MIT_LICENSE.txt`.

Inherited default configuration reference: [torch_ecg](https://github.com/DeepPSP/torch_ecg),
commit `a40c65f4fefa83ba7d3d184072a4c05627b7e226`, particularly
`model_configs/cnn/multi_scopic.py` and `model_configs/ecg_seq_lab_net.py`.
This dependency snapshot is a documented reconstruction choice; the author
repository does not pin its original installed torch_ecg revision.
See `../LICENSES/torch_ecg_MIT_LICENSE.txt`.

Retained: three branches, three blocks per branch, widths 16/32/64, the original
dilation scopes, branch kernels [3,3,3]/[5,5,3]/[9,7,5], max-pool stride 2,
batch normalization, ReLU and block dropout [0,.2,0]. Concatenated 192-channel
features pass through PHM squeeze/excitation (reduction 8) and the PHM
256/64/output MLP with Mish and dropout [.2,.2,0]. The last PHM layer in each
MLP uses n=1, as in the supplied implementation.

Adapted: 12 actual ECG channels instead of two channels padded to four; n stays
4. One binary record output replaces the original sequence-task head. Sequence
logits are linearly restored to the record length and averaged. Common train-only
normalization, record labels, weighted BCE and the shared training schedule
replace the original CPSC-2021 preprocessing, sequence labels and schedule.
No author pretrained weights or original-task scores are used.

Compatibility details: the author PHConv1D allocates a bias but does not use it
in its forward convolution. This behavior and its allocated parameter count are
preserved; `parameter_tensors.csv` identifies these tensors explicitly, and
`frozen.json` also records the count used in forward. Kronecker weights are
constructed device-independently as local tensors, removing the original CUDA
assumption without changing the operation. Each original convolution's batch
normalization is retained, including the wrapper's default normalization.

The reconstructed 4-input, 1-output configuration has **61,403 allocated
parameters**, consistent with the paper's rounded **61k** configuration. The
12-input adaptation has **61,947**; use this actual value in the common-protocol
table. Parameter-count agreement is a structural check, not validation of the
paper's original performance.

## Busia: tiny heartbeat transformer

Paper: DOI [10.1109/TBCAS.2024.3401858](https://doi.org/10.1109/TBCAS.2024.3401858).
[Preprint and Figure 3/Table II](https://arxiv.org/abs/2402.10748).
This is a **paper-based reimplementation**. No official model code was linked in
the consulted paper. Do not call it author-provided code or an exact original-task
reproduction.

Retained core: 198-point heartbeat window; convolutional tokenizer (kernel/stride
3, dimension 16); learned 66-by-16 positional embedding; one pre-normalized
8-head self-attention block; pre-normalized 16-to-128-to-16 feed-forward branch
with GELU after both layers; final layer normalization and mean token pooling;
two-input/two-output bias-free RR projection; concatenated classifier. The
1-input/5-class core has **6,643 parameters**, matching Table II exactly.
The 12-input/1-output core has **7,095 parameters**. Both use full-precision
training; the original paper's quantization-aware training/MCU deployment is
not reproduced here.

Record adaptation: R peaks are detected in lead II from the shared 100-Hz ECG,
using the deterministic, label-independent energy detector in `src/afficient_dx/cache.py`
(5–18-Hz bandpass, derivative energy, 120-ms integration and 250-ms refractory
period). This is a declared preprocessing choice, not the author's original
R-peak detector. All detected beats are used. Each beat receives a 0.55-second
12-lead window by linear interpolation onto the core's original 198 positions.
Interpolation changes window representation but adds no information beyond the
shared 100-Hz source. Boundary samples use zero padding. Preceding/following RR
intervals are measured in seconds and normalized to [-2,2] using training-only
minima/maxima; values beyond that range are clipped at evaluation. The first/
last missing interval uses the record's median detected interval (1 second if
unavailable). A record with no detected peak uses its center and 1-second RR
intervals; fallback counts are exported. Mean beat logits produce one binary
record AF logit. Original five-class heartbeat labels are replaced by PTB-XL's
record AF labels. The shared short-budget full-precision training protocol
replaces the original dataset-specific training procedure.

## AFficient-Dx and further channel reduction

Both use the residual CNN from the previously supplied `src/afficient_dx/models/residual.py`.
The 419-parameter configuration has stage widths (1,2,4). The prespecified
smaller configuration keeps the same three-block depth, residual connections,
normalization, kernels, dropout and classification task, with widths (1,1,1):
**238 parameters**, a **43.2% further reduction**. This measures the effect of
that channel reduction at a fixed sampling rate; it does not establish a global
parameter-count optimum.

## Reproducibility

`protocol.json` fixes the models, seeds, split, budget and source hashes before
training. `execution_plan.json` records the backend and common epoch cap. The
uploaded run selected six epochs using the earlier hardware timing rule; the
repository's paper configuration fixes that recorded six-epoch cap explicitly.
Neither policy uses validation/test performance to choose the budget. An altered
dataset, code or setting cannot resume into the same output directory.
All checkpoints are selected by selection-patient AUROC; Platt scaling
and Youden threshold fitting use separate fold-9 patients. Every fitted choice is
saved before test signals are read. Parameter tensors, best weights, normalization,
RR preprocessing, epoch histories, predictions, errors and metrics are exported.
Resume files include optimizer/scheduler and random-generator states. Confidence
intervals resample test patients together across all models, averaging metrics
over the paired training seeds; seed SD is reported separately.

Hardware-based budget selection remains available through `--auto-budget`; its
outputs are separate from the paper configuration.
