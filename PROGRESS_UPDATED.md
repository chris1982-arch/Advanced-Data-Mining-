# Edge-Ready ECG Classification — Updated Progress

Updated: 3 October 2026

This document records the reviewed Submission 1 workflow and the successful final checks shown in the author's Colab run. It supersedes the counts and lead-selection conclusions in the original PROGRESS document. It is a progress summary, not a replacement for the full CRISP-DM report.

## Submission 1: Code and EDA

The reviewed notebook implements the data-understanding and preparation workflow with English comments. It includes reference-annotation loading, label mapping, development-only EDA, MLII selection by name, filtering, fixed-size beat extraction, normalization, record/subject splitting, training-only class weights, augmentation demonstrations and output exports.

### Dataset and protocol

- Source: MIT-BIH Arrhythmia Database (48 records from 47 subjects in the full database).
- Selected subset: 43 records from 43 subjects.
- Sampling frequency: 360 Hz.
- Recording duration in the selected records: 1,805.56 seconds.
- Selected lead: MLII, found by name rather than assumed channel index.
- Record 114 uses channel 1 (its channel order is V5, MLII).
- Records 102, 104, 107 and 217 are outside the supplied DS1/DS2 partition.
- Strict patient separation excludes development record 201 because test record 202 belongs to the same person. This is a documented deviation from the recommended course lists.
- Validation records were selected using DS1 information; DS2 must not guide model or hyperparameter selection.

### Fixed record lists

**Training (17 records):** 101, 106, 108, 112, 114, 115, 116, 118, 119, 122, 203, 207, 208, 209, 215, 220, 230.

**Validation (4 records):** 223, 205, 124, 109.

**Test (22 records):** 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234.

### Label mapping and annotation audit

| Final class | Original annotation symbols | Development annotations before window rejection |
|---|---|---:|
| N | N, L, R, e, j | 44,231 |
| S | A, a, J, S | 816 |
| V | V, E | 3,590 |
| F | F | 413 |

The selected 43 records contain 101,685 total annotations, including events other than target beats. Paced/unclassifiable symbols `/`, `f` and `Q` are excluded from the target mapping. Other unsupported events are audited by symbol and record; OTHER is not exclusively a category of non-beat rhythm markers.

### EDA and data quality completed

- Five-second development ECG example.
- One example heartbeat for each N/S/V/F class.
- Development class-distribution chart.
- Record-level class-distribution chart.
- Development MLII amplitude histogram.
- Missing-value screening across both development channels.
- Extreme-amplitude screening: 1,720 samples in record 116 and 32 in record 203 exceed an absolute amplitude of 4 mV.
- Lead inventory and visual comparison of both channels of record 114.
- Consecutive-beat spacing inspection.
- Raw, filtered and normalized beat comparison.
- Real NSTDB baseline-wander, muscle-artifact and electrode-motion examples at clean, 12 dB, 6 dB and 0 dB conditions.

Development MLII amplitudes: minimum -5.120 mV, maximum 5.115 mV, mean -0.420 mV and standard deviation 0.501 mV. No NaN values were detected in the development-channel screening. Extreme amplitudes are inspection flags, not proof of motion artifacts. Screening does not establish that every retained segment is artifact-free.

### Preparation pipeline

Raw MLII signal → finite-value check → whole-record Butterworth bandpass → beat segmentation → boundary/invalid/flat-window rejection → per-beat z-score → float32 input.

- Filter: fourth-order Butterworth, 0.5–40 Hz, second-order sections, `sosfiltfilt`.
- Window: `[annotation_sample - 60, annotation_sample + 90)`.
- Input length: 150 samples (approximately 0.417 seconds).
- Window contents: 60 preceding samples, the annotated sample and 89 subsequent samples.
- Normalization: each window's own mean and standard deviation; standard-deviation floor 1e-8.
- Boundary rejection: 18 windows across the selected records (16 N, 1 V, 1 F).
- Reference annotations provide beat timing; an automatic R-peak detector has not been evaluated.
- Filtering is offline and non-causal; streaming deployment will need a separately validated implementation.

### Final retained dataset

| Split | Records | Subjects | N | S | V | F | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 17 | 17 | 35,582 | 709 | 2,961 | 380 | 39,632 |
| Validation | 4 | 4 | 8,643 | 107 | 629 | 32 | 9,411 |
| Test | 22 | 22 | 44,249 | 1,837 | 3,220 | 388 | 49,694 |
| Total | 43 | 43 | 88,474 | 2,653 | 6,810 | 800 | 98,737 |

### Class imbalance and augmentation

Class weights are computed only from training labels, after the final split:

| Class | Weight |
|---|---:|
| N | 0.278455 |
| S | 13.974612 |
| V | 3.346167 |
| F | 26.073684 |

Formula: `training_beats / (4 * training_class_count)`. Applying the weights to a loss belongs to Submission 2.

Seeded augmentation functions demonstrate amplitude scaling (0.9–1.1), zero-padded shifts (±10 samples) and Gaussian noise (standard deviation 0.05 in normalized units). They use a training example. The current exported datasets are not augmented. These examples are not model robustness results.

### Reproducibility: author's Colab run

| Component | Version/value |
|---|---|
| Python | 3.13.15 |
| WFDB | 4.3.1 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| SciPy | 1.16.3 |
| Matplotlib | 3.10.0 |
| NumPy generator seed | 42 |
| Strict patient split | True |

These versions were shown in the author's Colab output. The existing local `requirements-lock.txt` describes the separately verified local environment, which differs from Colab. Use the manifest from the run being submitted when documenting software versions.

### Final validation: PASS

The author added and executed the final validation cell in Colab. It reported:

```text
Train: PASS — 39,632 beats
Validation: PASS — 9,411 beats
Test: PASS — 49,694 beats

All final checks passed.
```

The checks verify fixed input dimensions, float32 storage, finite inputs, per-window normalization, the four required labels, split membership, disjoint records and subjects in strict mode, agreement between in-memory and exported NPZ arrays, training-only weights, and CSV statistics matching the datasets. Save the Colab notebook with this added cell and its output; the validation-cell addition was made in Colab and is not implied to exist in every earlier local copy.

### Exported outputs

- `results/train.npz`, `validation.npz`, `test.npz`: prepared inputs, labels, record IDs and annotation samples.
- `results/split_statistics.csv`: final split counts.
- `results/annotation_audit.csv`: original-symbol audit by record.
- `results/rejections.csv`: rejected target windows and reasons.
- `results/lead_selection.csv`: selected channel and recording metadata.
- `results/quality.csv`: quality-screening summaries.
- `results/class_weights.json`: training-only weights.
- `results/manifest.json`: actual runtime versions, split lists and preprocessing parameters.
- EDA and preprocessing PNG figures.

## Remaining before Submission 1

The core code is complete for Submission 1. Submission preparation still requires:

- Finalize the full CRISP-DM Stages 1–3 report, including application understanding, measurable success criteria, constraints, risks and interpretation of the figures.
- Add group/member details and review the proposed success thresholds.
- Document the strict patient protocol and the exclusion of 201.
- Disclose that the original notebook inspected DS2 during EDA; the revised workflow cannot undo historical exposure.
- Discuss the small F validation support (32 beats), the short window and the offline filtering limitation.
- Use the author's current Colab software versions in the final report.
- Save/download the executed notebook including the final checks and outputs.
- Include the preparation script, README and generated statistics/figures in the submission package.
- Do not submit complete raw PhysioNet datasets.

## Submission 2: Not yet completed

Baseline training, compact CNN training, model selection, FP32/INT8 comparison, per-class model evaluation, evaluated noise robustness and MAX78002 synthesis remain future work. No clinical validation or measured MAX78002 hardware performance is claimed.

## References

- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).
- [Database introduction: lead reversal and the shared subject in records 201/202](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm).
- [MIT-BIH Noise Stress Test Database](https://physionet.org/content/nstdb/1.0.0/).
