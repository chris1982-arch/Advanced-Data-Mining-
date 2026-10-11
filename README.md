# Edge-Ready ECG Classification — Submission 1

This project prepares MIT-BIH ECG data for four-class heartbeat classification (N/S/V/F), following the first three CRISP-DM stages. No model training is required for this submission.

## Project files

Place these files in the same directory:

```text
project/
├── Data_Processing_2nd_edition.ipynb
├── prepare_data.py
├── requirements.txt
└── README.md
```

The notebook includes exploratory analysis, explanatory text and preparation code. The Python script runs the same preparation pipeline without notebook Markdown or installation commands. Minor differences in validation error messages do not change the prepared datasets.

## Run in Google Colab

1. Upload `Data_Processing_2nd_edition.ipynb` to Google Colab.
2. Select a Python 3 runtime with CPU.
3. Run the installation cell (`%pip install -q wfdb`).
4. Select **Runtime → Run all**.

This workflow does not require uploading `requirements.txt`. Colab supplies the other scientific Python packages, and their actual versions are recorded in `results/manifest.json`. They may differ from the pinned local requirements.

An internet connection is required. The initial dataset download can take several minutes.

## Run locally

Python 3.13 is recommended. Open a terminal in the directory containing the four project files. Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The scientific-package versions are pinned to those recorded in the verified Colab run. This requirements file is not a complete transitive dependency lock.

### Run the preparation script

```bash
python prepare_data.py
```

Run from the project directory because data and output paths are relative to the current working directory. The script saves plots using a non-interactive Matplotlib backend.

### Run the notebook locally

```bash
jupyter notebook
```

Open `Data_Processing_2nd_edition.ipynb` and run every cell in order. If the requirements have already been installed, skip the notebook's installation cell to avoid changing the environment.

## Data download

Both entry points automatically download data through WFDB:

- MIT-BIH Arrhythmia Database: selected records are saved in `mitdb/`.
- MIT-BIH Noise Stress Test Database: `bw`, `ma` and `em` noise records are saved in `nstdb/`.

NSTDB records provide illustrative baseline-wander, muscle-artifact and electrode-motion examples. They are not classifier robustness results.

Do not include the complete raw `mitdb/` or `nstdb/` directories in the submission.

## Record and patient separation

The pipeline uses the course DS1/DS2 partition with `STRICT_PATIENT_SPLIT = True`. Development record 201 is excluded because test record 202 belongs to the same subject. This is a documented deviation from the recommended record lists.

**Training:** 101, 106, 108, 112, 114, 115, 116, 118, 119, 122, 203, 207, 208, 209, 215, 220, 230.

**Validation:** 223, 205, 124, 109.

**Test:** 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234.

Exploratory plots use development records only. Class weights use training labels only. Test data are processed to produce the required dataset and statistics, without using them for split selection or hyperparameter tuning.

Setting `STRICT_PATIENT_SPLIT = False` restores the full course lists but allows the subject shared by 201/202 across development and test. The statistics below apply only to the default strict protocol.

## Preprocessing

1. Select MLII by name, including channel 1 of record 114, and verify a sampling frequency of 360 Hz.
2. Stop with an error if MLII is missing or the selected raw signal contains non-finite values.
3. Filter the complete recording with a Butterworth bandpass: order parameter 4, 0.5–40 Hz, second-order sections and `sosfiltfilt`.
4. Extract `[annotation_sample - 60, annotation_sample + 90)` windows: 60 preceding samples, the annotated sample and 89 following samples, giving 150 samples (approximately 0.417 seconds).
5. Reject boundary, non-finite or effectively flat windows (standard deviation below 1e-8).
6. Normalize each window independently using its own mean and standard deviation, with a denominator floor of 1e-8.
7. Store model inputs as float32 and export the fixed record splits.

Class mapping:

| Class | Original annotation symbols |
|---|---|
| N | N, L, R, e, j |
| S | A, a, J, S |
| V | V, E |
| F | F |

Paced/unclassifiable symbols `/`, `f` and `Q`, and other unsupported events, are excluded from model inputs and counted in the annotation audit.

## Expected dataset statistics

| Split | Records | Subjects | N | S | V | F | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 17 | 17 | 35,582 | 709 | 2,961 | 380 | 39,632 |
| Validation | 4 | 4 | 8,643 | 107 | 629 | 32 | 9,411 |
| Test | 22 | 22 | 44,249 | 1,837 | 3,220 | 388 | 49,694 |

The generated `results/split_statistics.csv` is the authoritative output for each run.

Training-only weights use `training_beats / (4 × training_class_count)`. Their verified values are N=0.278455, S=13.974612, V=3.346167 and F=26.073684. They are prepared for future model training.

## Augmentation

Amplitude scaling (0.9–1.1), zero-padded shifts (up to ±10 samples) and Gaussian noise (standard deviation 0.05 in normalized units) are demonstrated on one training beat using a NumPy generator seeded with 42.

No augmentation is applied to any exported dataset. If used during subsequent training, augmentation will be restricted to training data.

## Generated outputs

```text
results/
├── train.npz
├── validation.npz
├── test.npz
├── split_statistics.csv
├── class_weights.json
├── manifest.json
├── lead_selection.csv
├── annotation_audit.csv
├── quality.csv
├── rejections.csv
└── *.png
```

Each NPZ contains `X` (float32, shape beats × 150), `y` (string labels), `record_ids` and `samples` (original annotation sample positions). Class order for future integer encoding is N/S/V/F.

```python
import numpy as np

with np.load("results/train.npz", allow_pickle=False) as data:
    X_train = data["X"]
    y_train = data["y"]
```

`manifest.json` records actual runtime versions, seed, filter settings, window dimensions, normalization and split lists. CSV audits record annotation exclusions, selected leads, quality screening and window rejections. PNG files contain ECG examples, class/record/amplitude distributions, preprocessing comparisons, augmentation demonstrations and NSTDB noise examples.

In Colab, download the generated files separately. Saving an IPYNB preserves its code and saved outputs, but does not include runtime folders or variables in memory. Include the generated statistics and figures with the submission; include prepared NPZ files if the submission size limits permit them.

## Final checks

The final validation section checks input dimensions, float32 storage, finite values, per-window normalization, required labels, record membership, record/patient independence, saved-array consistency, training-only weights and exported statistics.

The verified default run reports:

```text
Train: PASS — 39,632 beats
Validation: PASS — 9,411 beats
Test: PASS — 49,694 beats

All final checks passed.
```

These checks establish internal consistency; they are not model accuracy or clinical validation results.

## Limitations

Reference annotations supply beat timing; automatic R-peak detection has not been evaluated. Filtering is offline and non-causal, and the short window may omit P/T-wave information. A streaming deployment requires separate validation.

The original exploratory workflow inspected DS2. This historical exposure is disclosed; the revised workflow cannot retroactively make DS2 completely unseen. Future model selection must use development data only.

No trained-model results, clinical validation or measured MAX78002 hardware performance are claimed for Submission 1.

## References

- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).
- [MIT-BIH directory: subjects, leads and annotations](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm).
- [MIT-BIH Noise Stress Test Database](https://physionet.org/content/nstdb/1.0.0/).
