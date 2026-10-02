# Submission 1 — reviewed ECG preparation

Run from this directory with Python 3.13 and a working internet connection:

```powershell
python -m pip install -r requirements.txt
python prepare_data.py
```

Alternatively open `Data_Processing_Reviewed.ipynb` in Jupyter/Colab, install the requirements in that kernel, and run all cells in order. The script and notebook contain the same preparation steps. Set the working directory to this folder. Downloading MIT-BIH and NSTDB happens automatically through WFDB; the raw files stay in `mitdb/` and `nstdb/`. Do not include these raw directories in the submission.

Outputs appear in `results/`: split statistics, selected leads, annotation audit, beat rejection audit, quality summaries, class weights, reproducibility manifest, EDA images and train/validation/test NPZ files. NPZ arrays contain `X` (float32, shape beats × 150), string `y` labels, record IDs and annotation sample indices. Class order for future integer encoding is N/S/V/F. No trained model is required for Submission 1.

The default uses strict patient separation: record 201 is omitted because 202 in DS2 belongs to the same person. This is a documented deviation from the recommended record lists. `STRICT_PATIENT_SPLIT=False` reproduces the full course lists, but that setting is record-independent rather than fully patient-independent. Freeze this choice before modeling; do not choose it based on DS2 performance.

MLII is selected by name, including channel 1 of record 114. Filtering uses fourth-order Butterworth second-order sections, 0.5–40 Hz, followed by offline forward/backward filtering. Windows are `[annotation - 60, annotation + 90)`: 60 samples before, center and 89 after, 150 total at 360 Hz. Normalize each retained window by its own mean/std; reject boundary and flat/invalid windows. Class weights use training labels only. Demonstration augmentations use a seeded NumPy generator and a training beat. No augmentation is stored in validation/test datasets.

The input annotations provide reference beat timing; this is not an evaluated R-peak detector. Forward/backward filtering is non-causal and will require a separate streaming implementation and validation for deployment. No MAX78002 hardware performance or clinical validation is claimed.

`results/manifest.json` records actual package versions and all fixed parameters. After a successful local run, the supplied `requirements-lock.txt` pins the installed packages for reproducing that run.

The original notebook inspected DS2 during EDA and computed weights from all labels. The reviewed pipeline prevents new test-derived decisions, but historical exposure cannot be undone. Disclose it in the report.

Before submission: add your group/member details, read and personalize `report.md`, export it as PDF if required, and review the plots. Keep the notebook, script, requirements, README, report and generated summaries/figures. Include prepared NPZ files only if your course upload limits permit them.

Sources: [MIT-BIH database](https://physionet.org/content/mitdb/1.0.0/), [database introduction](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm), [noise database](https://physionet.org/content/nstdb/1.0.0/).
