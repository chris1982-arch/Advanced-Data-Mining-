# Edge-Ready ECG Classification — Progress Roadmap

This document summarizes work completed so far on **Submission 1** (CRISP-DM Stages 1–3: Business Understanding, Data Understanding, Data Preparation) for the MIT-BIH ECG heartbeat classification project. Intended as a quick reference for teammates.

**Dataset:** MIT-BIH Arrhythmia Database, 48 records, 4-class heartbeat classification (N/S/V/F), target platform MAX78002.

---

## 2. Data Understanding

### 2.1 Dataset Description — ✅ Done
Confirmed dataset structure: 48 records, 47 subjects, 2 channels per record, 360 Hz sampling rate, ~30 min per recording, expert beat annotations.

### 2.2 Target Classes — ✅ Done
Applied the standard **AAMI mapping** from original MIT-BIH symbols to the 4 required classes:
- N ← N, L, R, e, j
- S ← A, a, J, S
- V ← V, E
- F ← F
- Excluded: `/`, `f`, `Q` (paced/unclassifiable)

**Result:** 112,647 total annotations found; 101,451 map to the 4 target classes:
| Class | Count |
|---|---|
| N | 90,631 |
| S | 2,781 |
| V | 7,236 |
| F | 803 |

Severe class imbalance confirmed (N = 89% of beats, F = 0.8%).

### 2.3 Exploratory Data Analysis — ✅ Done
Produced:
- Raw signal plot (record 100, first 5s) — clear QRS complexes visible, ~72 bpm
- Example heartbeat per class (N/S/V/F) — V shows a wide/tall distorted peak, F shows a characteristic double peak, S closely resembles N (hard to separate)
- Class distribution bar chart — visually confirms the imbalance
- Beat distribution per record (stacked bar chart) — shows each patient has a very different class composition (e.g. record 232 is ~50% S beats, record 208 has almost all the F beats), motivating patient-independent splitting
- Amplitude histogram (all 48 records, ~30M samples) — range -5.12 to +5.115 mV, mean -0.339 mV, std 0.472 mV; confirms need for normalization

### 2.4 Data Quality — ✅ Done
- **Missing values:** 0 NaN values across all 48 records
- **Extreme amplitudes:** 1,986 samples exceeded ±4mV threshold, almost entirely concentrated in record 116 (1,720 samples / 0.26% of that record) — likely motion artifact
- **Channel/lead consistency:** 45/48 records use MLII as channel 0; records 102, 104, 114 use V5 instead
- **Beat spacing check:** minimum gap between consecutive beats across the dataset is 90 samples; with a 200-sample window, 39/48 records had overlap risk — reduced window to 150 samples (60 before / 90 after R-peak), cutting overlap risk to 20/48 records

### 2.5 Leakage Analysis — ⏳ Pending
Text-only section explaining why random beat-level splitting causes leakage; not yet written up.

---

## 3. Data Preparation

### 3.1 Record Selection — ✅ Done
Used standard DS1 (development, 22 records) / DS2 (test, 22 records) protocol.

### 3.2 ECG Channel Selection — ✅ Done
Selected MLII as the primary channel. Excluded records 102, 104, 114 (V5-only). This reduced DS1 from 22 → 21 usable records; DS2 remained fully intact at 22.

### 3.3 Beat Extraction — ✅ Done
Fixed window of 60 samples before / 90 samples after each R-peak (150 samples total). Beats too close to recording start/end were skipped.

**Result:** 99,286 beats extracted, shape (99286, 150).

### 3.4 Signal Preprocessing — ✅ Done
Pipeline: raw signal → 4th-order Butterworth bandpass filter (0.5–40 Hz, zero-phase via `filtfilt`) → beat segmentation → per-beat z-score normalization (mean=0, std=1).

Confirmed visually that filtering removes baseline wander/high-frequency noise while preserving QRS morphology.

### 3.5 Label Preparation — ✅ Done
AAMI mapping documented (see 2.2).

### 3.6 Class Imbalance — ✅ Done
Chose **class weights** (inverse frequency) over oversampling/SMOTE — avoids inflating training set size (important for lightweight Edge-AI training) and avoids synthetic/unrealistic beats.

**Result — computed weights:**
| Class | Weight |
|---|---|
| N | 0.280 |
| V | 3.454 |
| S | 8.964 |
| F | 31.105 |

To be applied in the loss function during training (Submission 2).

### 3.7 Data Augmentation — ✅ Done
Defined 3 augmentation functions (to be applied only to training data during Submission 2):
- Amplitude scaling (±10%)
- Time shift (±10 samples, zero-padded — not wrapped)
- Additive Gaussian noise (σ=0.05)

All verified visually on a sample beat.

### 3.8 Final Dataset — ✅ Done
Split by record (not by beat) to avoid patient leakage:
- **Train:** 17 DS1 records → 39,716 beats
- **Validation:** 4 DS1 records (223, 205, 124, 109 — manually chosen to ensure adequate F-class representation) → 9,411 beats
- **Test (DS2):** 22 records, untouched → 49,694 beats

**Final class distribution:**
| Split | N | S | V | F | Total |
|---|---|---|---|---|---|
| Train | 35,397 | 825 | 3,116 | 378 | 39,716 |
| Validation | 8,643 | 107 | 629 | 32 | 9,411 |
| Test | 44,249 | 1,837 | 3,220 | 388 | 49,694 |

### 3.9 Reproducibility — ✅ Done
- Software: Python 3.13.15, wfdb 4.3.1, numpy 2.1.3, pandas 2.2.3, scipy 1.16.3, matplotlib 3.10.0
- Random seed: 42
- Beat window: 60 samples before / 90 after R-peak (150 total)
- Filter: 4th-order Butterworth bandpass, 0.5–40 Hz, zero-phase
- Normalization: per-beat z-score

---

## Still to do (for Submission 1)
- [ ] 2.5 Leakage Analysis (write-up)
- [ ] Noise example plot from MIT-BIH Noise Stress Test Database (optional addition to 2.3)
- [ ] Compile full written report from these sections

## Next up (Submission 2)
- Baseline model (e.g. Logistic Regression / small CNN)
- Edge-optimized 1D CNN architecture
- Training with class weights + augmentation
- FP32 vs INT8 quantization comparison
- MAX78002 synthesis via ai8x-training/ai8x-synthesis
