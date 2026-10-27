# Model card: readmission-30d v1.0.0

*Format loosely follows Mitchell et al., "Model Cards for Model Reporting" (2019). All numbers
come from `reports/metrics.json`, `reports/subgroups_test.csv` and `reports/drift_demo*.{csv,json}`,
written by `python -m readmission.train` and `python scripts/drift_demo.py`.*

## Model details

| | |
|---|---|
| Task | Binary classification: will this inpatient encounter be followed by a readmission within 30 days of discharge? |
| Estimator | scikit-learn `HistGradientBoostingClassifier` (native categorical handling) inside a `Pipeline` that also does feature engineering and encoding. Selected over a logistic regression baseline on held-out ROC-AUC. |
| Output | Probability in [0, 1] plus a boolean `flag_for_follow_up` (risk at or above the top-10% threshold, 0.195, set on the test set). |
| Artefact | `models/model.joblib` (~0.2 MB) + `models/metadata.json` (training time, data SHA-256, metrics, feature list, library versions) + `models/reference_profile.json` (training feature distributions for drift monitoring). |
| Reproducibility | `python -m readmission.train`, seed 42. Re-running produces identical metrics. Requires the scikit-learn version recorded in the metadata to load the artefact. |
| Owner | Luke Wamalwa (portfolio project). Licence MIT. |

## Intended use

- **Intended**: a worked example of how a readmission risk model would be built, evaluated
  and served; teaching and portfolio demonstration.
- **Not intended**: any clinical, operational or reimbursement decision about real patients.
  The model has not been prospectively validated, has not been reviewed by clinicians, and was
  trained on data that is 15–25 years old.
- **Out of scope by design**: encounters ending in death or hospice (discharge disposition
  codes 11, 13, 14, 19, 20, 21). They are removed from training and the API rejects them
  with HTTP 422, because readmission is not a meaningful outcome for them.

## Training data

UCI Machine Learning Repository dataset 296, "Diabetes 130-US Hospitals for Years 1999–2008"
(Strack et al., 2014; CC BY 4.0): 101,766 inpatient encounters of 71,518 patients with any
diabetes diagnosis, hospital stay 1–14 days, from 130 US hospitals.

Cohort rules (`readmission.data.build_cohort`):

| Step | Encounters |
|---|---|
| Raw file | 101,766 |
| minus discharged to death / hospice | −2,423 |
| minus gender "Unknown/Invalid" | −3 |
| **Modelling cohort** | **99,340** (69,987 patients) |

Label: `readmitted == "<30"` (11.4% positive). `">30"` and `"NO"` are both negatives.

Split: `GroupShuffleSplit` by `patient_nbr`, 80/20 → 79,567 training and 19,773 test
encounters; no patient appears in both. Model selection used 5-fold `GroupKFold` on the
training portion only.

## Features

30 raw inputs per encounter (the API schema): age band, sex, race (as recorded), admission
type / source, discharge disposition, length of stay, counts of lab tests, procedures,
medications and diagnoses, prior-year outpatient / emergency / inpatient visits, medical
specialty, glucose and HbA1c test results, 8 diabetes drugs (No / Steady / Up / Down), whether
medication changed, and three ICD-9 diagnosis codes. Engineered inside the pipeline:
ICD-9 codes grouped into 9 categories (Strack et al., Table 2), age-band midpoint, total prior
visits, number of diabetes drugs active / changed. Category levels with fewer than 100 training
encounters (e.g. most of the ~70 medical specialties) are pooled.

Race and sex are used as inputs because they are in the source data and the goal here is to
measure, not hide, their relationship to performance. A deployment would need an explicit
decision on whether to include them.

## Performance (held-out patients, n = 19,773, prevalence 11.3%)

| Model | ROC-AUC (95% CI) | PR-AUC | Brier | ECE |
|---|---|---|---|---|
| Gradient boosting (served) | 0.682 (0.671–0.695) | 0.239 | 0.0951 | 0.006 |
| Logistic regression | 0.672 (0.660–0.684) | 0.226 | 0.0959 | 0.009 |
| Heuristic: rank by prior-year inpatient admissions | 0.614 | 0.191 | – | – |
| No-skill (predict prevalence) | 0.500 | 0.113 | 0.1006 | – |

Training-set GroupKFold CV: gradient boosting ROC-AUC 0.671 ± 0.006, logistic 0.659 ± 0.005.

**Operating point** – flag the 10% highest-risk discharges (threshold 0.195): 1,978 of 19,773
test encounters flagged, 563 of 2,244 readmissions caught (recall 25%), precision 28%
(2.5× the base rate). Flagging the top 10% by prior inpatient admissions alone catches
502 (recall 22%, precision 25%), so the model adds 61 caught readmissions per ~20,000
discharges over that one-line rule.

**Calibration** – predicted and observed rates agree within ~1 percentage point in every risk
decile (`reports/calibration_test.csv`), so no post-hoc recalibration was applied.

These numbers are in line with published work on this dataset: 30-day readmission is only
moderately predictable from administrative encounter data.

## Subgroup performance (`reports/subgroups_test.csv`)

| Group | n | Prevalence | ROC-AUC (95% CI) | Flagged | Recall at shared threshold |
|---|---|---|---|---|---|
| Age < 40 | 1,289 | 12.2% | 0.775 (0.735–0.813) | 14.7% | 43% |
| Age 40–59 | 5,376 | 10.4% | 0.714 (0.692–0.736) | 9.2% | 29% |
| Age 60–79 | 9,277 | 11.4% | 0.672 (0.653–0.689) | 10.0% | 24% |
| Age 80+ | 3,831 | 12.2% | 0.601 (0.575–0.629) | 9.5% | 18% |
| Female | 10,726 | 11.5% | 0.690 (0.674–0.705) | 10.3% | 26% |
| Male | 9,047 | 11.2% | 0.672 (0.656–0.690) | 9.7% | 24% |
| African American | 3,697 | 10.8% | 0.676 (0.651–0.702) | 11.2% | 24% |
| Caucasian | 14,813 | 11.7% | 0.680 (0.667–0.694) | 10.0% | 26% |
| Hispanic | 398 | 9.0% | 0.776 (0.701–0.842) | 8.8% | 36% |
| Other | 302 | 10.3% | 0.701 (0.606–0.796) | 6.3% | 23% |
| Race missing | 447 | 7.2% | 0.668 (0.556–0.782) | 4.5% | 13% |
| Asian | 116 | 8.6% | not estimated (n < 200) | 4.3% | 1 of 10 |

What this does and does not show:

- **Age is the clearest gap.** Discrimination falls steadily with age; for patients 80+ the
  model is only modestly better than chance (0.60) and catches 18% of their readmissions at
  the shared threshold, versus 43% for under-40s. Older patients are the group most likely to
  be targeted by a follow-up programme, so this matters.
- **Sex and the two large race groups** (Caucasian, African American) have overlapping
  confidence intervals; no difference is demonstrated.
- **Smaller race groups** have intervals too wide to conclude anything. "Not demonstrated"
  is not the same as "fair".
- Race is as recorded by the hospitals, with 2.2% missing; this is not self-identified
  ethnicity and the categories are coarse.
- Only discrimination and threshold behaviour were checked. Calibration within subgroups,
  intersections (e.g. older African American women), and outcome-label bias (readmissions
  outside the contributing hospitals are invisible and may differ by group) were not.

## Monitoring

`python -m readmission.monitor batch.csv` computes the Population Stability Index of every
model feature against the training profile (PSI ≥ 0.1 moderate, ≥ 0.25 major). Demo results
(`scripts/drift_demo.py`):

| Batch | Rows | Features PSI ≥ 0.1 | Features PSI ≥ 0.25 | ROC-AUC | Mean predicted vs observed |
|---|---|---|---|---|---|
| Held-out test patients | 19,773 | 0 | 0 | 0.682 | 11.3% vs 11.3% |
| Highest 20% of `encounter_id` in test | 3,955 | 5 | 1 (medical specialty) | 0.696 | 11.7% vs 10.5% |
| Perturbed test copy | 19,773 | 4 | 2 (HbA1c, medications) | – | 12.5% (labels not meaningful) |

The `encounter_id` slice is used as a proxy for later admissions; the dataset does not
document that ids are chronological. On that slice ranking held up but the model
over-predicted by about 1 percentage point, which is the kind of calibration drift PSI
alerts are meant to prompt a check for.

## Limitations

- **Old data**: 1999–2008. Coding practice (ICD-9, replaced by ICD-10-CM in the US in 2015),
  diabetes treatment (no GLP-1 agonists or SGLT2 inhibitors in the data) and readmission
  policy (the 2012 Hospital Readmissions Reduction Program) have all changed since.
- **Population**: US hospitals only; diabetic encounters only; stays of 1–14 days only.
- **Label**: a readmission is only visible if it happened at a hospital contributing to the
  source database; readmissions elsewhere count as negatives. No distinction between planned and unplanned readmissions.
- **Missing context**: no vitals, labs beyond glucose / HbA1c, social determinants, or
  post-discharge information.
- **Repeat patients**: 23% of patients have more than one encounter (46% of encounters belong to such patients). All encounters are
  kept; the split is by patient, but the model does not condition on the patient's own
  history beyond the prior-visit counts.
- **Threshold**: the top-10% threshold was chosen on the test set used for reporting; a real
  deployment would fix it on separate data and revisit it as capacity and prevalence change.
- **Not for clinical use.**

## Citation

Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, Clore JN. Impact of HbA1c
Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient
Records. *BioMed Research International* 2014; 2014:781670. doi:10.1155/2014/781670.
Dataset: UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/296 (CC BY 4.0).
