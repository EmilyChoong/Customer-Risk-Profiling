# Data

The real dataset used in this project is proprietary and is not included in this repository.

## Synthetic data (for demo / reproducibility)

A synthetic dataset that mirrors the real schema is provided for running the demo notebook:

```bash
cd data/
python generate_synthetic.py
# outputs: synthetic_data.csv (~99k rows, 10 engines, 60k customers)
```

The generator calibrates marker–decision relationships to reproduce the key findings:
- Decision mix: ~65% standard, ~20% non-standard, ~6% refer, ~2% postpone, ~0.5% decline
- All 8 markers outrank demographics on Cramér's V
- ~43% of (profile × engine) cells are internally inconsistent
- Propensity score AUC ~0.99 on synthetic data (vs ~0.92 on real — higher because synthetic has no QA normalisation noise)

## Schema

The dataset is a flat table in long format. After filtering to `benefit_type = 'Life'` and
dropping rows where all risk markers are missing, the working dataset has ~2.18M rows across 39 engines (real) or ~99k rows across 10 engines (synthetic).

### Row grain

One row per `(customer_key, application_key, enquiry_engine)`.

### Key columns

| Column | Type | Description |
|---|---|---|
| `customer_key` | string | Anonymised customer identifier |
| `application_key` | string | Anonymised application identifier (one customer can have multiple) |
| `enquiry_engine` | string | Which decision engine processed the application |
| `decision` | string | Engine output: `standard`, `non-standard`, `refer/evidence_required`, `postpone`, `decline` |
| `em_load` | float | Mortality loading % applied to non-standard accepts |
| `benefit_type` | string | Type of cover — analysis filtered to `Life` only |

### Risk markers (8 columns)

Each marker takes one of three values: `T` (true), `F` (false), `NAsk` (not asked by this engine).

| Column | Short name |
|---|---|
| `Compliant` | Compliant with treatment |
| `HBP_Hx` | History of / current high blood pressure |
| `Hosp` | Hospitalised previously due to blood pressure |
| `HBP_Related` | Hypertension related conditions / symptoms |
| `KnowsBP` | Knows blood pressure |
| `Meds` | Need / taking blood pressure medication |
| `Followup` | Requires follow-up |
| `WaitRef` | Waiting referral or investigation |

### Demographics

| Column | Notes |
|---|---|
| `age_at_application` | Continuous |
| `bmi` | Continuous |
| `gender` | Categorical: M / F |
| `smoker` | Categorical: Y / N |
| `sys_pressure` | **Overloaded**: numeric string when reading exists, `'F'` when not. Code converts to numeric + adds `bp_known` flag. |
| `dias_pressure` | Same overloading as `sys_pressure` |

### Decision distribution (real data)

| Decision | Share |
|---|---|
| standard | ~56% |
| non-standard | ~21% |
| refer / evidence required | ~20% |
| postpone | ~2.5% |
| decline | ~0.5% |
