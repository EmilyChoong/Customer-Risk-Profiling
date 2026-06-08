# Customer Risk Profiling & Propensity Scoring

Exploratory analysis and propensity modelling across **39 decision engines** (~2.18M customer applications). The project asks whether structured profile attributes collected at the point of customer application carry meaningful signal about final outcomes — and uses that signal to build a portable customer risk score.

The domain is life insurance underwriting, but the analytical pattern generalises directly to any multi-channel acquisition funnel where customer attributes drive a staged accept / escalate / decline decision.

---

## The problem

When a customer applies for life insurance, they answer a set of health-related questions. Different insurers word these questions differently, ask different subsets, and use different internal rules to reach a final decision. This project:

1. Standardises the question-answer data into 8 binary profile attributes (risk markers)
2. Tests whether those markers carry meaningful signal about the customer's final outcome
3. Builds a propensity score that places each customer on a 0–1 risk axis, portable across all 39 engines

The analogy in a product context: imagine 39 different onboarding flows, each collecting slightly different signals about a customer, and you want to know whether those signals predict conversion / churn / upgrade — and build a score that works regardless of which flow the customer came through.

---

## Repository structure

```
├── notebooks/
│   ├── 01_risk_marker_analysis.ipynb    # Funnel analysis, marker associations, decision profiling
│   └── 02_customer_segmentation.ipynb  # Random forest propensity score (p_decline_like)
│
├── data/
│   └── README.md                        # Schema and column definitions (no data included)
│
└── outputs/
    └── figures/                         # Plots not included (proprietary data — see note below)
```

---

## Notebooks

### 01 — Risk Marker Analysis

The core exploratory analysis. Structured as a **validation funnel** — each stage tests a progressively more specific hypothesis about where the signal lives.

**Stage 1 — Determinism check (how much context do you need?)**

Starting from marker profiles only and adding layers:

| Identity key | % of rows where outcome is deterministic |
|---|---|
| Markers only (no engine, no demographics) | ~45% |
| Markers + demographics | ~80% |
| Markers + demographics + date | ~86% |
| Markers + demographics + date + engine | ~98% |

Finding: customer-side attributes (markers + demographics) explain the vast majority of decision variability. Engine identity is incremental, not primary.

**Stage 2 — Feature association ranking (what matters most?)**

Cramér's V for categorical features, η² for continuous, measured against the final decision:

- All 8 risk markers rank above every demographic
- Top marker ("Waiting referral or investigation"): Cramér's V ≈ **0.71**
- Best demographic (diastolic BP): η² ≈ **0.08** — roughly 9× weaker
- Behavioural/procedural attributes (compliance, referral status) dominate physiological ones (age, BMI)

**Stage 3 — What resolves ambiguous profiles?**

When the same marker profile produces different decisions, which demographic variable explains the split? Diastolic BP resolves ~61% of ambiguous cases — it's the primary tie-breaker.

**Stage 4 — Does loading severity track risk?**

Non-standard decisions come with a mortality loading (25, 50, ... 325%). If the markers capture risk consistently, customers with higher loadings should score higher on the risk scale.

They don't. Spearman ρ between loading and marker-based risk score ≈ **0.05** pooled across engines. The markers identify *whether* a customer is high-risk; they don't grade severity within the non-standard band. Loading is engine-specific pricing convention, not a portable risk scale.

---

### 02 — Customer Propensity Scoring

Builds `p_decline_like`: a 0–1 score representing each customer's position on the standard↔decline axis.

**Design choices:**

- **Training population**: standards vs declines only (anchor populations). Non-standards are scored at inference but excluded from training — they don't shift smoothly between the two anchors and add noise.
- **Engine excluded from features**: the score is designed to be portable across engines. Including engine identity would make it engine-conditional.
- **Group-aware split**: 80/20 split on `application_key`, not rows — prevents the same customer's multiple applications leaking across train/test.
- **Class imbalance**: declines are ~0.5% of data. Handled with `class_weight='balanced'`. This means `p_decline_like` is **not** a real-world decline probability — it's a relative position on a balanced axis.

**Hyperparameter objective (three-way trade-off):**

| Objective | Why it matters |
|---|---|
| AUC ≥ 0.92 | Minimum discrimination threshold |
| Decile coverage | Score must spread across [0,1] — bimodal distributions produce empty segments |
| Cross-fold stability | Same customer should score similarly on models trained on different subsets |

These objectives conflict. The deepest random forest maximises AUC but produces a bimodal score useless for segmentation. The sweep picks a frontier point, not a maximum.

**Performance (winning config: depth=8, min\_leaf=100, balanced):**

- Held-out AUC (standards vs declines): **0.92**
- Score distribution: covers all 10 deciles
- Cross-fold mean score std: stable across group-aware folds

**Sanity checks on the final score:**

1. Score distribution by decision class — standards cluster near 0, declines near 1 ✓
2. Score by engine — similar distribution shapes across all 39 engines (confirms portability) ✓
3. Score by loading band within non-standards — roughly flat (confirms the em_load decoupling finding) ✓

---

## Key findings

| Finding | Detail |
|---|---|
| Markers carry strong signal | AUC 0.92 (markers + demographics, no engine ID) |
| Behavioural attributes dominate physiological ones | Top marker is 9× stronger than best demographic on Cramér's V |
| The decline boundary is universal | All 39 engines broadly agree on who to decline; where they differ is on how much to load |
| Loading is not a portable risk scale | Spearman ρ ≈ 0.05 between loading and marker-based score, pooled across engines |
| ~40% impurity sets a performance ceiling | Same marker profile, different decisions — partly demographics, partly information outside the 8-marker representation. Consistent with observed F1 ≈ 0.79 |
| Leakage was caught and corrected | First RF run hit AUC 0.995 — engine dummies were silently included in features. Customer-only rerun is the reported baseline |

**Note on data privacy**
The plots in this repository are generated from synthetic data and do not reflect the actual findings from the real dataset. To preserve confidentiality, the true results cannot be shared publicly. This repository is intended to showcase the analytical approach, methodology, and plot types produced during the exploratory phase of the project.

---

## Methodological notes

**Why decision trees for interpretation?**
The decision rules being modelled are themselves deterministic — decision trees mirror that structure and produce diagrams that domain experts can read directly. Trees also handle the three-state markers (T/F/NAsk) and missing BP readings natively.

**Why not KMeans for segmentation?**
Attempted and abandoned. The 8 markers sit on two roughly orthogonal severity axes — haemodynamic (BP, age, BMI) and procedural (compliance, referral, follow-up status). Any single-axis clustering forces one axis to dominate at the expense of the other. The propensity score in Notebook 2 is the cleaner answer to the segmentation question.

**The bp_known flag**
`sys_pressure` and `dias_pressure` use `'F'` to mean "no reading taken." Without an explicit flag, median imputation silently treats "no reading" as "average reading" — a meaningful difference that the model can exploit.

---

## Data

Proprietary — not included. See `data/README.md` for full schema and column definitions.

---

## Tech stack

Python 3 · pandas · scikit-learn · matplotlib · seaborn · scipy
