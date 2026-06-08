"""
Synthetic data generator for the UW Customer Segmentation demo.
Calibrated to reproduce key real-analysis findings:
  - Decision mix: ~56% standard, ~22% non-standard, ~19% refer, ~2.5% postpone, ~0.8% decline
  - Top marker (WaitRef) Cramer's V ~ 0.65-0.72 vs decision
  - Markers outrank all demographics on association strength
  - AUC (markers + demographics, no engine) ~ 0.90-0.93
  - em_load weakly correlated with marker-based risk (Spearman rho ~ 0.05 pooled)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_CUSTOMERS = 60_000
N_ENGINES   = 10

MARKERS = ['Compliant', 'HBP_Hx', 'Hosp', 'HBP_Related',
           'KnowsBP', 'Meds', 'Followup', 'WaitRef']

ENGINE_CONFIGS = {
    'Engine_01': {'coverage': 0.95, 'strictness': 0.0},
    'Engine_02': {'coverage': 0.80, 'strictness': 0.2},
    'Engine_03': {'coverage': 0.90, 'strictness':-0.2},
    'Engine_04': {'coverage': 0.70, 'strictness': 0.3},
    'Engine_05': {'coverage': 0.85, 'strictness': 0.0},
    'Engine_06': {'coverage': 0.95, 'strictness':-0.4},
    'Engine_07': {'coverage': 0.60, 'strictness': 0.4},
    'Engine_08': {'coverage': 0.90, 'strictness': 0.1},
    'Engine_09': {'coverage': 0.75, 'strictness': 0.2},
    'Engine_10': {'coverage': 0.85, 'strictness': 0.1},
}

ENGINE_MARKER_EXCEPTIONS = {
    'Engine_04': {'Hosp': 0.05},
    'Engine_07': {'KnowsBP': 0.10, 'HBP_Related': 0.15},
    'Engine_09': {'Hosp': 0.08},
}


def generate_customers(n):
    haemo_risk = RNG.beta(1.2, 6, n)   # most customers low-risk
    proc_risk  = RNG.beta(1.0, 7, n)
    latent     = 0.5 * haemo_risk + 0.5 * proc_risk

    age = np.clip(RNG.normal(45, 12, n) + latent * 10, 18, 80).astype(int)
    bmi = np.clip(RNG.normal(27, 5,  n) + latent * 6,  15, 55).round(1)

    gender   = RNG.choice(['M', 'F'], n, p=[0.55, 0.45])
    p_smoke  = np.clip(0.15 + latent * 0.1, 0, 1)
    smoker   = np.array(['Y' if RNG.random() < p else 'N' for p in p_smoke])

    sys_true  = np.clip(RNG.normal(125, 18, n) + haemo_risk * 35, 90, 220).round(0)
    dias_true = np.clip(RNG.normal(80,  12, n) + haemo_risk * 20, 55, 140).round(0)

    return pd.DataFrame({
        'customer_key': [f'CUST_{i:06d}' for i in range(n)],
        'latent_risk':  latent,
        'haemo_risk':   haemo_risk,
        'proc_risk':    proc_risk,
        'age':          age,
        'bmi':          bmi,
        'gender':       gender,
        'smoker':       smoker,
        'sys_true':     sys_true,
        'dias_true':    dias_true,
    })


def assign_markers(customers):
    n  = len(customers)
    lr = customers['latent_risk'].values
    pr = customers['proc_risk'].values
    hr = customers['haemo_risk'].values

    return pd.DataFrame({
        'WaitRef':     RNG.random(n) < (0.005 + 0.55 * lr**1.5),
        'Followup':    RNG.random(n) < (0.010 + 0.45 * lr**1.2),
        'HBP_Related': RNG.random(n) < (0.015 + 0.40 * lr),
        'Compliant':   RNG.random(n) < (0.92  - 0.65 * pr),
        'Meds':        RNG.random(n) < (0.10  + 0.50 * hr),
        'HBP_Hx':      RNG.random(n) < (0.12  + 0.45 * hr),
        'KnowsBP':     RNG.random(n) < (0.28  + 0.35 * hr),
        'Hosp':        RNG.random(n) < (0.003 + 0.08 * lr**2),
    }, index=customers.index).astype(bool)


def get_score(cust_row, marker_row, strictness):
    s = 0.0
    if marker_row['WaitRef']:       s += 2.8
    if marker_row['Followup']:      s += 2.0
    if marker_row['HBP_Related']:   s += 1.6
    if not marker_row['Compliant']: s += 2.2
    if marker_row['Meds']:          s += 0.9
    if marker_row['HBP_Hx']:        s += 0.8
    if marker_row['Hosp']:          s += 1.2
    if marker_row['KnowsBP']:       s += 0.2

    s += max(0, (cust_row['age']      - 60) / 12)
    s += max(0, (cust_row['bmi']      - 35) / 6)
    s += max(0, (cust_row['sys_true'] - 170) / 20)
    s += max(0, (cust_row['dias_true']- 105) / 15)
    if cust_row['smoker'] == 'Y': s += 0.3

    s += RNG.normal(0, 0.6)   # noise — produces ~40% impurity
    s += strictness            # engine-level shift
    return s


def score_to_decision(s):
    if s < 1.6:  return 'standard'
    if s < 3.6:  return 'non-standard'
    if s < 5.0:  return 'refer/evidence_required'
    if s < 6.4:  return 'postpone'
    return 'decline'


def score_to_em_load(s, decision):
    if decision != 'non-standard':
        return np.nan
    bands = [25, 50, 75, 100, 125, 150, 175, 200, 250, 325]
    # em_load driven partly by score but mostly by engine-specific rules
    # (reproduces the weak rho finding)
    engine_noise = RNG.normal(0, 1.5)
    idx = int(max(0, (s - 2.2 + engine_noise) / 0.25))
    return bands[min(idx, len(bands) - 1)]


def build_dataset(customers, markers_df):
    rows    = []
    engines = list(ENGINE_CONFIGS.keys())

    for idx, cust in customers.iterrows():
        n_eng   = RNG.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        chosen  = RNG.choice(engines, n_eng, replace=False)
        app_key = f'APP_{idx:06d}'

        for eng in chosen:
            cfg        = ENGINE_CONFIGS[eng]
            exceptions = ENGINE_MARKER_EXCEPTIONS.get(eng, {})
            obs = {}
            for m in MARKERS:
                cov = exceptions.get(m, cfg['coverage'])
                obs[m] = 'NAsk' if RNG.random() > cov else (
                    'T' if markers_df.loc[idx, m] else 'F')

            score    = get_score(cust, markers_df.loc[idx], cfg['strictness'])
            decision = score_to_decision(score)
            em_load  = score_to_em_load(score, decision)

            bp_known  = RNG.random() > 0.25
            sys_val   = str(int(cust['sys_true']))  if bp_known else 'F'
            dias_val  = str(int(cust['dias_true'])) if bp_known else 'F'

            rows.append({
                'customer_key':    cust['customer_key'],
                'application_key': app_key,
                'enquiry_engine':  eng,
                'decision':        decision,
                'em_load':         em_load,
                'benefit_type':    'Life',
                'age_at_application': cust['age'],
                'bmi':             cust['bmi'],
                'gender':          cust['gender'],
                'smoker':          cust['smoker'],
                'sys_pressure':    sys_val,
                'dias_pressure':   dias_val,
                **obs,
            })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("Generating customers...")
    customers = generate_customers(N_CUSTOMERS)

    print("Assigning markers...")
    markers_df = assign_markers(customers)

    print("Building dataset...")
    df = build_dataset(customers, markers_df)
    print(f"  {len(df):,} rows, {df['customer_key'].nunique():,} customers, "
          f"{df['enquiry_engine'].nunique()} engines")

    print("\nDecision distribution:")
    dist = df['decision'].value_counts()
    for d, c in dist.items():
        print(f"  {d:<32} {c:>7,}  ({c/len(df)*100:.1f}%)")

    df.to_csv('/home/claude/synthetic_data.csv', index=False)
    print(f"\nSaved.")
