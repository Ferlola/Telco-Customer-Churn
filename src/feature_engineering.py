import pandas as pd
import numpy as np


def create_avg_monthly_spend(df):
    df_feat = df.copy()
    df_feat['Avg Monthly Spend'] = np.where(
        df_feat['Tenure Months'] > 0,
        df_feat['Total Charges'] / df_feat['Tenure Months'],
        df_feat['Monthly Charges']
    )
    return df_feat


def create_tenure_group(df):
    df_feat = df.copy()
    bins = [0, 6, 12, 24, 48, 72, df_feat['Tenure Months'].max() + 1]
    labels = ['0-6 meses', '6-12 meses', '1-2 años', '2-4 años', '4-6 años', '6+ años']
    df_feat['Tenure Group'] = pd.cut(df_feat['Tenure Months'], bins=bins, labels=labels, right=False)
    return df_feat


def create_num_services(df):
    service_cols = [
        'Phone Service', 'Multiple Lines', 'Online Security',
        'Online Backup', 'Device Protection', 'Tech Support',
        'Streaming TV', 'Streaming Movies'
    ]
    existing_cols = [c for c in service_cols if c in df.columns]
    df_feat = df.copy()
    df_feat['Num Services'] = 0
    for col in existing_cols:
        df_feat['Num Services'] += (df_feat[col] == 'Yes').astype(int)
    if 'Internet Service' in df.columns:
        df_feat['Has Internet'] = (df_feat['Internet Service'] != 'No').astype(int)
    return df_feat


def create_premium_customer(df):
    df_feat = df.copy()
    monthly_threshold = df_feat['Monthly Charges'].quantile(0.75)
    tenure_threshold = df_feat['Tenure Months'].quantile(0.75)
    df_feat['Premium Customer'] = (
        (df_feat['Monthly Charges'] >= monthly_threshold) &
        (df_feat['Tenure Months'] >= tenure_threshold)
    ).astype(int)
    return df_feat


def create_new_customer(df):
    df_feat = df.copy()
    df_feat['New Customer'] = (df_feat['Tenure Months'] <= 6).astype(int)
    return df_feat


def create_has_dependents_partner(df):
    df_feat = df.copy()
    df_feat['Has Dependents or Partner'] = (
        (df_feat['Dependents'] == 'Yes') | (df_feat['Partner'] == 'Yes')
    ).astype(int)
    return df_feat


def create_engagement_score(df):
    df_feat = df.copy()
    df_feat['Engagement Score'] = df_feat['Num Services'] / 8.0
    return df_feat


def engineer_features(df):
    print("Aplicando ingeniería de características...")
    df_feat = df.copy()
    df_feat = create_avg_monthly_spend(df_feat)
    df_feat = create_tenure_group(df_feat)
    df_feat = create_num_services(df_feat)
    df_feat = create_premium_customer(df_feat)
    df_feat = create_new_customer(df_feat)
    df_feat = create_has_dependents_partner(df_feat)
    df_feat = create_engagement_score(df_feat)

    new_features = [
        'Avg Monthly Spend', 'Tenure Group', 'Num Services',
        'Has Internet', 'Premium Customer', 'New Customer',
        'Has Dependents or Partner', 'Engagement Score'
    ]
    existing_new = [f for f in new_features if f in df_feat.columns]
    print(f"Nuevas variables creadas: {existing_new}")
    print(f"Shape después de feature engineering: {df_feat.shape}")
    return df_feat
