#!/usr/bin/env python3
"""
train.py - Standalone Training Script for Supplier Delivery Delay Risk Prediction
ML Zoomcamp 2026 - Midterm Project

Loads USAID SCMS Global Delivery dataset, parses ERP anomalies, engineers
point-in-time booking features, trains the champion XGBoost classifier,
and serializes the full pipeline artifact to model.bin.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "SCMS_Delivery_History_Dataset.csv")
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "model.bin")


def load_and_clean_data(path: str) -> pd.DataFrame:
    print(f"Loading raw dataset from {path}...")
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns.")

    # Convert dates with safe coercion
    for col in ['PO Sent to Vendor Date', 'Scheduled Delivery Date', 'Delivered to Client Date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Define delay and target
    df['delivery_delay_days'] = (df['Delivered to Client Date'] - df['Scheduled Delivery Date']).dt.days
    df['is_late'] = (df['delivery_delay_days'] > 0).astype(int)

    # Parse ERP anomalies in Weight and Freight
    df['weight_numeric'] = pd.to_numeric(df['Weight (Kilograms)'], errors='coerce')
    df['freight_numeric'] = pd.to_numeric(df['Freight Cost (USD)'], errors='coerce')

    def extract_weight_status(val):
        if pd.isna(val):
            return 'missing'
        s = str(val).strip()
        if 'Weight Captured Separately' in s:
            return 'captured_separately'
        elif 'See ' in s:
            return 'consolidated'
        elif s.replace('.', '', 1).isdigit():
            return 'numeric'
        return 'other'

    def extract_freight_status(val):
        if pd.isna(val):
            return 'missing'
        s = str(val).strip()
        if 'Freight Included' in s:
            return 'included_in_commodity'
        elif 'Invoiced Separately' in s:
            return 'invoiced_separately'
        elif 'See ' in s:
            return 'consolidated'
        elif s.replace('.', '', 1).isdigit():
            return 'numeric'
        return 'other'

    df['weight_status'] = df['Weight (Kilograms)'].apply(extract_weight_status)
    df['freight_status'] = df['Freight Cost (USD)'].apply(extract_freight_status)

    # Known-at-booking features
    df['planned_lead_time_days'] = (df['Scheduled Delivery Date'] - df['PO Sent to Vendor Date']).dt.days
    df['po_date_missing'] = df['planned_lead_time_days'].isnull().astype(int)

    df['scheduled_year'] = df['Scheduled Delivery Date'].dt.year
    df['scheduled_month'] = df['Scheduled Delivery Date'].dt.month
    df['scheduled_quarter'] = df['Scheduled Delivery Date'].dt.quarter
    df['scheduled_dayofweek'] = df['Scheduled Delivery Date'].dt.dayofweek

    df['log_line_item_value'] = np.log1p(df['Line Item Value'].clip(lower=0))
    df['log_line_item_quantity'] = np.log1p(df['Line Item Quantity'].clip(lower=0))

    df['dosage_clean'] = df['Dosage'].fillna('Not Applicable')
    df['shipment_mode_clean'] = df['Shipment Mode'].fillna('Not Applicable')
    df['insurance_clean'] = df['Line Item Insurance (USD)'].fillna(0)

    return df


def train_champion_model():
    df = load_and_clean_data(DATA_PATH)

    # Split dataset 60/20/20
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_late'])
    df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42, stratify=df_train_full['is_late'])

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train['is_late'].values
    y_val = df_val['is_late'].values
    y_test = df_test['is_late'].values

    # Impute medians strictly derived from training set
    train_lead_median = df_train['planned_lead_time_days'].median()
    train_weight_median = df_train['weight_numeric'].median()
    train_freight_median = df_train['freight_numeric'].median()

    for d in [df_train, df_val, df_test]:
        d['planned_lead_time_clean'] = d['planned_lead_time_days'].fillna(train_lead_median)
        d['weight_clean'] = d['weight_numeric'].fillna(train_weight_median)
        d['freight_clean'] = d['freight_numeric'].fillna(train_freight_median)

    # Curation of sparse categories
    high_card_cols = ['Country', 'Vendor', 'Manufacturing Site']
    frequent_categories = {}

    for col in high_card_cols:
        counts = df_train[col].value_counts()
        frequent_categories[col] = set(counts[counts >= 20].index)
        for d in [df_train, df_val, df_test]:
            d[f'{col}_clean'] = d[col].apply(lambda x: x if x in frequent_categories[col] else 'OTHER')

    cat_features = [
        'Country_clean', 'Managed By', 'Fulfill Via', 'Vendor INCO Term', 
        'shipment_mode_clean', 'Product Group', 'Sub Classification', 
        'Vendor_clean', 'Manufacturing Site_clean', 'First Line Designation',
        'dosage_clean', 'weight_status', 'freight_status'
    ]

    num_features = [
        'planned_lead_time_clean', 'po_date_missing', 'scheduled_year', 'scheduled_month',
        'scheduled_quarter', 'scheduled_dayofweek', 'log_line_item_value', 
        'log_line_item_quantity', 'Pack Price', 'Unit Price', 'insurance_clean',
        'weight_clean', 'freight_clean'
    ]

    all_features = cat_features + num_features

    # Feature transformation
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train[all_features].to_dict(orient='records'))
    X_val = dv.transform(df_val[all_features].to_dict(orient='records'))
    X_test = dv.transform(df_test[all_features].to_dict(orient='records'))

    print(f"Training champion XGBoost model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    model = XGBClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Validation evaluation & threshold tuning
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    val_pr_auc = average_precision_score(y_val, val_probs)
    print(f"Validation ROC-AUC: {val_auc:.4f} | Validation PR-AUC: {val_pr_auc:.4f}")

    # Optimal threshold for F1
    best_f1 = 0.0
    optimal_threshold = 0.35
    for t in np.linspace(0.05, 0.85, 33):  # same grid as notebook.ipynb
        f1 = f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = t

    print(f"Optimal Decision Threshold: {optimal_threshold:.2f} (Val F1: {best_f1:.4f})")

    # Test set evaluation
    test_probs = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    test_pr_auc = average_precision_score(y_test, test_probs)
    test_f1 = f1_score(y_test, (test_probs >= optimal_threshold).astype(int), zero_division=0)
    print(f"Test ROC-AUC: {test_auc:.4f} | Test PR-AUC: {test_pr_auc:.4f} | Test F1: {test_f1:.4f}")

    # Serialize bundle
    artifact = {
        'dv': dv,
        'model': model,
        'optimal_threshold': float(optimal_threshold),
        'train_medians': {
            'planned_lead_time': float(train_lead_median),
            'weight': float(train_weight_median),
            'freight': float(train_freight_median)
        },
        'frequent_categories': {k: list(v) for k, v in frequent_categories.items()},
        'features': all_features
    }

    with open(MODEL_OUTPUT_PATH, 'wb') as f_out:
        pickle.dump(artifact, f_out)

    print(f"Artifact successfully saved to {MODEL_OUTPUT_PATH} ({os.path.getsize(MODEL_OUTPUT_PATH) / 1024:.1f} KB).")


if __name__ == "__main__":
    train_champion_model()
