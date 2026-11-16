#!/usr/bin/env python3
"""
serve.py - FastAPI Web Service for Supplier Delivery Delay Risk Prediction
ML Zoomcamp 2026 - Midterm Project
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.bin")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run train.py first.")

with open(MODEL_PATH, "rb") as f_in:
    artifact = pickle.load(f_in)

dv = artifact['dv']
model = artifact['model']
optimal_threshold = artifact['optimal_threshold']
train_medians = artifact['train_medians']
frequent_categories = artifact['frequent_categories']

app = FastAPI(
    title="Supplier SLA Delivery Risk Engine",
    description="Predicts the likelihood of an SLA delay breach at PO scheduling time.",
    version="1.0.0"
)


class PORequest(BaseModel):
    country: str = Field(..., example="Nigeria")
    managed_by: str = Field(..., example="PMO - US")
    fulfill_via: str = Field(..., example="From RDC")
    vendor_inco_term: str = Field(..., example="N/A - From RDC")
    shipment_mode: Optional[str] = Field("Not Applicable", example="Truck")
    scheduled_delivery_date: str = Field(..., example="15-Nov-26")
    po_sent_to_vendor_date: Optional[str] = Field(None, example="N/A - From RDC")
    product_group: str = Field(..., example="ARV")
    sub_classification: str = Field(..., example="Adult")
    vendor: str = Field(..., example="SCMS from RDC")
    manufacturing_site: Optional[str] = Field("OTHER", example="Generic Hospital Supplies")
    first_line_designation: str = Field("Yes", example="Yes")
    dosage: Optional[str] = Field("Not Applicable", example="Not Applicable")
    dosage_form: Optional[str] = Field("Tablet - FDC", example="Tablet - FDC")
    line_item_quantity: float = Field(..., example=25000.0)
    line_item_value: float = Field(..., example=312500.0)
    pack_price: float = Field(..., example=12.50)
    unit_price: float = Field(..., example=0.42)
    weight_kilograms: Optional[Union[float, int, str]] = Field(None, example="Weight Captured Separately")
    freight_cost_usd: Optional[Union[float, int, str]] = Field(None, example="Freight Included in Commodity Cost")
    line_item_insurance_usd: Optional[float] = Field(0.0, example=512.50)


class PredictionResponse(BaseModel):
    late_delivery_probability: float
    late_delivery_prediction: bool
    optimal_threshold: float
    risk_tier: str
    recommended_action: str


def extract_weight_status(val):
    if val is None or pd.isna(val):
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
    if val is None or pd.isna(val):
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


def preprocess_record(po: PORequest) -> dict:
    # Safe date parsing
    sched_dt = pd.to_datetime(po.scheduled_delivery_date, errors='coerce')
    po_dt = pd.to_datetime(po.po_sent_to_vendor_date, errors='coerce')

    if pd.isna(sched_dt):
        sched_year = 2026
        sched_month = 11
        sched_quarter = 4
        sched_dayofweek = 0
    else:
        sched_year = sched_dt.year
        sched_month = sched_dt.month
        sched_quarter = sched_dt.quarter
        sched_dayofweek = sched_dt.dayofweek

    if pd.isna(sched_dt) or pd.isna(po_dt):
        lead_time = train_medians['planned_lead_time']
        po_missing = 1
    else:
        lead_time = float((sched_dt - po_dt).days)
        po_missing = 0

    weight_status = extract_weight_status(po.weight_kilograms)
    weight_num = pd.to_numeric(po.weight_kilograms, errors='coerce')
    weight_clean = train_medians['weight'] if pd.isna(weight_num) else float(weight_num)

    freight_status = extract_freight_status(po.freight_cost_usd)
    freight_num = pd.to_numeric(po.freight_cost_usd, errors='coerce')
    freight_clean = train_medians['freight'] if pd.isna(freight_num) else float(freight_num)

    def filter_cat(col_name, val):
        allowed = frequent_categories.get(col_name, set())
        return val if val in allowed else 'OTHER'

    record = {
        'Country_clean': filter_cat('Country', po.country),
        'Managed By': po.managed_by,
        'Fulfill Via': po.fulfill_via,
        'Vendor INCO Term': po.vendor_inco_term,
        'shipment_mode_clean': po.shipment_mode or 'Not Applicable',
        'Product Group': po.product_group,
        'Sub Classification': po.sub_classification,
        'Vendor_clean': filter_cat('Vendor', po.vendor),
        'Manufacturing Site_clean': filter_cat('Manufacturing Site', po.manufacturing_site or 'OTHER'),
        'First Line Designation': po.first_line_designation,
        'dosage_clean': po.dosage or 'Not Applicable',
        'weight_status': weight_status,
        'freight_status': freight_status,
        'planned_lead_time_clean': lead_time,
        'po_date_missing': po_missing,
        'scheduled_year': sched_year,
        'scheduled_month': sched_month,
        'scheduled_quarter': sched_quarter,
        'scheduled_dayofweek': sched_dayofweek,
        'log_line_item_value': float(np.log1p(max(0.0, po.line_item_value))),
        'log_line_item_quantity': float(np.log1p(max(0.0, po.line_item_quantity))),
        'Pack Price': float(po.pack_price),
        'Unit Price': float(po.unit_price),
        'insurance_clean': float(po.line_item_insurance_usd or 0.0),
        'weight_clean': weight_clean,
        'freight_clean': freight_clean
    }
    return record


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_type": "XGBClassifier",
        "model_version": "1.0.0",
        "optimal_threshold": optimal_threshold
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(po: PORequest):
    try:
        record = preprocess_record(po)
        X = dv.transform([record])
        prob = float(model.predict_proba(X)[0, 1])
        is_late = bool(prob >= optimal_threshold)

        if prob >= 0.40:
            tier = "HIGH"
            action = "High SLA breach risk: Add buffer lead-time, confirm carrier reservation 14 days prior, or reconsider freight mode / supplier."
        elif prob >= optimal_threshold:
            tier = "MEDIUM"
            action = "Moderate delivery risk: Monitor vendor milestone updates and verify Advance Shipping Notice (ASN) on dispatch."
        else:
            tier = "LOW"
            action = "Standard delivery expected: Follow routine tracking cadence."

        return PredictionResponse(
            late_delivery_probability=round(prob, 4),
            late_delivery_prediction=is_late,
            optimal_threshold=round(optimal_threshold, 2),
            risk_tier=tier,
            recommended_action=action
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=9696, reload=False)
