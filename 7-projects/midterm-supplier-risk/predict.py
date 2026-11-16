#!/usr/bin/env python3
"""
predict.py - Client Test Script for Supplier SLA Delivery Risk API
ML Zoomcamp 2026 - Midterm Project

Sends two realistic Purchase Orders (Higher-Risk HRDT vs. Low-Risk ARV)
to the running web service and displays the predicted probability and recommendation.
"""

import sys
import json
import requests

# Default to live Cloud Run endpoint (or pass http://localhost:9696 as 1st argument)
LIVE_CLOUD_URL = "https://supplier-sla-risk-628043583125.europe-west1.run.app"
SERVER_URL = sys.argv[1] if len(sys.argv) > 1 else LIVE_CLOUD_URL

# 30-second timeout accommodates serverless cold-starts on Google Cloud Run
TIMEOUT_SECONDS = 30

# Two real PO lines taken from the SCMS dataset (illustrative: one scored high risk, one low risk)
sample_high_risk_po = {
    "country": "Nigeria",
    "managed_by": "PMO - US",
    "fulfill_via": "Direct Drop",
    "vendor_inco_term": "EXW",
    "shipment_mode": "Air",
    "scheduled_delivery_date": "9-Jan-13",
    "po_sent_to_vendor_date": "10/16/12",
    "product_group": "HRDT",
    "sub_classification": "HIV test",
    "vendor": "Orgenics, Ltd",
    "manufacturing_site": "Alere Medical Co., Ltd.",
    "first_line_designation": "Yes",
    "dosage": "Not Applicable",
    "dosage_form": "Test kit",
    "line_item_quantity": 2744.0,
    "line_item_value": 244216.0,
    "pack_price": 89.0,
    "unit_price": 0.89,
    "weight_kilograms": "1235",
    "freight_cost_usd": "88176.17",
    "line_item_insurance_usd": 342.39
}

sample_low_risk_po = {
    "country": "Vietnam",
    "managed_by": "PMO - US",
    "fulfill_via": "Direct Drop",
    "vendor_inco_term": "EXW",
    "shipment_mode": "Air",
    "scheduled_delivery_date": "5-Jul-13",
    "po_sent_to_vendor_date": "2/6/13",
    "product_group": "ARV",
    "sub_classification": "Adult",
    "vendor": "HETERO LABS LIMITED",
    "manufacturing_site": "Hetero Unit III Hyderabad IN",
    "first_line_designation": "No",
    "dosage": "300mg",
    "dosage_form": "Tablet",
    "line_item_quantity": 683.0,
    "line_item_value": 8366.75,
    "pack_price": 12.25,
    "unit_price": 0.2,
    "weight_kilograms": "See ASN-19859 (ID#:59267)",
    "freight_cost_usd": "See ASN-19859 (ID#:59267)",
    "line_item_insurance_usd": 10.35
}


def test_api():
    print(f"Checking service health at {SERVER_URL}/health...")
    try:
        health_resp = requests.get(f"{SERVER_URL}/health", timeout=TIMEOUT_SECONDS)
        print(f"Health check status: {health_resp.status_code}")
        print("Response:", health_resp.json())
    except requests.exceptions.Timeout:
        print(f"[TIMEOUT] Request to {SERVER_URL}/health timed out (> {TIMEOUT_SECONDS}s).")
        print("Note: On Cloud Run, an idle container waking up (cold start) can take 6-10s. Try again in a few seconds.")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Could not connect to {SERVER_URL}. Is the service running?")
        sys.exit(1)

    print("\n" + "="*60)
    print("Test Case 1: Higher-Risk PO (HRDT test kits to Nigeria)")
    print("="*60)
    resp1 = requests.post(f"{SERVER_URL}/predict", json=sample_high_risk_po, timeout=TIMEOUT_SECONDS)
    print("HTTP Status:", resp1.status_code)
    print("Result:", json.dumps(resp1.json(), indent=2))

    print("\n" + "="*60)
    print("Test Case 2: Low-Risk PO (ARV to Vietnam)")
    print("="*60)
    resp2 = requests.post(f"{SERVER_URL}/predict", json=sample_low_risk_po, timeout=TIMEOUT_SECONDS)
    print("HTTP Status:", resp2.status_code)
    print("Result:", json.dumps(resp2.json(), indent=2))


if __name__ == "__main__":
    test_api()
