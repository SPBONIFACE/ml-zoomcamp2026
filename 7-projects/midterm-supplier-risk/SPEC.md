# Technical & Business Specification: Supplier SLA & Delivery Delay Risk Engine

**Project:** ML Zoomcamp 2026 — Midterm Project (Part 1 of the *Procurement AI Trilogy*)  
**Domain:** B2B Procurement, Global Supply Chain & ERP Risk Analytics  
**Dataset:** USAID Supply Chain Management System (SCMS) Global Delivery History (`10,324` PO/shipment records, `33` raw attributes)  
**Target Score:** 16 / 16 Rubric Points + Client-Ready Consultancy Demo (Complementing *Supplier iScan*)

---

## 1. Executive Summary & Business Problem

### The Procurement Pain Point
In global B2B and public-health procurement, supply chain teams measure supplier and logistics reliability via **On-Time Delivery (OTD)** against the agreed **Scheduled Delivery Date**. However, traditional ERP systems (SAP, Oracle, Coupa) only report SLA breaches **retrospectively**—after a shipment has already missed its delivery window.

When critical shipments arrive late:
- Downstream production or clinical distribution faces **stockouts**.
- Buyers are forced into **emergency air-charter expediting** at 3x–5x freight cost.
- Buffer inventory (working capital) must be inflated across the board to hedge against unpredictable lead times.

### The AI Consultancy Solution
This project delivers an **Order-Time PO Delivery Risk Scoring API** that predicts the probability of an SLA breach (`delivery_delay_days > 0`) **at the moment a Purchase Order / Requisition is scheduled**.
- **Complementary Portfolio Positioning:** While **Supplier iScan** (`ai-dev-tools-zoomcamp`) provides qualitative GenAI supplier due diligence and risk scanning, this **SLA Delivery Risk Engine** provides quantitative, transaction-level predictive ML directly on ERP Purchase Order lines.
- **Operational Actionability:** Every prediction returns:
  1. `late_delivery_probability` ($0.00 \rightarrow 1.00$)
  2. `late_delivery_prediction` (`bool` based on tuned F1/cost-optimal threshold)
  3. `risk_tier` (`LOW`, `MEDIUM`, `HIGH`)
  4. `recommended_action` (e.g., *"Standard monitoring"*, *"Confirm ASN & carrier booking 14 days prior"*, or *"High SLA breach risk: add lead-time buffer or switch freight mode"*).

---

## 2. Dataset Audit & Domain-Specific Data Hygiene

### Raw Dataset Profile
- **Source:** [USAID SCMS Delivery History Dataset](data/SCMS_Delivery_History_Dataset.csv) (`10,324` rows, `33` columns).
- **Target Variable (`is_late`):**
  $$\text{delivery\_delay\_days} = \text{Delivered to Client Date} - \text{Scheduled Delivery Date}$$
  $$\text{is\_late} = \mathbb{I}(\text{delivery\_delay\_days} > 0)$$
  - **Late (`> 0 days`):** `1,186` shipments (**11.49%**) — classic imbalanced operational risk distribution.
  - **Exact On-Time (`0 days`):** `6,324` shipments (**61.26%**).
  - **Early (`< 0 days`):** `2,814` shipments (**27.26%**).

### Key Domain Insights & Missing-Value Funnel Strategy
1. **6-Stage Missing Value Investigation Funnel ([missing_value_playbook.md](../missing_value_playbook.md)):**
   - Rather than ad-hoc `.fillna()`, our `.ipynb` notebook implements the structured 6-Stage Funnel (`MissingDataFunnel`):
     - **Stage 1 & 2 (Global Health & Triage):** Quantify dataset-wide missingness and rank columns (`Dosage` ~16.8% missing, `Shipment Mode` ~3.5% missing, `Line Item Insurance (USD)` ~2.8% missing).
     - **Stage 3 & 4 (Co-occurrence & Automated Root-Cause Discovery):** Use separation-power tests across `Product Group` and `Fulfill Via` + temporal checks across `Scheduled Delivery Date` years to prove *why* values are missing (e.g., `Dosage` is structurally `MNAR` for `HIV test kits` / `Laboratory` items; `Line Item Insurance (USD)` missingness is temporally clustered in early contract years).
     - **Stage 5 & 6 (Action Matrix & Leak-Free Split-First Imputation):** Document the exact root-cause matrix in `.ipynb` and fit numeric medians (`Line Item Insurance (USD)`) strictly on `X_train` after splitting, alongside binary missingness indicators (`insurance_was_missing`).
2. **Fulfillment Channel Risk Asymmetry (`Fulfill Via`):**
   - `Direct Drop` (4,920 shipments, direct from manufacturer via `EXW`, `DDP`, `FCA`, `CIP`): **5.26% late rate** (note: `DDP` has an **8.04%** late rate vs. `EXW` at **4.97%** and `CIP` at **0.36%**).
   - `From RDC` (5,404 shipments, Regional Distribution Center replenishment): **17.15% late rate**—driven by regional customs bottlenecks and last-mile `Truck` transport (**16.08% late**) and `Ocean` freight (**17.52% late**) vs. `Air` (**9.60% late**).
3. **Real-World ERP String Anomalies in Numeric Columns:**
   - `Weight (Kilograms)` has `3,952` non-numeric entries: `"Weight Captured Separately"` (`1,507`) and consolidated ASN/DN cross-references (`"See ASN-..."`, `"See DN-..."`, `2,445`).
   - `Freight Cost (USD)` has `4,126` non-numeric entries: `"Freight Included in Commodity Cost"` (`1,442`), `"Invoiced Separately"` (`239`), and consolidated ASN/DN references (`2,445`).
   - **Consultancy Engineering Strategy:** Rather than dropping 40% of the ERP table or naively coercing to zero, we parse numeric values, resolve cross-referenced ASN/DN IDs where possible (or impute by train-set median product-group weight/freight ratio), and extract high-signal categorical flags:
     - `weight_status`: `numeric`, `captured_separately`, `consolidated_asn`
     - `freight_status`: `numeric`, `included_in_commodity`, `invoiced_separately`, `consolidated_asn`
4. **Preventing Data Leakage (Strict Point-in-Time Contract):**
   - **Excluded Post-Booking Columns:** `Delivered to Client Date`, `Delivery Recorded Date`, `ID`, `PQ #`, `PO / SO #`, `ASN/DN #`.
   - **Engineered Known-at-Booking Features:**
     - Temporal seasonality from `Scheduled Delivery Date`: `scheduled_year`, `scheduled_month`, `scheduled_quarter`, `scheduled_day_of_week`.
     - Lead-time features: `pq_to_schedule_days` and `po_to_schedule_days` (with explicit flags when `PQ First Sent to Client Date` or `PO Sent to Vendor Date` is `"Date Not Captured"` / `"N/A - From RDC"`).
     - Commercial & logistics ratios: `log_line_item_value`, `log_line_item_quantity`, `unit_to_pack_price_ratio`, `freight_to_value_ratio`, `has_insurance`.
     - **Sparse-Class Curation:** Group long-tail `Country` (43 unique), `Vendor` (73 unique), and `Manufacturing Site` (88 unique) values with `< 25` shipments (computed on `X_train`) into `"OTHER_TAIL"` so linear and tree models generalize cleanly without overfitting 1-off routes.

---

## 3. Modeling & Evaluation Strategy (3/3 Rubric Points)

### Validation Split
- **60% Train / 20% Validation / 20% Test** (stratified on `is_late`, with `random_state=42`).

### Evaluation Metrics
Because late deliveries represent **11.49%** of shipments:
- **Primary Metric:** **ROC-AUC** (ranking capability across all thresholds, standard ML Zoomcamp benchmark).
- **Secondary Metrics:** **PR-AUC (Average Precision)** and **F1-Score / Recall at Optimal Decision Threshold** (tuned on Validation set to balance false alarms vs. missed late deliveries).

### Model Families & Hyperparameter Tuning
1. **Model 1 — Logistic Regression (Linear Baseline):**
   - Pipeline: `DictVectorizer` + `StandardScaler(with_mean=False)` + `LogisticRegression(class_weight='balanced')`
   - Tuned parameter: Regularization strength `C` $\in [0.001, 0.01, 0.1, 1.0, 10.0]$.
2. **Model 2 — Random Forest Classifier (Bagging Ensemble):**
   - Tuned parameters: `max_depth` $\in [5, 10, 15, 20]$, `min_samples_leaf` $\in [1, 3, 5, 10]$, `n_estimators` $\in [50, 100, 200]$.
3. **Model 3 — XGBoost Classifier (Gradient Boosted Trees):**
   - Tuned parameters: `eta` (learning rate) $\in [0.01, 0.05, 0.1, 0.3]$, `max_depth` $\in [3, 6, 8, 10]$, `min_child_weight` $\in [1, 5, 10]$, `subsample` / `colsample_bytree`.

---

## 4. `.ipynb`-First Workflow, Production Architecture & API Contract

### Execution Order (`.ipynb` First $\rightarrow$ `.py` Export)
Per [project-instructions.md](../project-instructions.md):
1. **Phase 1 — Interactive Notebook (`notebook.ipynb`, 5 rubric pts):** All data preparation, 6-Stage Missing Value Funnel ([missing_value_playbook.md](../missing_value_playbook.md)), ERP anomaly parsing, EDA, feature importance analysis, and multi-model hyperparameter tuning happen interactively inside `notebook.ipynb`.
2. **Phase 2 — Script Export (`train.py`, 1 rubric pt):** Once the winning model and preprocessing pipeline are finalized in `notebook.ipynb`, export the training logic into `train.py` to reproducibly generate `model.bin`.
3. **Phase 3 — Web Service & Containerization (`serve.py` / `predict.py` + `Dockerfile`, 5 rubric pts):** Wrap `model.bin` in a FastAPI service, containerize with Docker, and verify locally and on the cloud.

### Project File Structure
```text
7-projects/midterm-supplier-risk/
├── SPEC.md                  # This Lean Consultancy Blueprint
├── README.md                # Executive 16-section documentation (good-readme)
├── data/
│   └── SCMS_Delivery_History_Dataset.csv
├── assets/                  # High-res EDA & model evaluation plots for README & LinkedIn
├── notebook.ipynb           # End-to-end EDA, feature importance & model tuning
├── train.py                 # Standalone reproducible training script -> saves model.bin
├── predict.py               # Standalone test client script for local & cloud verification
├── serve.py                 # FastAPI web service (GET /health, POST /predict) on port 9696
├── model.bin                # Pickled artifact (DictVectorizer + best model + threshold metadata)
├── pyproject.toml           # uv dependency definition
├── uv.lock                  # Reproducible lockfile
├── requirements.txt         # Exported pinned requirements for Docker & peer reviewers
└── Dockerfile               # Production container for local Docker & GCP Cloud Run
```

### FastAPI Endpoint Contract (`serve.py`)
- **`GET /health`**: Returns `{"status": "healthy", "model_version": "1.0.0"}`
- **`POST /predict`**:
  - **Request Payload (PO Line Booking Attributes):**
    ```json
    {
      "country": "Nigeria",
      "managed_by": "PMO - US",
      "fulfill_via": "From RDC",
      "vendor_inco_term": "N/A - From RDC",
      "shipment_mode": "Truck",
      "scheduled_delivery_date": "15-Nov-26",
      "po_sent_to_vendor_date": "N/A - From RDC",
      "pq_first_sent_to_client_date": "10-Aug-26",
      "product_group": "ARV",
      "sub_classification": "Adult",
      "vendor": "SCMS from RDC",
      "dosage_form": "Tablet - FDC",
      "first_line_designation": "Yes",
      "line_item_quantity": 25000,
      "line_item_value": 312500.0,
      "pack_price": 12.50,
      "unit_price": 0.42,
      "weight_kilograms": "Weight Captured Separately",
      "freight_cost_usd": "Freight Included in Commodity Cost",
      "line_item_insurance_usd": 512.50
    }
    ```
  - **Response Payload:**
    ```json
    {
      "late_delivery_probability": 0.684,
      "late_delivery_prediction": true,
      "optimal_threshold": 0.35,
      "risk_tier": "HIGH",
      "recommended_action": "High SLA breach risk: confirm regional stock & carrier capacity immediately or switch shipment mode."
    }
    ```
