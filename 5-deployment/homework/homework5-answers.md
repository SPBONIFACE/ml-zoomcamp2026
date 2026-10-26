# Homework 5: Deploying Machine Learning Models — Answers

## Setup Verification

```bash
uv sync --locked
uv run python smoke_test.py
shasum -a 256 pipeline.bin
```

Artifact SHA-256 checksum verification:
```text
1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1  pipeline.bin
```

---

## Question 1 — Environment check

**Prompt:** Run `uv --version` and record the version in your notes. This is a setup check, not a graded answer: your local `uv` version is not a property of the model.

```bash
uv --version
```

**Output:**
```text
uv 0.12.24 (5411378eb 2026-10-08 aarch64-apple-darwin)
```
*(Note: `0.12.0` was also recorded earlier in `homework5.ipynb`, and the container uses `uv 0.10.11`.)*

**Answer:** `uv 0.12.24`

---

## Question 2 — Locked dependency

**Prompt:** Open `pyproject.toml` and `uv.lock`. Which Scikit-Learn version is part of the 2026 reference environment?
- `1.6.1`
- `1.7.2`
- `1.8.0`
- `2.0.0`

From `pyproject.toml`:
```toml
dependencies = [
    "fastapi==0.119.1",
    "numpy==2.3.3",
    "scikit-learn==1.7.2",
    "uvicorn==0.38.0",
]
```

**Answer:** **`1.7.2`**

---

## Question 3 — Load the model

**Prompt:** Write a Python script that loads `pipeline.bin` with `pickle` and calls `predict_proba` for this lead:

```json
{
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}
```
Report the probability of conversion rounded to three decimal places.

**Script:**
```python
import pickle

with open("pipeline.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

prob = float(pipeline.predict_proba([lead])[0, 1])
print(f"Conversion probability: {prob:.6f} (rounded: {prob:.3f})")
```

**Output:**
```text
Conversion probability: 0.532994 (rounded: 0.533)
```

**Answer:** **`0.533`**

---

## Question 4 — Serve the model

**Prompt:** Start the provided API (`uv run uvicorn predict:app --host 0.0.0.0 --port 9696`) and send this second lead to `POST /predict`:

```bash
curl -s http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74
  }'
```

**Response:**
```json
{
  "conversion_probability": 0.769799,
  "conversion": true
}
```

**Answer:** **`0.770`** (or `0.77`)

---

## Question 5 — Look at the container configuration

**Prompt:** Which Python base-image tag is declared in the `Dockerfile`?
- `python:3.9-slim-bullseye`
- `python:3.11.15-slim-bookworm`
- `python:3.12-slim-bookworm`
- `python:3.13.10-slim-bookworm`

From `Dockerfile`:
```dockerfile
FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3
```

**Answer:** **`python:3.11.15-slim-bookworm`**

---

## Question 6 — Run the container

**Prompt:** Build and run the container, then run the same request from Question 4 against the container using `python q6_test.py` and verify `GET /health`.

```bash
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
```

Testing prediction (`python q6_test.py`):
```text
{'conversion_probability': 0.769799, 'conversion': True}
```

Testing health endpoint (`curl -s http://localhost:9696/health`):
```json
{
  "status": "ok",
  "model_sha256": "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1",
  "feature_defaults_sha256": "1694f842b908fac9f53c0cc19be6d91f934befc63af23cc32fefafde69acd0cd"
}
```

**Answer:** **`0.770`** (or `0.77`)

---

## Summary Table

| Question | Topic | Answer |
| :--- | :--- | :--- |
| **Q1** | `uv --version` (ungraded setup check) | `uv 0.12.24` |
| **Q2** | Locked Scikit-Learn version | `1.7.2` |
| **Q3** | Offline `predict_proba` (`pipeline.bin`) | `0.533` (`0.532994`) |
| **Q4** | FastAPI `/predict` probability | `0.770` (`0.769799`) |
| **Q5** | Dockerfile Python base-image tag | `python:3.11.15-slim-bookworm` |
| **Q6** | Docker container `/predict` probability | `0.770` (`0.769799`) |
