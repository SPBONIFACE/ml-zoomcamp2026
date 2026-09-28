# Module 1: Introduction to Machine Learning

This folder contains the notes, code implementations, and homework assignment for **Module 1** of the [Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp).

---

## 📑 Module Overview

Module 1 introduces the core principles of Machine Learning Engineering and sets up the foundational technical environment.

### Key Concepts Covered:
* **Machine Learning vs. Rule-Based Systems**: Understanding when ML is appropriate vs. hardcoded conditional logic.
* **Supervised Machine Learning**: Features, target variables, training, and prediction fundamentals.
* **CRISP-DM Framework**: The standard iterative workflow for structuring ML projects (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment).
* **Environment & Tools Setup**: Configuring Python 3.11, Conda environments, Jupyter Notebooks, and key data science libraries.
* **Refresher Libraries**:
  * **NumPy**: Vectorized operations, array manipulation, dot products, transposes, matrix inversion (`np.linalg.inv`).
  * **Pandas**: DataFrames, reading CSV files, handling missing values (`fillna`, `isnull`), indexing, `groupby`, and statistical operations.
  * **Linear Algebra**: Closed-form Linear Regression equation $w = (X^T X)^{-1} X^T y$.

---

## 📁 Directory Files

* [`ML Zoomcamp Homework 1.ipynb`](./ML%20Zoomcamp%20Homework%201.ipynb): Complete solution notebook for Homework 1.
* [`homework-1-intro.md`](./homework-1-intro.md): Official homework assignment text and guidelines.
* [`README.md`](./README.md): Documentation for Module 1.

---

## 📝 Homework 1 Summary & Solutions

The homework uses the **2026 Car Fuel Efficiency Dataset** (`car_fuel_efficiency.csv` / `car_fuel_efficiency_2026.csv`).

| Question | Topic | Pandas / NumPy Operation | Answer |
| :--- | :--- | :--- | :---: |
| **Q1** | Pandas version | `pd.__version__` | `2.3.2` |
| **Q2** | Records count | `len(df)` or `df.shape[0]` | `9704` |
| **Q3** | Fuel types count | `df['fuel_type'].nunique()` | `2` |
| **Q4** | Missing value columns | `(df.isnull().sum() > 0).sum()` or `df.isnull().any().sum()` | `4` |
| **Q5** | Max fuel efficiency (Asia) | `df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()` | `23.75` |
| **Q6** | Horsepower median change | `df['horsepower'].fillna(mode_val)` | `Increased` (from 149.0 to 152.0) |
| **Q7** | Sum of weights ($w = (X^T X)^{-1} X^T y$) | Matrix inversion & multiplication using NumPy | Calculated in notebook |

---

## 🚀 How to Run the Notebook

1. Ensure the `ml-zoomcamp` Conda environment is activated:
   ```bash
   conda activate ml-zoomcamp
   ```

2. Launch Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter notebook "ML Zoomcamp Homework 1.ipynb"
   ```

3. Execute cells sequentially.
