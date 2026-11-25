# Employee Workforce Segmentation

Unsupervised workforce segmentation using **Principal Component Analysis (PCA)**, **K-Means clustering**, and **Hierarchical (Ward) clustering** to identify distinct employee personas, pinpoint high-attrition risk groups, quantify regretted attrition among high performers, and support targeted HR retention strategies.

> **Leakage-Free Design:** `Attrition`, `Department`, and `EmployeeID` are deliberately **excluded** from feature scaling, PCA, and clustering. They are joined back afterwards solely to interpret and validate the discovered segments.

---

## Key Findings

- **4 Distinct Workforce Segments:** Reducing 12 standardized HR features to **7 principal components** retains **86.1%** of total variance and yields four stable clusters (`k = 4`) across both K-Means and Ward Hierarchical clustering (**Adjusted Rand Index = 0.738**).
- **Concentrated Attrition Risk:** Overall company attrition is **16.1%** (236 of 1,470 employees), but the **Overworked & Disengaged** segment (21.4% of staff, 314 employees) has a **43.0%** attrition rate (**2.68x** company average) and accounts for **57.2%** of all leavers (135 of 236).
- **Actionable Employee Watch-List:** Flagging employees who simultaneously work overtime (`OverTime = Yes`), report low job satisfaction (`JobSatisfaction <= 2`), and have poor work-life balance (`WorkLifeBalance <= 2`) isolates **188 employees** with **45.7%** attrition—nearly four times the **11.7%** rate of the remaining 1,282 employees. Of those 188, **166** belong to *Overworked & Disengaged*.
- **Segment Matters More Than Department:** Department-level attrition varies only modestly (**Sales: 17.7%**, **Human Resources: 17.6%**, **Research & Development: 15.2%**), whereas *Overworked & Disengaged* employees leave at **41.5%–52.6%** across every department.
- **Regretted Attrition Among High Performers:** High performers (`PerformanceRating >= 4`) have lower attrition overall (**8.4%**) than Solid (**20.7%**) or Low (**17.1%**) performers, yet **23 of 55** high performers in *Overworked & Disengaged* left (**41.8%**), compared to just **8 of 339** (**2.4%**) in *Engaged High Performers*.

---

## Project Structure

```text
.
├── Employee_Workforce_Segmentation.ipynb   # End-to-end analysis notebook (code, tables, plots, exports)
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies
├── dashboard/
│   └── workforce_dashboard.html            # Standalone interactive Chart.js analytics dashboard
├── data/
│   ├── generate_employee_data.py           # Reproducible synthetic IBM HR-Analytics style data generator
│   └── employee_data.csv                   # Generated dataset (1,470 employees, 16 columns)
├── figures/
│   ├── 01_eda.png                          # Distributions, department headcount, and attrition drivers
│   ├── 02_pca_scree_loadings.png           # PCA scree plot, cumulative variance, and PC1–PC4 loadings
│   ├── 03_kmeans_selection.png             # Elbow, Silhouette, and Davies-Bouldin curves for k = 2..8
│   ├── 04_dendrogram.png                   # Ward hierarchical dendrogram (last 30 merges, cut at k = 4)
│   ├── 05_pca_clusters.png                 # K-Means vs. Ward clusters and attrition in PC1–PC2 space
│   ├── 06_segment_profile_heatmap.png      # Standardized (z-score) feature profiles per segment
│   ├── 07_attrition_by_segment.png         # Segment attrition rates and share of total leavers
│   ├── 08_department_segments.png          # Workforce segment composition inside each department
│   └── 09_performance_groups.png           # Performance mix per segment and high-performer attrition
├── outputs/
│   ├── dashboard_data.json                 # Aggregated metrics and sampled PCA coordinates for dashboard
│   ├── employee_segments.csv               # Employee-level cluster, risk tier, performance, and watchlist labels
│   ├── pca_loadings_top4.csv               # Feature loadings for PC1–PC4
│   ├── pca_summary.csv                     # Eigenvalues and explained/cumulative variance for PC1–PC12
│   └── segment_profile.csv                 # Mean feature values, headcount, share %, and attrition % by cluster
└── report/
    └── Employee_Segmentation_Report.docx   # Formatted executive report with findings and recommendations
```

---

## Methodology

### 1. Dataset & Preprocessing
- **Dataset:** [`data/employee_data.csv`](data/employee_data.csv) contains **1,470 records** and **16 columns** with zero missing values and zero duplicate rows, modeled after the IBM HR Analytics schema (generated via [`data/generate_employee_data.py`](data/generate_employee_data.py) with `seed = 42`).
- **Clustering Features (12):** `Age`, `JobLevel`, `MonthlyIncome` (transformed with `np.log1p` to correct right-skewness), `YearsAtCompany`, `YearsSinceLastPromotion`, `DistanceFromHome`, `OverTime` (binary `0/1`), `JobSatisfaction`, `EnvironmentSatisfaction`, `WorkLifeBalance`, `PerformanceRating`, and `TrainingTimesLastYear`.
- **Excluded Columns:** `TotalWorkingYears` is dropped due to high collinearity with `Age`; `EmployeeID`, `Department`, and `Attrition` are withheld from modeling.
- **Scaling:** All 12 clustering features are standardized (`StandardScaler`) to mean `0` and standard deviation `1`.

### 2. Dimensionality Reduction (PCA)
PCA is fitted on the 12 standardized features. Retaining components until cumulative explained variance reaches **>= 85%** selects **7 principal components (86.08% cumulative variance)**:

| Component | Eigenvalue | Variance % | Cumulative % | Primary Drivers (Loadings) |
|---|---|---|---|---|
| **PC1** | 3.385 | 28.19% | 28.19% | **Seniority & Pay:** `YearsAtCompany` (+0.47), `Age` (+0.47), `MonthlyIncome` (+0.45), `JobLevel` (+0.41) |
| **PC2** | 2.365 | 19.69% | 47.89% | **Engagement & Performance:** `PerformanceRating` (+0.45), `JobSatisfaction` (+0.43), `EnvironmentSatisfaction` (+0.35), `WorkLifeBalance` (+0.33) |
| **PC3** | 1.259 | 10.48% | 58.37% | **Workload vs. Balance:** `OverTime` (+0.53), `WorkLifeBalance` (-0.42), `JobLevel` (+0.36), `MonthlyIncome` (+0.33) |
| **PC4–PC7** | 0.998–0.705 | 8.31%–5.87% | 86.08% | Commute distance (`DistanceFromHome` +0.97 on PC4) and residual variance |

Clustering in the 7-PC space produces nearly identical assignments to clustering on all 12 standardized features (**ARI = 0.989**).

### 3. Clustering & Validation
- **Choosing `k`:** K-Means (`n_init = 20`, `random_state = 42`) was evaluated across `k = 2..8`. Within the actionable range `k = 3..6`, the pre-specified selection rule chooses the largest `k` whose Silhouette score is within `0.01` of the maximum: `k = 3` (Silhouette `0.271`, Davies-Bouldin `1.367`) and `k = 4` (Silhouette `0.268`, Davies-Bouldin `1.371`) are effectively tied, so **`k = 4`** is selected because `k = 3` merges early-career staff and overworked staff into a single group.
- **Algorithm Comparison (`k = 4`):**

| Method | Silhouette (↑) | Calinski-Harabasz (↑) | Davies-Bouldin (↓) | Notes |
|---|---|---|---|---|
| **K-Means** (`n_init = 50`) | **0.268** | **504.436** | **1.371** | Final model; 10-seed stability ARI = **1.000**; supports `.predict()` |
| **Hierarchical (Ward)** | 0.233 | 434.742 | 1.624 | Validation model; agrees with K-Means (**ARI = 0.738**) |

### 4. Rule-Based Segment Naming & Risk Tiering
Cluster centroids are named automatically from standardized profile scores and assigned retention-risk tiers relative to the **16.1%** company attrition rate:
- **High Risk:** `>= 1.25x` company average (`>= 20.1%`)
- **Medium Risk:** `0.85x` to `< 1.25x` company average (`13.7%` to `< 20.1%`)
- **Low Risk:** `< 0.85x` company average (`< 13.7%`)

---

## Segment Profiles

| Segment | Employees (Share) | Attrition % (Leavers) | Share of All Leavers | Overtime % | Avg Job Sat. (1–4) | Avg WLB (1–4) | Avg Age / Tenure | Avg Monthly Income | Yrs Since Promo | Avg Perf. (2–5) | Risk Tier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Overworked & Disengaged** | 314 (21.4%) | **43.0%** (135) | **57.2%** | **90.1%** | 1.94 | 1.84 | 35.3 yrs / 5.1 yrs | 8,740 | 2.8 yrs | 2.90 | High risk |
| **Early-Career Starters** | 426 (29.0%) | **14.3%** (61) | 25.8% | 25.1% | 2.75 | 2.86 | 26.6 yrs / 1.9 yrs | 5,888 | 0.7 yrs | 2.84 | Medium risk |
| **Stagnant Veterans** | 301 (20.5%) | **9.6%** (29) | 12.3% | 11.0% | 2.33 | 2.69 | 49.0 yrs / 18.0 yrs | 11,208 | 8.9 yrs | 2.45 | Low risk |
| **Engaged High Performers** | 429 (29.2%) | **2.6%** (11) | 4.7% | 15.2% | 3.41 | 3.12 | 38.5 yrs / 7.9 yrs | 12,005 | 1.6 yrs | 4.04 | Low risk |

### Recommended HR Actions
1. **Overworked & Disengaged (Immediate Priority):** Audit overtime allocation and workload distribution; initiate manager retention check-ins for the **188 watch-list employees** (166 in this segment); prioritize the **55 high performers** in this group (41.8% attrition) for compensation/workload intervention.
2. **Early-Career Starters:** Strengthen structured onboarding, mentorship, and transparent pay/promotion pathways—they represent the second-largest source of departures (**25.8%** of leavers).
3. **Stagnant Veterans:** Although attrition is low (9.6%), an average of **8.9 years since promotion** and **54.8% Low performance ratings** (`<= 2`) indicate plateauing and skill-currency risk; introduce reskilling programs and lateral mobility tracks.
4. **Engaged High Performers:** Safeguard manageable overtime (15.2%) and strong work-life balance (3.12/4); engage these employees (79.0% rated `>= 4`) as peer mentors for early-career staff.

---

## Interactive Dashboard & Deliverables

- **Interactive Dashboard ([`dashboard/workforce_dashboard.html`](dashboard/workforce_dashboard.html)):** Open directly in any modern browser (requires internet access to load Chart.js 4.4.1 from CDN; data is embedded from `outputs/dashboard_data.json`). Includes KPI cards, the segment risk table, an interactive **PC1 vs. PC2 scatter plot** with a *"show leavers only"* toggle, department mix, performance distribution, PCA scree curve, and regretted attrition charts.
- **Executive Report ([`report/Employee_Segmentation_Report.docx`](report/Employee_Segmentation_Report.docx)):** Full written report covering executive summary, PCA tables, clustering validation, department/performance breakdowns, recommendations, and limitations.
- **Exports (`outputs/`):** Machine-readable CSVs (`employee_segments.csv`, `segment_profile.csv`, `pca_summary.csv`, `pca_loadings_top4.csv`) and `dashboard_data.json`.

---

## Getting Started

### Prerequisites
- Python 3.9+
- Dependencies listed in [`requirements.txt`](requirements.txt): `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `jupyter`

### Installation & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate the synthetic dataset (data/employee_data.csv)
python data/generate_employee_data.py

# 3. Launch and run the analysis notebook
jupyter notebook Employee_Workforce_Segmentation.ipynb
```

Running all cells in `Employee_Workforce_Segmentation.ipynb` regenerates all 9 charts in `figures/` and all 5 output files in `outputs/`.

> **Using Real HR Data:** To run this pipeline on an authentic IBM HR Analytics dataset, replace `data/employee_data.csv` with your CSV containing the 12 feature columns plus `EmployeeID`, `Department`, and `Attrition`, then re-run `Employee_Workforce_Segmentation.ipynb`.

---

## Limitations

- **Synthetic Data:** `data/employee_data.csv` was synthetically generated to match the IBM HR-Analytics schema because no external dataset file was supplied; results demonstrate the analytical workflow rather than empirical findings from a real organization.
- **Associational Findings:** Clustering identifies correlated workforce patterns and does not establish causal relationships between overtime/satisfaction and turnover.
- **Small Subgroup Sizes:** Cross-tabulations in small departments (e.g., HR within *Overworked & Disengaged*, `n = 19`) are indicative only.
- **Fixed `k = 4` Naming Rules:** The automated segment-naming logic assumes at least four clusters and would require adjustment if `k` changes on a different dataset.
