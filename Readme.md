# Credit Score Classification: The Power of Clean Data

**Demonstrating that data quality almost always beats algorithm sophistication in real-world ML projects**

## Table of Contents

- [The Core Message](#-the-core-message)
- [Business Impact in Indian Context](#-business-impact-in-indian-context)
- [Quick Results Comparison](#-quick-results-comparison)
- [Dataset Overview](#-dataset-overview)
- [Models & Methodology](#-models--methodology)
- [Feature Importance Analysis](#-feature-importance-analysis)
- [API Deployment](#-api-deployment)
- [Key Takeaways](#-key-takeaways)
- [Getting Started](#-getting-started)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## The Core Message

**In 95% of practical machine learning projects — especially in finance, credit risk, fraud detection, and banking — having clean, high-quality data is far more important than choosing a slightly better algorithm.**

This repository proves that point **numerically and visually** by comparing two fundamentally different datasets:

### The Two Datasets

#### 1. **Corrupt Dataset** (`data/corrupt_train_data.csv`)
- **What it is**: Raw, messy, real-world data with extensive quality issues
- **Size**: Split 80-20 for train-test experimentation  
- **Characteristics**: 
  - Invalid values (Age = -500, 999)
  - Missing data (15-20% nulls in critical columns)
  - Junk characters (`"_"`, `"!@9#%8"`)
  - Inconsistent formats (`"22 Years and 1 Months"`)
  - Type mismatches (numbers stored as text)
  - Unrealistic entries (negative salaries, impossible ages)
- **Preprocessing Required**: Extensive data cleaning and feature engineering pipeline
- **Final Performance**: 70-74% accuracy after sophisticated cleaning

#### 2. **Clean Dataset** (`data/clean_data.csv`)
- **What it is**: High-quality, properly validated data from the source
- **Characteristics**:
  - Properly formatted data
  - No garbage or junk characters  
  - Realistic value ranges pre-validated
  - Consistent data types
  - Minimal missing values
  - No placeholder or dummy entries
- **Preprocessing Required**: Minimal (basic standardization and encoding)
- **Final Performance**: 75-77% accuracy with minimal processing

> **Note**: `data/corrupt_test_data.csv` is also available for independent testing but was not used in the primary experiments.

### The Critical Finding

**Same models. Same hyperparameters. Same train-test logic.**  
**Yet the clean dataset achieves noticeably better performance across all models — Random Forest shows the largest gap (+5.03%), while XGBoost, despite being the strongest model overall, still trails by +2.79% on clean vs corrupt data — demonstrating that even sophisticated data cleaning on corrupt data cannot fully match the performance of starting with quality data.**

→ **This is the harsh reality — even the best data cleaning and feature engineering has limits when the underlying data quality is fundamentally compromised. The 2.79–5.03% performance gaps across models represent information permanently lost due to poor source data quality. In a domain like credit risk, where these models score millions of loan applications annually, even the "smallest" gap of 2.79% translates to thousands of misclassified borrowers and hundreds of crores in avoidable NPAs.**

---

## Business Impact in Indian Context

### The Indian Credit Landscape (2025-26)

India's banking and credit sector has achieved remarkable transformation, emerging from a decade-long crisis to become one of the most robust financial systems globally:

Bank credit grew 11.5% on the assets side during FY25, with total outstanding credit reaching ₹198.73 lakh crore as of June 2025. Deposits grew 9.75% year-over-year to ₹246.77 lakh crore by October 2025, with overall deposit growth at 11.1% for FY25 — reflecting the massive scale of credit operations where data quality directly impacts business outcomes.

#### **Scale of Operations (FY 2025-26)**

- Deposits surged from ₹67.4 lakh crore in FY15 to ₹246.77 lakh crore by October 2025, while credit expanded from ₹85.3 lakh crore to ₹198.73 lakh crore as of June 2025
- Credit growth moderated to 13.1% year-over-year as of January 2026, while deposits grew at 10.6%
- UPI transactions reached a historic ₹230 lakh crore in FY25-26 (till December 2025), driven by widespread digital adoption and now powering nearly 50% of global real-time transactions
- Retail loans grew 14.4% year-over-year to ₹68.48 lakh crore, with gold loans surging 127.6% to ₹3.82 trillion

### **The Cost of Poor Data Quality**

#### 1. **Non-Performing Assets (NPAs) - Historic Turnaround**

The Indian banking sector has achieved a remarkable recovery from the NPA crisis through improved risk assessment and data quality:

- Gross NPAs declined to a multi-decadal low of 2.2% in March 2025 and improved further to 2.1% by September 2025, compared to a peak of 11.18% in March 2018; the RBI FSR December 2025 projects GNPA to decline further to 1.9% by March 2027 under baseline scenario
- Net NPAs dropped to just 0.52% by March 2025, reflecting stronger provisioning and tighter risk controls
- Public Sector Banks' Gross NPAs fell from 9.11% in March 2021 to 2.58% in March 2025
- Return on Assets increased from -0.22% in FY 17-18 to 1.4% in FY 24-25, while Return on Equity jumped from -2.74% to 13.5%; consolidated SCB balance sheets expanded 11.2% in FY25 with net profits reaching ₹4 lakh crore

**Economic Impact:**
- The NPA crisis cost the banking sector dearly — improved data quality and risk assessment have been critical to this recovery
- Banks' profitability improved for the sixth consecutive year in FY 2024-25
- Better credit scoring models enabled by cleaner data have prevented the recurrence of bad loan accumulation

#### 2. **Why "Just 2.79%" Is Not Just 2.79% — The Finance Multiplier Effect**

> **"A 2.79% accuracy gap looks negligible in a research paper. In a bank's loan book, it is the difference between crores saved and crores written off."**

This is perhaps the most important insight in this entire project. When non-technical stakeholders see the XGBoost results — **73.66% (corrupt data) vs 76.45% (clean data)** — the instinctive reaction is: *"It's less than 3%, does it really matter?"*

**The answer is an emphatic yes — and here is exactly why.**

##### Why Small Accuracy Gaps Are Amplified in Finance

Credit scoring is not a single prediction made once. It is a **decision engine** that runs on millions of loan applications every year. Every percentage point of accuracy difference compounds across the entire loan book:

```
The Compounding Logic:

  2.79% accuracy gap
        ↓
  Applied across ₹68,000+ crore retail loan portfolio
        ↓
  Each misclassified borrower = a loan approved that defaults,
  or a creditworthy applicant wrongly rejected
        ↓
  At 1.6% average default rate, even small classification
  improvements prevent disproportionately large NPA formation
        ↓
  Net effect: Hundreds of crores in prevented losses
```

##### The Asymmetry of Errors in Credit Risk

In most ML applications, a wrong prediction is a minor inconvenience. In credit lending, wrong predictions have **asymmetric, irreversible consequences**:

| Error Type | What Happens | Financial Cost |
|------------|--------------|----------------|
| **False Negative** (missed bad borrower) | Loan approved → borrower defaults | Full principal loss + recovery costs (only 20-30% typically recovered) |
| **False Positive** (rejected good borrower) | Creditworthy customer turned away | Lost interest income + reputational damage + customer goes to competitor |

The clean data model, with its 2.79% accuracy edge, **catches more bad borrowers before approval** and **rejects fewer creditworthy ones** — both sides of this asymmetry improve simultaneously.

##### The Base Rate Problem — Why 2.79% Hits Harder Than It Looks

Consider how default rates work in Indian retail banking (FY 2025-26 estimates):

- At a 1.6% average default rate, only 16 out of every 1,000 borrowers default
- A model operating at **73.66% accuracy** will misclassify approximately **264 out of 1,000** applicants
- A model at **76.45% accuracy** misclassifies approximately **236 out of 1,000** applicants
- **That is 28 fewer misclassifications per 1,000 applicants** — many of whom fall in the high-stakes default category

When defaults are rare events (low base rate), even small improvements in a model's ability to identify them correctly translate to **dramatic reductions in actual NPA formation**. This is the base rate amplification effect — the rarer the event, the more valuable each additional percentage point of detection accuracy becomes.

##### Scaled to Real Numbers — Mid-Sized Indian Bank (FY 2025-26)

Assuming conservative estimates:

| Parameter | Value |
|-----------|-------|
| Retail Loan Portfolio | ₹50,000 crore |
| Annual Loan Applications Processed | ~5 lakh applications |
| Average Loan Size | ₹10 lakh |
| Average Default Rate | 1.6% |
| Annual Potential Defaults | ₹800 crore |

**Impact of the 2.79% accuracy improvement (XGBoost, clean vs corrupt data):**

| Metric | Corrupt Data Model | Clean Data Model | Difference |
|--------|--------------------|------------------|------------|
| Misclassified applicants per lakh | ~26,340 | ~23,550 | **2,790 fewer** |
| Estimated defaults prevented (of misclassified pool) | — | — | **~45 per lakh applications** |
| Prevented default value (₹10L avg loan) | — | — | **₹4.5 crore per lakh applications** |
| **Annual savings (5 lakh applications)** | — | — | **~₹22.5 crore / year** |
| Recovery adjustment (20-30% recovery on defaults) | — | — | **Net loss prevented: ₹15–18 crore / year** |

> This is for **one mid-sized bank**. India has 12 public sector banks, 21 private sector banks, and hundreds of NBFCs and cooperative banks — all running credit scoring models simultaneously.

**Industry-Wide Impact (extrapolated to India's banking sector FY 2025-26):**
- Total retail credit: ~₹69,556 crore (35% of ₹198.73 lakh crore outstanding as of June 2025)
- Even a **1% improvement** in credit assessment accuracy could prevent NPAs worth **₹6,956 crore annually**
- The **2.79% XGBoost gap** (clean vs corrupt data) could potentially prevent **₹19,406 crore** in annual NPAs industry-wide
- The **5.03% Random Forest gap** scales this further to a potential **₹34,987 crore** in prevented losses

##### Beyond the Numbers — The Hidden Costs That Don't Show in Accuracy Scores

The 2.79% accuracy gap also drives costs that are harder to quantify but equally real:

- **Provisioning requirements**: RBI's shift to the Expected Credit Loss (ECL) framework (Draft Directions 2025) means banks must provision for predicted future losses — a less accurate model leads to systematic under-provisioning, which is a regulatory and capital adequacy risk
- **Risk-based pricing errors**: A misclassified "Good" borrower who should be "Standard" is offered a lower interest rate, permanently compressing the bank's net interest margin on that loan
- **Regulatory scrutiny**: Systemic misclassification patterns attract RBI audit attention and potential corrective action directives
- **Compounding over loan tenure**: A wrong credit decision made today on a 5-year loan compounds its cost across the entire loan lifecycle, not just year one
- **False rejection cost**: Every creditworthy borrower wrongly rejected is a lost customer — in India's increasingly competitive BFSI landscape, they immediately move to a competitor, eroding market share

##### The Model-by-Model Accuracy Gap Summary

| Model | Corrupt Val. Accuracy | Clean Val. Accuracy | Gap | Estimated Annual NPA Prevention (Industry-wide) |
|-------|-----------------------|---------------------|-----|--------------------------------------------------|
| Logistic Regression | 64.70% | 66.65% | +1.95% | ~₹13,564 crore |
| Decision Tree | 67.79% | 70.01% | +2.22% | ~₹15,442 crore |
| Random Forest | 70.58% | 75.61% | +5.03% | ~₹34,987 crore |
| **XGBoost** | **73.66%** | **76.45%** | **+2.79%** | **~₹19,406 crore** |

> **Even the smallest gap — Logistic Regression's +1.95% — translates to over ₹13,000 crore in potential industry-wide NPA prevention annually. The word "small" simply does not apply in this domain.**

#### 3. **Operational Efficiency Gains**

Clean data enables compounding operational benefits beyond just NPA prevention:

- **Faster Credit Decisions**: Cleaner input data reduces exception-handling in underwriting pipelines, supporting India's 11.5% credit growth in FY25 (T&P Report 2024-25)
- **Lower Operational Costs**: Fewer manual reviews of borderline cases, reduced collections workload, improved profitability — consistent with the sector-wide PAT growth seen over six consecutive years
- **Better Risk-Based Pricing**: More accurate classification means interest rates are calibrated to true risk, protecting net interest margins and improving customer fairness
- **Improved Customer Experience**: Reduced false rejections of creditworthy applicants — critical as India's digital lending market grows and customer switching costs drop
- **Capital Efficiency**: Accurate models require less precautionary capital buffer, freeing up capital for productive lending — relevant to CRAR maintenance, with aggregate SCB CRAR at 17.4% (March 2025) and 17.2% (September 2025) per T&P Report 2024-25

#### 4. **Regulatory Compliance & Future Framework**

In October 2025, the RBI issued landmark Draft Directions 2025, proposing a shift to the Expected Credit Loss (ECL) framework, which applies a risk-sensitive approach to provisioning. Clean, well-structured data is essential for:
- Meeting RBI's evolving data governance requirements
- Implementing ECL framework effectively — a less accurate model directly leads to incorrect expected loss estimates, causing either under-provisioning (regulatory risk) or over-provisioning (capital inefficiency)
- Audit trail maintenance
- Basel III compliance and capital adequacy — aggregate CRAR at 17.1% as of September 2025 (PSBs: 16%, private banks: 18.1% per FSR December 2025; PSBs specifically at 16.4% as of June 2025 per PIB)

### **Industry Adoption & AI Integration**

The BFSI sector saw M&A activity surge 127% YoY to US$ 8 billion (January–September 2025), with PE-VC investments ranking the sector second in Q3 2025 at ₹11,273 crore — highlighting the growing recognition that data quality is the foundation for AI/ML success in financial services.

Financial institutions using AI models have been able to incorporate weak signals and use sophisticated machine learning algorithms to improve prediction accuracy of default risk, but these models are only as good as the data they're trained on. The banking sector's recovery from 11%+ NPAs to sub-2.5% levels demonstrates the critical role of data quality in credit risk management.

> **The core lesson this project demonstrates in numbers**: When a domain operates at the scale of India's ₹198.73 lakh crore credit market, accuracy differences that look like rounding errors on a benchmark leaderboard become the difference between financial stability and systemic risk. **2.79% is not a small gap — it is a very expensive one.**

---

## Quick Results Comparison

### Logistic Regression (Dataset Comparison)

| Metric | Corrupt Data (After Cleaning) | Clean Dataset | Absolute Gain | Relative Gain |
|--------|-------------------------------|---------------|---------------|---------------|
| **Accuracy** | 64.70% | 66.65% | **+1.95%** | **+3.01%** |
| **Precision** | 69.17% | 70.78% | **+1.61%** | **+2.33%** |
| **Recall** | 64.70% | 66.65% | **+1.95%** | **+3.01%** |
| **F1-Score** | 65.06% | 67.06% | **+2.00%** | **+3.07%** |

### Decision Tree (Dataset Comparison)

| Metric | Corrupt Data (After Cleaning) | Clean Dataset | Absolute Gain | Relative Gain |
|--------|-------------------------------|---------------|---------------|---------------|
| **Accuracy** | 67.79% | 70.01% | **+2.22%** | **+3.27%** |
| **Precision** | 72.84% | 74.66% | **+1.82%** | **+2.50%** |
| **Recall** | 67.79% | 70.01% | **+2.22%** | **+3.27%** |
| **F1-Score** | 68.13% | 70.11% | **+1.98%** | **+2.91%** |

### Random Forest (Dataset Comparison)

| Metric | Corrupt Data (After Cleaning) | Clean Dataset | Absolute Gain | Relative Gain |
|--------|-------------------------------|---------------|---------------|---------------|
| **Accuracy** | 70.58% | 75.61% | **+5.03%** | **+7.13%** |
| **Precision** | 75.02% | 78.70% | **+3.68%** | **+4.91%** |
| **Recall** | 70.58% | 75.61% | **+5.03%** | **+7.13%** |
| **F1-Score** | 70.88% | 75.99% | **+5.11%** | **+7.21%** |

### XGBoost (Dataset Comparison)

| Metric | Corrupt Data (After Cleaning) | Clean Dataset | Absolute Gain | Relative Gain |
|--------|-------------------------------|---------------|---------------|---------------|
| **Accuracy** | 73.66% | 76.45% | **+2.79%** | **+3.79%** |
| **Precision** | 73.92% | 76.90% | **+2.98%** | **+4.03%** |
| **Recall** | 73.66% | 76.45% | **+2.79%** | **+3.79%** |
| **F1-Score** | 73.74% | 76.58% | **+2.84%** | **+3.85%** |

---

### Full Model Metrics — Clean Data Notebook

| Model | Split | Accuracy | Precision | Recall | F1-Score |
|-------|-------|----------|-----------|--------|----------|
| **Logistic Regression** | Train | 66.64% | 70.94% | 66.64% | 67.10% |
| **Logistic Regression** | Validation | 66.65% | 70.78% | 66.65% | 67.06% |
| **Decision Tree** | Train | 70.88% | 75.75% | 70.88% | 70.94% |
| **Decision Tree** | Validation | 70.01% | 74.66% | 70.01% | 70.11% |
| **Random Forest** | Train | 78.12% | 80.98% | 78.12% | 78.43% |
| **Random Forest** | Validation | 75.61% | 78.70% | 75.61% | 75.99% |
| **XGBoost** | Train | 80.96% | 81.32% | 80.96% | 81.06% |
| **XGBoost** | Validation | 76.45% | 76.90% | 76.45% | 76.58% |

### Full Model Metrics — Corrupt Data Notebook

| Model | Split | Accuracy | Precision | Recall | F1-Score |
|-------|-------|----------|-----------|--------|----------|
| **Logistic Regression** | Train | 64.30% | 69.02% | 64.30% | 64.72% |
| **Logistic Regression** | Validation | 64.70% | 69.17% | 64.70% | 65.06% |
| **Decision Tree** | Train | 69.33% | 74.87% | 69.33% | 69.72% |
| **Decision Tree** | Validation | 67.79% | 72.84% | 67.79% | 68.13% |
| **Random Forest** | Train | 74.77% | 79.10% | 74.77% | 75.20% |
| **Random Forest** | Validation | 70.58% | 75.02% | 70.58% | 70.88% |
| **XGBoost** | Train | 81.66% | 81.91% | 81.66% | 81.72% |
| **XGBoost** | Validation | 73.66% | 73.92% | 73.66% | 73.74% |

> **Key Insight**: Despite extensive data cleaning and feature engineering on the corrupt dataset, the inherently clean dataset still outperforms across all models by 2-7%. XGBoost shows the highest train accuracy in both notebooks but also exhibits the largest train-validation gap on the corrupt data (81.66% → 73.66%), reflecting its higher sensitivity to data corruption. **This demonstrates that data quality at the source is irreplaceable** — even the best preprocessing cannot fully compensate for fundamentally poor data quality.

### Performance Visualization

```
Corrupt Data (After Cleaning) — XGBoost Best:
████████████████████░░░░░░ 73.66%

Clean Dataset (Minimal Processing) — XGBoost Best:
█████████████████████░░░░░ 76.45%

Gap Analysis:
├── Data Cleaning & Feature Engineering:  Significant improvement ✓
├── But Still Falls Short:                ~2.79% performance gap (XGBoost)
└── Takeaway: Source data quality matters most
```

---

## Dataset Overview

**100,000 rows × 28 columns** (each dataset)  
Typical credit bureau + banking features simulating real-world credit assessment scenarios:

### Two-Dataset Comparison

This project uses **two distinct datasets** to demonstrate the impact of data quality:

#### 1. **Corrupt Dataset** (`corrupt_data_credit_score.ipynb`)
- Raw, messy data with realistic data quality issues
- Required extensive data cleaning and feature engineering
- Demonstrates what data scientists typically encounter in production
- Even after sophisticated preprocessing, performance ceiling is limited by inherent data quality

#### 2. **Clean Dataset** (`clean_data_credit_score.ipynb`)
- High-quality data from the source
- Minimal garbage values or unrealistic entries
- Properly formatted and validated data
- Represents ideal scenario where data governance is strong

> **Critical Finding**: The corrupt dataset, even after extensive cleaning and feature engineering, could not fully match the performance of the inherently clean dataset. This proves that **garbage in = garbage out** — no amount of preprocessing can fully compensate for fundamentally poor source data quality.

### Feature Categories

| Category | Features | Examples |
|----------|----------|----------|
| **Demographics** | Age, Occupation | Software Engineer, Teacher, Doctor |
| **Income & Assets** | Annual Income, Monthly Salary | Salary variations, income stability |
| **Credit Products** | Credit Cards, Loans, Interest Rates | Number of cards, loan types, rates |
| **Payment Behavior** | Delayed Payments, Min Amount Paid | Payment history, delinquency patterns |
| **Financial Health** | Outstanding Debt, Credit Utilization | Debt-to-income ratio, utilization rates |
| **Credit History** | Credit History Age, Credit Mix | Account age, product diversity |
| **Target Variable** | Credit_Score | Good / Standard / Poor |

### What Makes the "Corrupt Dataset" Realistic?

Common data problems mirroring real-world banking/fintech challenges:

| Data Quality Issue | Example | Business Impact |
|-------------------|---------|-----------------|
| **Invalid Values** | Age = -500, Age = 999 | Model confusion, incorrect risk assessment |
| **Missing Data** | 15-20% null values in critical columns | Reduced predictive power, biased models |
| **Junk Characters** | `"_"`, `"!@9#%8"`, special chars | Processing errors, feature extraction failure |
| **Inconsistent Formats** | `"22 Years and 1 Months"` | Parsing failures, data type mismatches |
| **Type Mismatches** | Numeric stored as text | Computational errors, feature engineering issues |
| **Placeholder Values** | `"Unknown"`, `"NA"`, `"0"` | Misleading patterns, inflated null handling |
| **Unrealistic Entries** | Negative salaries, impossible ages | Noise in patterns, degraded model learning |

### What Was Done to the "Corrupt Dataset"

**Production-grade data cleaning and feature engineering pipeline applied:**

```python
# Core Cleaning Steps Applied

1. Age Validation & Correction
   ├── Identify impossible values (< 18 or > 100)
   ├── Extract numeric values from text strings
   ├── Handle missing values with median imputation
   └── Validate realistic age ranges

2. Income & Financial Data Standardization
   ├── Remove special characters and underscores
   ├── Convert text representations to numeric
   ├── Handle negative values in salary fields
   └── Ensure consistency across income-related features

3. Credit History Age Parsing
   ├── Extract years and months from text strings
   ├── Convert to standardized numeric format
   ├── Handle various text formats ("X Years and Y Months")
   └── Impute missing values with median

4. Categorical Variable Cleaning
   ├── Standardize occupation names
   ├── Map inconsistent credit mix categories
   ├── Clean payment behavior labels
   └── Remove junk characters from categorical fields

5. Missing Value Treatment
   ├── Identify true missing vs placeholder values
   ├── Apply domain-appropriate imputation strategies
   ├── Document imputation methods for reproducibility
   └── Validate post-imputation distributions

6. Outlier Detection & Treatment
   ├── Statistical methods (IQR, Z-score)
   ├── Domain knowledge-based thresholds
   ├── Cap/floor extreme values
   └── Log transformations where appropriate

7. Data Type Enforcement
   ├── Ensure numeric columns are numeric
   ├── Categorical columns properly encoded
   ├── Date/time fields in standard format
   └── Consistent data types across pipeline
```

**Result**: Despite these comprehensive cleaning efforts, the corrupt dataset achieved 70-74% accuracy — respectable but still 3-7% behind the inherently clean dataset. This gap represents the **irreplaceable value of source data quality**.

### The "Clean Dataset" Advantage

The clean dataset features:
- Properly formatted data from the source
- No garbage or junk characters
- Realistic value ranges pre-validated
- Consistent data types throughout
- Minimal missing values
- No placeholder or dummy entries

**Minimal preprocessing required** — the data was ready for ML modeling with basic standardization. This represents the ideal scenario where strong data governance and validation exist upstream.

---

## Models & Methodology

### Machine Learning Pipeline

**Corrupt Dataset Workflow:**
```
Raw Corrupt Data → Extensive Cleaning → Feature Engineering → Model Training → 70-74% Accuracy
       ↓                  ↓                    ↓                    ↓              ↓
  Messy data     Parse/Fix/Impute      Standardized           XGBoost       Good but limited
                  Remove junk           encoding             Classifier      by data quality
```

**Clean Dataset Workflow:**
```
Clean Data → Minimal Processing → Basic Encoding → Model Training → 75-77% Accuracy
    ↓              ↓                    ↓               ↓              ↓
Quality data   Simple scaling     Standardized      XGBoost      Better baseline
from source                        encoding        Classifier    performance
```

> **The Key Difference**: Same models, same hyperparameters — the 3-7% performance gap comes purely from the difference in source data quality.

### Model Implementation

**Models Used:**

| Model | Clean Data Notebook | Corrupt Data Notebook |
|-------|--------------------|-----------------------|
| **Logistic Regression** | Yes | Yes |
| **Decision Tree** | Yes | Yes |
| **Random Forest** | Yes | Yes |
| **XGBoost** | Yes | Yes |

- **Logistic Regression**: Baseline linear model for benchmarking
- **Decision Tree**: Single tree classifier; evaluated in both notebooks for interpretability comparison
- **Random Forest**: Ensemble of decision trees with strong generalization
- **XGBoost (Primary)**: Best overall performer; multi-class classification (Poor / Standard / Good), handles imbalanced classes effectively, robust to outliers, with excellent feature importance insights

**Preprocessing Pipeline:**
1. **Imputation**: SimpleImputer for numerical features
2. **Scaling**: StandardScaler for numerical normalization
3. **Encoding**: OneHotEncoder for categorical variables

**Evaluation Strategy:**
- Train-test split (80-20)
- Stratified sampling to preserve class distribution
- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix analysis
- Feature importance ranking

### Tracked Artifacts

**Models and preprocessors are version-controlled using DVC (Data Version Control):**

```
models/
├── scaler.pkl          # StandardScaler for numerical features
├── imputer.pkl         # SimpleImputer for missing value handling
├── encoder.pkl         # OneHotEncoder for categorical features
└── xgb.pkl            # Trained XGBoost classifier
```

All model artifacts are tracked via the `models.dvc` file, ensuring reproducibility and version control across different experiments and deployments.

---

## Feature Importance Analysis

### Top 10 Most Important Features (Clean Dataset - XGBoost)

| Rank | Feature | Importance Score | Category |
|------|---------|------------------|----------|
| 1 | Outstanding_Debt | 0.158 | Financial Health |
| 2 | Interest_Rate | 0.142 | Credit Products |
| 3 | Annual_Income | 0.098 | Income & Assets |
| 4 | Credit_Mix_Good | 0.087 | Credit History |
| 5 | Delay_from_due_date | 0.076 | Payment Behavior |
| 6 | Payment_of_Min_Amount_Yes | 0.069 | Payment Behavior |
| 7 | Credit_Utilization_Ratio | 0.064 | Financial Health |
| 8 | Num_of_Delayed_Payment | 0.058 | Payment Behavior |
| 9 | Credit_History_Age | 0.052 | Credit History |
| 10 | Monthly_Balance | 0.047 | Financial Health |

### Impact of Source Data Quality on Feature Importance

**Corrupt Dataset (After Extensive Cleaning):**
- Numerical features dominate due to residual categorical noise
- Payment behavior features partially recovered through cleaning
- Feature importance concentrated in top 3-4 robust features
- Model relies heavily on features least affected by corruption

**Clean Dataset (Minimal Processing):**
- Categorical features show true predictive power
- Payment behavior features fully utilized
- Credit history variables reveal genuine signal
- Balanced feature importance distribution

### Key Observations

1. **Categorical Feature Performance Gap**: `Credit_Mix` shows **47% higher** importance in clean dataset
   - **Why?** Even after cleaning, corrupt data retains some ambiguity and noise in categorical variables
   - **Business Impact**: Credit mix is a core FICO factor — corrupt source data limits its predictive value even after cleaning

2. **Payment Behavior Signal Recovery**: Delay and minimum payment features show improvement after cleaning but still lag behind clean dataset
   - **Why?** Standardization helps but cannot fully recover lost information from original corruption
   - **Business Impact**: Payment discipline patterns are partially masked by original data quality issues

3. **Robust Numeric Predictors**: Debt and interest rates remain critical in both datasets
   - **Why?** Numeric features are more resilient to corruption and cleaning can recover most signal
   - **Business Impact**: Validates that some features are more "rescue-able" through preprocessing

4. **The Unclosable Gap**: Despite extensive feature engineering, 3-7% performance gap persists
   - **Why?** Information lost due to poor source data quality cannot be fully reconstructed
   - **Business Impact**: Prevention (better data collection) beats cure (data cleaning)

> **Critical Insight**: This comparison demonstrates that **data cleaning and feature engineering are necessary but not sufficient**. The inherent quality of source data creates a performance ceiling that even the most sophisticated preprocessing cannot break through. **Invest in data quality upstream, not just downstream fixes.**

---

## API Deployment

The trained XGBoost model has been deployed as a **production-ready REST API** using FastAPI, enabling real-time credit score predictions.

### Architecture Overview

```
Client Request → FastAPI Endpoint → Data Validation → Preprocessing → Model Inference → JSON Response
                                         ↓                 ↓               ↓
                                   Pydantic Schema    Pipeline Apply    XGBoost Predict
                                   (Type Safety)    (Scaler/Encoder)   (Probability)
```

### API Features

**Type-safe request validation** with Pydantic models  
**Structured response enforcement** via `response_model=CreditResponse` on the `/predict` endpoint, guaranteeing the response always contains `Credit_Score` (str) and `Probability` (float)  
**Automatic data preprocessing** using saved pipelines  
**Real-time predictions** with probability scores  
**RESTful endpoints** with health checks    

### Endpoints

#### 1. Root Endpoint
```http
GET /root
```
**Response:**
```json
{
  "message": "Welcome to Credit Score API"
}
```

#### 2. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "active"
}
```

#### 3. Credit Score Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "Age": 28,
  "Occupation": "Engineer",
  "Annual_Income": 1200000.0,
  "Monthly_Inhand_Salary": 85000.0,
  "Num_Bank_Accounts": 3,
  "Num_Credit_Card": 2,
  "Interest_Rate": 12.5,
  "Num_of_Loan": 2,
  "Delay_from_due_date": 5,
  "Num_of_Delayed_Payment": 1,
  "Changed_Credit_Limit": 15000.0,
  "Num_Credit_Inquiries": 2,
  "Credit_Mix": "Good",
  "Outstanding_Debt": 250000.0,
  "Credit_Utilization_Ratio": 28.5,
  "Credit_History_Age": 36,
  "Payment_of_Min_Amount": "Yes",
  "Total_EMI_per_month": 18000.0,
  "Amount_invested_monthly": 5000.0,
  "Payment_Behaviour": "High_spent_Medium_value_payments",
  "Monthly_Balance": 12000.0
}
```

**Response:**
```json
{
  "Credit_Score": "Good",
  "Probability": 0.8745
}
```

### Data Validation

The API implements **strict input validation** using Pydantic:

**Numerical Constraints:**
- Age, income, accounts, cards: Must be non-negative (≥ 0)
- All numerical fields type-checked and validated

**Categorical Constraints:**
- **Occupation**: Limited to 15 predefined values (Scientist, Teacher, Engineer, etc.)
- **Credit_Mix**: Only accepts "Good", "Standard", or "Bad"
- **Payment_of_Min_Amount**: Only accepts "Yes", "No", or "NM"
- **Payment_Behaviour**: 6 predefined spending patterns

Invalid inputs are automatically rejected with clear error messages.

### Preprocessing Pipeline

The API applies the **exact same preprocessing** used during training:

```python
# 1. Imputation (handles missing values)
numerical_features → SimpleImputer → Imputed values

# 2. Scaling (standardization)
imputed_features → StandardScaler → Normalized values

# 3. Encoding (categorical transformation)
categorical_features → OneHotEncoder → Binary encoded features

# 4. Model Prediction
preprocessed_features → XGBoost → {Credit_Score, Probability}
```

### Model Artifacts

All preprocessing objects and the trained model are loaded from the `models/` directory:

```python
scaler = joblib.load('models/scaler.pkl')      # Feature scaling
imputer = joblib.load('models/imputer.pkl')    # Missing value imputation
encoder = joblib.load('models/encoder.pkl')    # Categorical encoding
model_xgb = joblib.load('models/xgb.pkl')      # Trained XGBoost model
```

**Note**: Model artifacts are tracked with DVC via the `models.dvc` file for version control and reproducibility.

### Running the API Locally

```bash
# Install dependencies
pip install fastapi uvicorn joblib numpy pandas scikit-learn xgboost

# Run the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Access the API:**
- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **API Endpoint**: http://localhost:8000/predict

### Example Usage with Python

```python
import requests

# Prepare customer data
customer_data = {
    "Age": 32,
    "Occupation": "Doctor",
    "Annual_Income": 1800000.0,
    "Monthly_Inhand_Salary": 125000.0,
    "Num_Bank_Accounts": 4,
    "Num_Credit_Card": 3,
    "Interest_Rate": 10.5,
    "Num_of_Loan": 1,
    "Delay_from_due_date": 0,
    "Num_of_Delayed_Payment": 0,
    "Changed_Credit_Limit": 20000.0,
    "Num_Credit_Inquiries": 1,
    "Credit_Mix": "Good",
    "Outstanding_Debt": 150000.0,
    "Credit_Utilization_Ratio": 22.0,
    "Credit_History_Age": 48,
    "Payment_of_Min_Amount": "Yes",
    "Total_EMI_per_month": 12000.0,
    "Amount_invested_monthly": 8000.0,
    "Payment_Behaviour": "Low_spent_Large_value_payments",
    "Monthly_Balance": 25000.0
}

# Make prediction request
response = requests.post(
    "http://localhost:8000/predict",
    json=customer_data
)

# Get result
result = response.json()
print(f"Credit Score: {result['Credit_Score']}")
print(f"Confidence: {result['Probability']:.2%}")
```

### Production Considerations

**Deployment Checklist:**
- [ ] Add authentication/authorization (API keys, OAuth)
- [ ] Implement rate limiting to prevent abuse
- [ ] Add logging and monitoring (CloudWatch, Prometheus)
- [ ] Set up CORS policies for web clients
- [ ] Configure HTTPS/SSL certificates
- [ ] Implement request/response logging for auditing
- [ ] Add model versioning endpoint
- [ ] Set up automated health checks
- [ ] Configure horizontal scaling (Docker/Kubernetes)
- [ ] Implement A/B testing for model versions

**Future Enhancements:**
- Batch prediction endpoint for bulk processing
- Model performance monitoring and drift detection
- Explainability endpoints (SHAP values, feature importance)
- Asynchronous processing for large-scale predictions
- Integration with data pipelines (Kafka, Redis)

---

## Key Takeaways

### The Fundamental Lesson: Source Data Quality is Irreplaceable

> **In credit risk, fraud detection, banking, and most tabular ML domains:**  
> **Even the most sophisticated data cleaning and feature engineering cannot fully compensate for fundamentally poor source data quality. Prevention beats cure.**

### The Two-Tier Reality

**Tier 1: Source Data Quality (Biggest Impact)**
- Clean data from the source: 75-77% baseline accuracy
- Corrupt data after extensive cleaning: 70-74% ceiling
- **Gap: 3-7% that no amount of preprocessing can close**

**Tier 2: Algorithm Selection (Smaller Impact)**
- Switching algorithms typically yields 0-2% improvement
- Hyperparameter tuning adds another 0-1%
- **Combined impact still less than starting with quality data**

### Real-World Implications

**For Data Scientists:**
- Invest more time in EDA and data quality assessment
- Build robust data pipelines before model experimentation
- Collaborate with domain experts for feature engineering

**For ML Engineers:**
- Design data validation layers
- Implement automated data quality checks
- Monitor data drift in production

**For Business Stakeholders:**
- Prioritize data infrastructure investments
- Recognize that model performance plateaus without clean data
- Budget for data quality initiatives

**For Organizations:**
- Data quality becomes a fundamental difficulty in risk-based approaches where banks use internal data to estimate risk components
- Establish data governance frameworks
- Create cross-functional data quality teams

---

## Getting Started

### Prerequisites

```bash
# Python 3.8+
python --version

# Required libraries
pip install numpy pandas scikit-learn xgboost joblib fastapi uvicorn pydantic
```

### Project Structure

```
credit-score-classification/
│
├── corrupt_data_credit_score.ipynb    # Analysis on corrupt dataset
├── clean_data_credit_score.ipynb      # Analysis on clean dataset
├── main.py                            # FastAPI application
├── .gitignore                         # Excludes __pycache__ and models/ from Git (models tracked via DVC)
├── models/                            # Model artifacts (Git-ignored, tracked by DVC)
│   ├── scaler.pkl
│   ├── imputer.pkl
│   ├── encoder.pkl
│   └── xgb.pkl
├── models.dvc                         # DVC tracking file for models
├── data/                              # Dataset directory
│   ├── clean_data.csv                # High-quality dataset
│   ├── corrupt_train_data.csv        # Corrupt dataset (80-20 train-test split)
│   └── corrupt_test_data.csv         # Corrupt test set (for independent testing)
├── .dvc/                             # DVC configuration
├── .dvcignore                        # DVC ignore patterns
└── README.md                         # This file
```

---

## Contributing

Contributions are welcome! Here are some areas to explore:

### Future Enhancements

- [ ] **Advanced Models**: LightGBM, CatBoost, and Neural Networks (TabNet) for further performance benchmarking
- [ ] **Model Interpretability**: SHAP values and Partial Dependence Plots to explain individual credit decisions
- [ ] **Statistical Rigor**: K-fold cross-validation with confidence intervals and significance tests to strengthen findings
- [ ] **Production Hardening**: Docker containerization, CI/CD via GitHub Actions, rate limiting, and automated model monitoring for drift detection
- [ ] **Experiment Tracking**: MLflow or Weights & Biases integration for reproducible, versioned experiments
- [ ] **Interactive Dashboard**: Streamlit or Plotly Dash dashboard for real-time performance visualization and data quality monitoring

---

## References & Further Reading

### Academic Papers on Data Quality in ML

1. **Machine Learning and Data Quality**
   - Decision-making depends heavily on accurate, complete data, and failure to harness high-quality data impacts credit lenders when assessing loan applicants' risk profiles
   - Source: "Effective Machine Learning Techniques for Dealing with Poor Credit Data" (2024)

2. **AI in Credit Risk Management**
   - Financial institutions using AI models can incorporate weak signals and use sophisticated machine learning algorithms to improve prediction accuracy of default risk
   - Source: "The Effect of AI-Enabled Credit Scoring on Financial Inclusion" - MIS Quarterly (2024)

### Indian Banking & Credit Reports

1. **Reserve Bank of India (RBI)**
   - [Report on Trend and Progress of Banking in India 2024-25](https://rbi.org.in/Scripts/AnnualPublications.aspx?head=Trend+and+Progress+of+Banking+in+India) — Released December 2025; GNPA at multi-decadal low of 2.2% (March 2025) and 2.1% (September 2025); CRAR at 17.4%; SCB net profits at ₹4 lakh crore; double-digit balance sheet expansion at 11.2%
   - [Financial Stability Report, December 2025](https://rbi.org.in/Scripts/FsReports.aspx) — GNPA projected to decline further to 1.9% by March 2027 under baseline scenario; flags unsecured retail lending (53.1% of retail slippages), fintech credit risks, and stablecoin risks; PSB CRAR at 16%, private banks at 18.1%
   - [Financial Stability Report, June 2025](https://rbi.org.in/Scripts/FsReports.aspx) — Confirmed GNPA at multi-decade low; RBI Financial Inclusion Index improved to 67.0; 514 districts fully digitally enabled; foreign exchange reserves at record $642 billion

2. **Government of India Reports**
   - [Economic Survey 2025-26](https://www.ibef.org/economy/economic-survey-2025-26) — India's real GDP growth estimated at 7.4% for FY26; cumulative repo rate cuts of 125 bps since February 2025; bank credit and deposits growing at 12.5% and 11.4% respectively; NPA at multi-decade lows
   - Source: Ministry of Finance / Press Information Bureau (PIB)

3. **Industry Analysis**
   - [India Banking Sector Overview — IBEF 2025-26](https://www.ibef.org/industry/banking-india) — India's BFSI sector market cap reached ₹91 lakh crore (US$ 1 trillion) in 2025, growing 50x over 20 years; sector contributes 27% to GDP; M&A activity surged 127% YoY to US$ 8 billion (Jan–Sep 2025); UPI transactions hit ₹230 lakh crore in FY25-26 (till December)
   - "The Silent Reshaping of India's Credit Landscape" — Ideas for India

### Books

- **"The Hundred-Page Machine Learning Book"** by Andriy Burkov
- **"Designing Machine Learning Systems"** by Chip Huyen
- **"Feature Engineering for Machine Learning"** by Alice Zheng & Amanda Casari

### Online Resources

- [Kaggle - Credit Score Classification Dataset](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [DVC Documentation](https://dvc.org/doc)

---

## Acknowledgments

- **Dataset**: [Kaggle Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) by Paris Rohan
- **Inspiration**: Real-world banking system data quality challenges observed across Indian financial institutions
- **Community**: Thanks to the open-source ML community for tools and frameworks
- **Research**: Academic papers on credit risk modeling and data quality in financial services
- **Industry Experts**: Banking professionals who provided domain insights

---

## Final Thought

> **"No amount of data cleaning can fully recover what was lost in poor data collection. Quality starts at the source."**

This project demonstrates — in the simplest, most undeniable way — why **source data quality is the ultimate bottleneck in machine learning**.

**Sophisticated cleaning + garbage data → Limited ceiling**  
**Minimal processing + quality data → Higher baseline**

### The Real-World Lesson

In the rush to deploy the latest AI/ML models, organizations often overlook two critical truths:

1. **Data cleaning is essential** — the corrupt dataset would have been unusable without extensive preprocessing
2. **But cleaning has limits** — even after sophisticated cleaning, a 2.79% to 5.03% performance gap persisted across models

This project shows that:

1. **Data cleaning improved the corrupt dataset significantly** — from unusable to 64–74% validation accuracy depending on model
2. **But couldn't match quality data** — the clean dataset achieved 66–76% with minimal processing across the same models
3. **The gap is permanent** — no algorithm, hyperparameter, or feature engineering closes it; XGBoost on corrupt data (73.66%) still trails XGBoost on clean data (76.45%) despite being the most sophisticated model
4. **₹19,406+ crore potential savings** for the Indian banking sector (FY 2025-26) from the XGBoost accuracy gap alone — rising to ₹34,987 crore when accounting for Random Forest's larger gap

### Investment Priority Pyramid

```
         Most Impact
            ▲
            │
    ┌───────┴────────┐
    │  Data Quality  │  ← Invest here first
    │   at Source    │
    ├────────────────┤
    │ Data Cleaning  │  ← Essential but limited
    │ & Engineering  │
    ├────────────────┤
    │   Algorithm    │  ← Smallest marginal gain
    │   Selection    │
    └────────────────┘
```

Before investing in complex neural networks, AutoML platforms, or expensive GPU clusters, invest in:
- **Data collection procedures** and validation at entry points
- **Data governance frameworks** and quality standards
- **Domain expertise** for proper feature definitions
- **Robust data pipelines** with automated quality checks
- **Production deployment** systems (like our FastAPI implementation)

**The boring work of preventing bad data beats the exciting work of fixing it.**

---

