# Credit Score Classification: The Power of Clean Data

> **Demonstrating that data quality almost always beats algorithm sophistication in real-world ML projects**

## The Core Message

**In 95% of practical machine learning projects — especially in finance, credit risk, fraud detection, and banking — having clean, high-quality data is far more important than choosing a slightly better algorithm.**

This repository proves that point **numerically and visually** using the **exact same model** on two versions of the **same dataset**:

1. **corrupt_data_credit_score.ipynb** → raw, messy, real-world corrupted data  
2. **clean_data_credit_score.ipynb** → the same data after realistic, production-grade cleaning

**Same model. Same hyperparameters. Same train-test logic.**  
**Yet the performance jumps by ~6–7% purely because the data is clean.**

→ **This is not cherry-picking — this is what happens in almost every real ML project.**

## Quick Results Comparison

| Metric | Corrupt Data | Clean Data | Absolute Gain | Relative Gain |
|--------|--------------|------------|---------------|---------------|
| **Accuracy** | 70.28% | 74.98% | **+4.7%** | **+6.7%** |
| **Precision** | ~74.98% | ~78.36% | **+3.4%** | **+4.5%** |
| **Recall** | 70.28% | 74.98% | **+4.7%** | **+6.7%** |
| **F1-Score** | ~70.65% | ~75.40% | **+4.75%** | **+6.7%** |

**→ All improvement came from data cleaning alone — no hyperparameter tuning, no new model, no advanced feature engineering.**

## Dataset Overview

**100,000 rows × 28 columns**  
Typical credit bureau + banking features:

- **Demographics:** Age, Occupation, Income
- **Credit Products:** Credit Cards, Loans, Interest Rates
- **Payment Behavior:** Delayed Payments, Min Amount Paid
- **Financial Health:** Outstanding Debt, Credit Utilization
- **History:** Credit History Age, Credit Mix
- **Target:** Credit_Score (Good / Standard / Poor)

### What Makes the "Corrupt" Version Realistic?

Common data problems in banking/fintech:

- Negative/impossible ages (`-500`)
- Missing values in critical columns
- Junk strings & placeholders (`"_"`, `"!@9#%8"`)
- Inconsistent formatting (`"22 Years and 1 Months"`)
- Special characters in numeric fields
- Type mismatches

### What the "Clean" Version Fixes

- Realistic age validation and correction
- Intelligent missing value imputation (domain-aware)
- Standardized date/time formats
- Removal of junk characters and placeholders
- Proper categorical encoding
- Outlier detection and treatment
- Feature type consistency

**Key point:** These are realistic, production-grade cleaning steps — not artificial data manipulation.

## Model Used (Identical in Both)

```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=35,
    min_samples_leaf=12,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

**→ Deliberately simple. No difference in modeling at all.**

## Feature Importance Comparison

| Rank | Corrupt Data | Importance | Clean Data | Importance |
|------|--------------|------------|------------|------------|
| 1 | Outstanding_Debt | 0.153 | Outstanding_Debt | 0.126 |
| 2 | Interest_Rate | 0.100 | Interest_Rate | 0.105 |
| 3 | Credit_Mix_Good | 0.066 | Credit_Mix_Good | 0.097 |
| 4 | Delay_from_due_date | 0.058 | Delay_from_due_date | 0.067 |
| 5 | Payment_of_Min_Amount_No | 0.053 | Credit_Mix_Standard | 0.054 |

**Key observation:** Cleaning makes categorical features like `Credit_Mix` +47% more important (0.066 → 0.097), which aligns perfectly with financial domain knowledge. The model can now properly learn meaningful credit behavior patterns.

## Key Takeaway

> **In credit risk, fraud detection, banking, and most tabular ML domains:**  
> **80%+ of your performance lift will come from data quality, feature engineering, and domain understanding — NOT from switching XGBoost → LightGBM → CatBoost → TabNet.**

### Recommended Workflow Order

```
1. Clean the data (make it trustworthy)
2. Do thoughtful feature engineering (with domain knowledge)
3. Try simple models first (Random Forest, Logistic Regression)
4. THEN — and only then — worry about algorithms & tuning
```

**Not:** `Try 10 different algorithms → hope for the best → wonder why nothing works`

## Contributing

Contributions welcome! Areas to explore:

- [ ] Additional model comparisons (XGBoost, LightGBM, Neural Networks)
- [ ] SHAP/LIME interpretability analysis
- [ ] Cross-validation with statistical tests
- [ ] Production pipeline with CI/CD
- [ ] Data versioning with DVC
- [ ] Interactive results dashboard

## Acknowledgments

- Dataset: [Kaggle Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- Inspired by real-world banking system data quality challenges
- Thanks to the open-source ML community

---

## Final Thought

**The most sophisticated algorithm in the world cannot learn meaningful patterns from fundamentally corrupted data.**

This small project shows — in the simplest, most undeniable way — why **data quality is the biggest lever in ML**.

**Garbage in → garbage out** — even if you have the strongest model in the world.

---

**If this project helped you or your team, please star the repository!**

*Built for better ML practices in 2025 and beyond*