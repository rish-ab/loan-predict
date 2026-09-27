# Loan Detector

Exploring what drives loan default and loan approval risk across three independent
lending datasets, and building baseline classifiers (Logistic Regression + Random
Forest) for each.

## Datasets

| Label (used in code) | Source file | Rows | Columns | Target | Positive rate |
|---|---|---|---|---|---|
| Credit Risk | `raw_data/Loan_default.csv` | 255,347 | 18 | `Default` | ~11.6% |
| Demographic | `raw_data/Loan_approval.csv` | 50,000 | 20 | `loan_status` | TBD |
| Behavioral | `raw_data/UCI.csv` (UCI "Default of Credit Card Clients") | 30,000 | 25 | `default.payment.next.month` | TBD |

All three are independent binary classification problems — each dataset is modeled
separately rather than merged.

## Data Preparation

- **One-hot encoding** for unordered categoricals: `LoanPurpose`, `product_type`, `loan_intent`.
- **Ordinal encoding** for ordered categoricals: `Education`, `EmploymentType`, `MaritalStatus`, `occupation_status`.
- **Binary/category codes** for yes-no fields: `HasMortgage`, `HasDependents`, `HasCoSigner`.
- **Engineered features** (Credit Risk dataset only):
  - `LTI` — loan amount ÷ income
  - `MonthlyDebtRatio` — (income ÷ 12) × DTI ratio
  - `InterestBurden` — loan amount × interest rate
- ID columns (`LoanID`, `customer_id`, `ID`) are dropped before modeling — they carry no
  signal (confirmed via near-zero correlation with the target) and would otherwise break
  the scaler/model on non-numeric input.

## Exploratory Findings

### 1. Credit Risk (`Loan_default.csv`)

Strongest linear correlations with `Default`:

| Increases default risk | r | Reduces default risk | r |
|---|---|---|---|
| Loan-to-income ratio (`LTI`) | +0.179 | Age | −0.168 |
| Interest burden | +0.150 | Income | −0.099 |
| Interest rate | +0.131 | Months employed | −0.097 |
| Loan amount | +0.087 | Employment type | −0.045 |
| Number of credit lines | +0.028 | Has co-signer | −0.039 |

Notably, `CreditScore` correlates only weakly with `Default` (r = −0.034) in this
dataset — loan structure (how much, at what rate, relative to income) matters more
here than the borrower's credit score.

### 2. Demographic (`Loan_approval.csv`)

Strongest linear correlations with `loan_status`:

| Increases approval likelihood | r | Reduces approval likelihood | r |
|---|---|---|---|
| Credit score | +0.496 | Delinquencies (last 2 yrs) | −0.318 |
| Age | +0.312 | Debt-to-income ratio | −0.317 |
| Credit history (years) | +0.277 | Defaults on file | −0.263 |
| Years employed | +0.219 | Derogatory marks | −0.225 |
| Annual income | +0.158 | Interest rate | −0.185 |

Here credit score is by far the dominant single factor — a sharp contrast with the
Credit Risk dataset above, where it barely mattered. This is a useful cross-dataset
insight: the two "credit score" fields behave very differently depending on how each
dataset was constructed.

### 3. Behavioral (UCI "Default of Credit Card Clients")

Strongest linear correlations with `default.payment.next.month`:

| Increases default risk | r | Reduces default risk | r |
|---|---|---|---|
| Most recent repayment status (`PAY_0`) | +0.325 | Credit limit (`LIMIT_BAL`) | −0.154 |
| `PAY_2` | +0.264 | Payment amount (`PAY_AMT1`) | −0.073 |
| `PAY_3` | +0.235 | Payment amount (`PAY_AMT2`–`6`) | −0.05 to −0.07 |
| `PAY_4` | +0.217 | Sex | −0.040 |
| `PAY_5` | +0.204 | | |

Recent repayment status (`PAY_0`–`PAY_6`) dwarfs everything else, including bill
amounts and demographics — recent behavior predicts future default far better than
account balances do.

## Modeling

**Approach:** 80/20 stratified train/test split → features standardized (`StandardScaler`)
for Logistic Regression only → both models trained with `class_weight='balanced'` to
account for class imbalance → evaluated on ROC-AUC and PR-AUC (PR-AUC matters more here
given the minority class is only ~11.6% of the Credit Risk data).

| Dataset | Model | ROC-AUC | PR-AUC | Precision (positive class) | Recall (positive class) |
|---|---|---|---|---|---|
| Credit Risk | Logistic Regression | 0.760 | 0.334 | 0.227 | 0.696 |
| Credit Risk | Random Forest | 0.754 | 0.323 | 0.260 | 0.581 |
| Demographic | Logistic Regression | *pending* | *pending* | – | – |
| Demographic | Random Forest | *pending* | *pending* | – | – |
| Behavioral | Logistic Regression | *pending* | *pending* | – | – |
| Behavioral | Random Forest | *pending* | *pending* | – | – |

Logistic Regression and Random Forest perform similarly on Credit Risk (ROC-AUC ~0.75–0.76),
but Logistic Regression catches more actual defaulters (69.6% recall vs. 58.1%) at the
cost of more false positives — worth deciding which trade-off matters more before
picking a "winner."

**Random Forest feature importance (Credit Risk):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | Age | 0.219 |
| 2 | Interest rate | 0.117 |
| 3 | Loan-to-income (`LTI`) | 0.116 |
| 4 | Interest burden | 0.109 |
| 5 | Months employed | 0.093 |
| 6 | Income | 0.080 |
| 7 | Loan amount | 0.053 |
| 8 | Monthly debt ratio | 0.043 |
| 9 | Credit score | 0.041 |
| 10 | DTI ratio | 0.031 |

Interesting mismatch: linear correlation ranks `LTI`/interest-related features as the
top risk signals, but the Random Forest ranks `Age` highest — suggesting age has a
nonlinear relationship with default risk that a simple correlation coefficient
understates.

## Key Takeaways So Far

- **Behavioral history beats demographics.** In both datasets where a repayment-history
  signal exists (`PAY_0`–`PAY_6` in the Behavioral dataset; delinquencies/defaults-on-file
  in the Demographic dataset), it dominates every other feature.
- **"Credit score" isn't a single consistent signal across datasets.** It's the strongest
  predictor in the Demographic dataset (r = 0.496) and one of the weakest in the Credit
  Risk dataset (r = −0.034) — a reminder to check what a field actually measures before
  assuming it transfers between datasets.
- **Class imbalance keeps precision low.** Credit Risk defaults are ~11.6% of the data;
  even with `class_weight='balanced'`, precision on the positive class tops out around
  25–26%. ROC-AUC ~0.75 looks fine, but PR-AUC (~0.32–0.33) is a more honest picture of
  how hard this problem is.
- Two of the three datasets still need their model results generated (see Next Steps).

## Next Steps

- [ ] Run the (now-fixed) pipeline on the Demographic and Behavioral datasets to fill in
      the modeling table above.
- [ ] Compare all three datasets on equal footing and decide whether a single unified
      model or three separate models is the right framing.
- [ ] Try a gradient-boosted model (XGBoost/LightGBM) as a stronger baseline than Random Forest.
- [ ] Tune the classification threshold instead of using the default 0.5, given the
      class imbalance.
- [ ] Investigate the `Age` feature's nonlinear importance in the Credit Risk Random Forest
      (e.g. partial dependence plot).

## Project Structure

```
├── Loan_Detector.ipynb
├── raw_data/
│   ├── Loan_default.csv
│   ├── Loan_approval.csv
│   └── UCI.csv
└── README.md
```

## Running the Notebook

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
jupyter notebook Loan_Detector.ipynb
```

Place the three CSVs under `raw_data/` before running.
