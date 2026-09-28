# Loan Detector

Exploring what drives loan default and loan approval risk across four lending datasets
that don't share a join key, a schema, or even always the same target — and turning that
into two things: **Part 1** builds a baseline classifier per dataset, and **Part 2** builds
one model that generalizes across products and uses it to flag risk in a dataset that has
no default label of its own.

## Datasets

| Label (used in code) | Source file | Rows | Columns | Target | Positive rate |
|---|---|---|---|---|---|
| Credit Risk | `raw_data/Loan_default.csv` | 255,347 | 18 | `Default` | 11.6% |
| Demographic | `raw_data/Loan_approval.csv` | 50,000 | 20 | `loan_status`* | 55.0% |
| Behavioral | `raw_data/UCI.csv` (UCI "Default of Credit Card Clients") | 30,000 | 25 | `default.payment.next.month` | 22.1% |
| Mortgage | `raw_data/Loan_Default.csv` (newly added — see Part 2) | 148,670 | 34 | `Status` | 24.6% |

\* **Important:** `loan_status` is an *approval decision* (1 = approved), not a default outcome —
a 55% positive rate is far too high to be a default rate, and the "1" group has *better*
credit profiles, not worse. This dataset answers "would this application get approved?", a
different question from "will this loan default?" See Part 2 for how it's actually used.

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
for Logistic Regression only → Logistic Regression and Random Forest trained with
`class_weight='balanced'`, Gradient Boosting trained with equivalent balanced
`sample_weight`s (it has no `class_weight` argument) → evaluated on ROC-AUC, PR-AUC
(matters more here given the minority class is only ~11.6% of the Credit Risk data),
and Brier score (calibration quality — see below).

| Dataset | Model | ROC-AUC | PR-AUC | Brier | Precision (positive class) | Recall (positive class) |
|---|---|---|---|---|---|---|
| Credit Risk | Logistic Regression | 0.760 | 0.334 | *pending* | 0.227 | 0.696 |
| Credit Risk | Random Forest | 0.754 | 0.323 | *pending* | 0.260 | 0.581 |
| Credit Risk | Gradient Boosting | *pending* | *pending* | *pending* | – | – |
| Demographic | Logistic Regression | *pending* | *pending* | *pending* | – | – |
| Demographic | Random Forest | *pending* | *pending* | *pending* | – | – |
| Demographic | Gradient Boosting | *pending* | *pending* | *pending* | – | – |
| Behavioral | Logistic Regression | *pending* | *pending* | *pending* | – | – |
| Behavioral | Random Forest | *pending* | *pending* | *pending* | – | – |
| Behavioral | Gradient Boosting | *pending* | *pending* | *pending* | – | – |

Gradient Boosting and the Brier score column are new — the Credit Risk LogReg/RF numbers
above are from the original run and predate both, hence "pending" next to them too.

### Model Calibration (Reliability Diagrams)

The notebook now also plots a reliability diagram per dataset (predicted probability vs.
actual fraction of positives) for all three models. This checks something ROC-AUC and
PR-AUC don't: whether a predicted probability of 0.7 really means "70% of similar cases
defaulted." It's a useful check here specifically because every model is trained with
balanced class/sample weights, which tends to push predicted probabilities away from the
true base rate — worth confirming rather than assuming, before those probabilities get
used for anything like a risk score or a cutoff.

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

## Part 2: Cross-Dataset Default Risk Scoring

**The problem:** the three Part 1 models were each trained on their own dataset's full,
native feature set — none of that transfers directly to a dataset built for a different
purpose. `Loan_approval.csv` has no default label at all (see the footnote above), so it
can't be evaluated the normal way. Instead, the goal here is to build a model that predicts
*genuine* default risk, trained only on datasets with a real default label, then use it to
score `Loan_approval.csv`'s applicants and see where the current accept/reject split might
be missing risk.

**Which sources go where, and why:**
- `Loan_default.csv` (personal loans, real `Default` label) + `Loan_Default.csv` (mortgages,
  real `Status` label) → **pooled for training.** Both have a genuine default outcome and a
  comparable "loan application" feature vocabulary (income, credit score, rate, DTI, age).
- `UCI.csv` (credit cards) → **left out of the pooled model.** It has a real default label
  too, but its features (repayment history, bill amounts) don't overlap with the other
  three at all — pooling it in would mean imputing away most of what makes it useful.
  It stays as its own standalone model in Part 1.
- `Loan_approval.csv` → **scoring target only**, since it has no default label to train on.

**Harmonization fixes required** (found by inspecting the real mortgage data, not assumed):

| Issue | Fix |
|---|---|
| `age` is binned as text (`"25-34"`) | Converted to numeric bin midpoints |
| `income` is monthly, others are annual | Multiplied by 12 |
| `dtir1` is on a 0–100 scale, others are 0–1 | Divided by 100 |
| Mortgage loan amounts run ~10x personal loan amounts | Used loan-to-income ratio instead of raw dollars |
| `Credit_Score` uses a 500–900 scale, others use ~300–850 | Kept raw + added a `loan_product` flag so the tree model can split per-product rather than assume the scales match |
| 0.85% of mortgage rows report $0 income (÷0 in the ratio above) | Treated as missing, median-imputed |

**Calibration mattered more here than anywhere else in this project.** Both pooled models
are trained with `class_weight='balanced'`, which trains as if defaults were ~50% of the
data. Confirmed on real data: the raw model's *average* predicted probability on held-out
data was 0.35–0.38, against a true default rate of 0.16–0.17 — more than double. Isotonic
calibration (5-fold internal cross-fitting) brought the average prediction back in line with
the true rate almost exactly, while ROC-AUC barely moved (calibration fixes the scale, not
the ranking). Using the *raw* model to flag risk in `Loan_approval.csv` flagged 98% of
approved loans as "high risk" — meaningless. The calibrated model gives a usable answer.

**Validated findings** (real data, checked on a representative subsample twice for
consistency — this development environment has 1 CPU core, so the full ~400K-row fit runs
in the notebook itself rather than here; expect the exact numbers to shift slightly at full
scale, but not the story):

- The pooled model separates mortgage defaults far more easily than personal-loan defaults
  on these 6 shared features (ROC-AUC ~0.97 vs. ~0.72–0.73 on held-out data of each type) —
  a reminder that "generalizable" doesn't mean "equally strong everywhere."
- Scoring `Loan_approval.csv` with the calibrated model and flagging at the pooled base
  default rate: **roughly 8–10% of currently-approved loans look high-risk**, and
  **roughly 65–70% of denied applicants look low-risk**, by this model.

**Read that second number carefully.** It's not proof the denials were wrong. The
harmonized model only sees six general features — real underwriting likely used
information this model doesn't have (verification, fraud checks, policy exceptions). Treat
it as a prioritized list for manual review, not an auto-override.

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

- [x] Add Gradient Boosting as a third model alongside Logistic Regression and Random Forest.
- [x] Add reliability diagrams and Brier score to check probability calibration.
- [x] Harmonize Loan_default + Loan_Default (mortgages) into one pooled, cross-product
      default-risk model.
- [x] Calibrate the pooled model and use it to score `Loan_approval.csv`.
- [ ] Re-run the full pipeline on real, full-scale data to replace every "pending"/validated
      number above with the exact final figure.
- [ ] Pick a threshold more deliberately than "pooled base rate" — a cost-based threshold
      (expected loss from a missed default vs. a foregone good loan) would be more
      defensible for the Part 2 risk-flagging table.
- [ ] For the "denied but flagged low-risk" group, check whether `Loan_approval.csv` has
      any fields hinting at *why* they were denied (fraud flags, verification failures)
      before treating that group as a real missed-opportunity list.
- [ ] Compare all three Part 1 datasets on equal footing and decide whether a single
      unified model or three separate models is the right framing there too.
- [ ] Try XGBoost/LightGBM as a stronger, faster-training alternative to sklearn's
      Gradient Boosting — likely to help most on the single-core-unfriendly pooled fit.
- [ ] Investigate the `Age` feature's nonlinear importance in the Credit Risk Random Forest
      (e.g. partial dependence plot).

## Project Structure

```
├── Loan_Detector.ipynb
├── raw_data/
│   ├── Loan_default.csv
│   ├── Loan_Default.csv    # new: mortgage loans, used in Part 2
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
