# Problem Set 5: Right 93% of the Time

ECON 3916 · Module 3: The ML Pipeline · Notebook: `Econ_3916_PS5.ipynb`

A vendor's loan-default model reports 93% accuracy. On 8,000 simulated applications with a 7% default rate, this notebook tests whether that number means anything and sets a decline threshold from the cost of each error.

## Recommendation to the head of risk

**Sign, but we set the decline threshold, not the vendor.**

93% accuracy means nothing on its own: approving everyone scores 92.9%. The model does rank risk (ROC AUC 0.85; PR AUC 0.37 against 0.07 for random). Expected loss per applicant is \$850 if we approve everyone, \$787 at the vendor's 0.5 threshold, and \$582 at our threshold.

**Threshold: decline at a predicted default probability of 14% or more.** It catches 95 of 170 defaults. It assumes a missed default costs 60% of principal and a wrongful decline 10% (8% margin plus 2% reputational cost). The probabilities are well calibrated, so the threshold holds.

Use the penalised logistic model: the random forest is not better (McNemar p = 0.14) and is harder to explain to a regulator.

**What would change my answer:** our own loss-given-default data. A lower loss rate raises the threshold.

## Key results (test set, 2,400 applications)

| | Result |
|---|---|
| Null model (approve everyone): accuracy | 92.9% |
| Penalised logistic (LASSO): accuracy / ROC AUC / PR AUC | 93.17% / 0.847 / 0.367 |
| Random forest: accuracy / ROC AUC / PR AUC | 92.87% / 0.799 / 0.276 |
| Forest vs LASSO, exact McNemar test | b = 5, c = 12, p = 0.144 |
| Cost-minimising threshold (chosen on training rows) | 0.142 (closed form 0.143) |
| Expected cost per applicant: approve all / 0.5 / 0.142 | \$850 / \$787 / \$582 |
| Threshold on Platt-calibrated probabilities | 0.148 |

**Decision rule:** decline when the predicted probability of default is at least 1 / (1 + R), where R is the cost of a missed default divided by the cost of a wrongful decline.

## AI-assisted checks (Phase 4, Claude Opus 5.5)

- **Part A:** the AI chose the exact McNemar test, which is correct for paired predictions. With the accuracies hidden, it guessed the forest scored higher; it scored lower, so only the two-sided p is reported.
- **Part B:** the AI's calibration script, run on this notebook's data, found predicted and observed default rates consistent near the threshold, and Platt and isotonic gave no detectable gain. It had reused settings from an earlier draft (threshold 0.184, loan-weighted cost); the call corrects them.

## How to run

Open the notebook in Google Colab and run all cells (Runtime → Run all). The first cell installs scikit-learn 1.9 or later. All randomness is seeded (`RANDOM_STATE = 42`).
