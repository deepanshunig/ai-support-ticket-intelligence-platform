# Frozen evaluation protocol

## Data roles

| Source | Records | Permitted use |
|---|---:|---|
| Fit | 7,000 | TF-IDF, model weights, transformer head/scaler |
| Development | 1,001 | Candidate/parameter choice and runtime tradeoffs |
| Calibration | 1,001 | One global temperature parameter |
| Policy | 1,001 | Assignment/review thresholds and supported-request coverage |
| Authored challenge development | 48 | Review policy diagnostics and threshold selection |
| Official test | 3,080 | Final reporting after model and policy freeze |
| Authored challenge held out | 52 | Final review recall by challenge group |

Normalized duplicate groups stay together across the four training partitions.
The publisher's train/test overlap of seven groups is disclosed. Demo activity and
reviewer corrections are never merged into benchmark scores or automatic retraining.

## Model choice

The development-selected deployment candidate is Logistic Regression C=10. It
outperformed Multinomial Naive Bayes and frozen DistilBERT features. Encoder weights
were not fine-tuned in the transformer comparison. Retain the exact model artifact,
class order and explicit team mapping used for calibration and policy selection.

## Calibration and routing

Fit one positive temperature on calibration negative log likelihood. Scaling
log-probabilities preserves the top intent and changes confidence; it does not
turn an in-domain classifier into an unknown-request detector. Assess scoring
quality and reliability on the separate policy partition.

Search the documented confidence/margin grid on policy data. Require observed
accepted intent accuracy and mapped-team accuracy of at least 95%, at least 50
accepted supported examples, and at least 75% review recall in each authored
development challenge group. Among eligible choices, maximize supported coverage.
These are prototype design targets, not bank operating requirements or guarantees.
If no candidate meets them, keep all requests in manual review and report why.

Always review a request with no known vocabulary features. A fixed clause guard
also reviews text with strong category signals for different demo teams. This is
an English-language heuristic; it can miss multi-issue requests or review a single
issue unnecessarily. Its constants are fixed before held-out evaluation.

Freeze the selected temperature, thresholds, guard version and exact model/team
mapping hashes. Final evaluation loads this policy; it cannot optimize its choices.
Any later improvement requires a new validation/final-evaluation protocol rather
than tuning repeatedly on the same reported held-out examples.

## Final evidence

Report full 77-class accuracy, macro-F1, top-three accuracy, per-class support and
errors. Report mapped-team accuracy separately. For automatic assignment, report
coverage, accepted and reviewed counts, accepted intent/team accuracy and Wilson
95% intervals. A narrow, highly accurate accepted subset must not be presented as
accuracy for all incoming tickets.

Report review recall and automatic assignments separately for unrelated,
unsupported-banking, ambiguous and multi-intent challenge groups. The authored
set is intentionally review-only and small: it cannot estimate review precision,
customer-stream prevalence, production workload or overall deployed safety.

NLL and Brier scores assess several aspects of predictive quality; neither alone
proves calibration. Reliability diagrams also have sampling uncertainty.
See [probability calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).
