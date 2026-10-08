# Banking-support classifier

## Intended use

Suggest one of the 77 Banking77 intents for an English support request and show
the best alternatives to a reviewer. A separate, explicitly versioned policy
decides whether a suggestion is strong enough to assign to a demo support team.
The classifier cannot verify a transaction or identify the bank responsible.

## Baseline experiment

Compare word unigram/bigram TF-IDF with Logistic Regression and Multinomial Naive
Bayes. Vocabulary and classifier weights use only `fit.csv`. Choose regularization
and smoothing from development macro-F1, breaking ties with accuracy. Both models
are retained; the development winner is a candidate for deployment, subject to
later calibration and routing validation.

Keep fit, development, calibration, policy and official test roles distinct.
Do not refit on the combined training set after selecting a policy: that changes
the probability distribution on which calibration and thresholds depend.

## Report contract

Baseline evidence belongs in `reports/baselines`: candidate parameters, accuracy,
macro-F1, per-class precision/recall/F1/support, confusion pairs and example errors.
Also report training time, serialized model size and measured single-request
latency. CPU timings are local measurements, not cloud throughput guarantees.
Official-test metrics are absent until model and routing choices are frozen.

## Confidence and review

Raw model probabilities are scores, not measured routing accuracy. A single global
temperature (0.58584) was fitted only on the calibration partition. The confidence
threshold (0.70) and top-two margin (0.20) were selected on the separate policy
partition and 48 authored policy-development challenges. The frozen policy's final
test is documented in [the final evaluation](../reports/final/RESULTS.md). Report
accepted accuracy together with coverage and accepted counts; a high accuracy
obtained by reviewing nearly everything is not useful automation.

Intent correctness and correctness under the project-defined team mapping are
different measures. Unknown, ambiguous and multi-intent requests have diagnostic
review tests. A high class probability alone cannot establish that a request is
supported or contains only one issue.

## Limits

- Short English requests, not multilingual conversations or complete bank cases.
- Uneven class sizes and small per-class validation samples.
- Publisher train/test text overlap is disclosed in the data audit.
- Authored challenges are small and not representative of production traffic.
- Team assignments are prototype configuration, not a bank's verified taxonomy.
- No priority, SLA, refund eligibility or root-cause ground truth is available.
- Demo corrections are not automatically added to training or held-out evaluation.

References: [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
and [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html).
