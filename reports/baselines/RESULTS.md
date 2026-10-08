# Baseline development results

Measured on 8 October 2026. Both models fit only 7,000 examples; parameters were
selected on the 1,001-example development partition. These are selection results,
not final official-test estimates.

| Measure | Logistic Regression, C=10 | Naive Bayes, alpha=0.1 |
|---|---:|---:|
| Intent accuracy | 86.71% | 82.22% |
| Macro-F1 | 86.44% | 80.44% |
| Top-three accuracy | 95.90% | 93.21% |
| Project-defined team accuracy | 95.30% | 92.21% |
| Negative log likelihood, raw probabilities | 0.694 | 0.784 |
| Multiclass Brier score, sum across classes | 0.255 | 0.294 |
| Serialized artifact MB, decimal | 10.96 | 1.19 |
| Warmed batch-one classification p50 | 1.11 ms | 5.73 ms |
| Warmed batch-one classification p95 | 1.70 ms | 10.26 ms |
| Total candidate training time | 40.36 s | 0.79 s |

Logistic Regression is the deployment candidate because it wins development
macro-F1 and accuracy. Naive Bayes remains a useful, much smaller comparison.
Latency covers local `Pipeline.predict`, including TF-IDF, with one computational
thread and 100 requests after five warmups. It excludes HTTP, database work, model
loading, probability ranking and review-policy processing. The measurements are
host-specific and are not a production throughput estimate.

## Error interpretation

Common Logistic Regression errors include card arrival versus delivery estimate,
transfer timing versus balance not updated after transfer, and ATM versus card
payment exchange-rate charges. Naive Bayes also confuses unrecognized card
payments with unrecognized direct debits. These distinctions explain why a
prediction should include alternatives and allow a reviewer to correct it.

The five-team mapping combines many intents, so team accuracy can hide a wrong
intent within the correct team. It is not accuracy for routing to real banks.
Top-three accuracy shows the potential usefulness of suggestions; it does not
measure how accurately a reviewer will select the correct option.

Calibration, routing-policy, challenge and official-test results have not yet
been measured in this step. Raw high probabilities do not establish that unknown
or multi-issue requests are safe to assign automatically.

Detailed aggregate evidence: report.json, operational_metrics.json,
candidates.csv, confusion_pairs.csv and confusion_matrix.png. Row-level error
examples and machine-specific runtime metadata are kept in the local workspace.
