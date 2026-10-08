# Frozen routing evaluation

Evaluated 8 October 2026 after freezing the Logistic Regression model, one global
temperature and one assignment/review policy. The policy version is
`routing-v1-6f0251218d47`. It uses temperature **0.58584**, confidence threshold
**0.70**, top-two margin **0.20** and the fixed `multi-team-clauses-v1` guard.
The machine-readable report contains aggregate and per-class results, the policy
hash and exact test-data hash in [`report.json`](report.json). Row-level error
examples are kept in the local workspace.

## Official test

The untouched publisher test partition contains 3,080 Banking77 requests across
77 intents. Full-set intent accuracy was **87.60%**, macro-F1 **87.62%**, top-three
accuracy **96.23%**, mapped demo-team accuracy **96.10%**, negative log likelihood
**0.4397**, and multiclass Brier sum **0.1783**.

The frozen policy automatically assigned **2,547/3,080 requests (82.69%)** and sent
533 to review. Within the accepted subset, intent accuracy was **95.09%** (Wilson
95% interval **94.18%-95.87%**) and demo-team accuracy was **98.90%** (Wilson 95%
interval **98.42%-99.24%**). Full-set intent accuracy remains 87.60%; selective
accuracy describes only the accepted requests.

The policy met the prespecified 95% accepted-intent point target, but its Wilson
interval extends below 95%. That uncertainty matters when interpreting the result.

## Authored held-out review challenges

All 52 authored examples were intended for human review. The system reviewed 45,
for **86.54% overall review recall**, and automatically assigned 7. Recall by
challenge group was:

| Group | Reviewed / examples | Review recall |
|---|---:|---:|
| Clearly unrelated | 13 / 13 | 100.00% |
| Nearby unsupported banking | 12 / 13 | 92.31% |
| Multi-intent | 11 / 13 | 84.62% |
| Ambiguous supported | 9 / 13 | 69.23% |

Ambiguous-request review recall fell below the prespecified 75% design target.
The development slice was small and did not predict this weakness. The policy
remains frozen; these results were not used to select another threshold. A future
policy revision needs new independent validation data and a new final evaluation.

The challenge set is balanced and authored, not sampled from a customer stream. It
cannot estimate review precision, request prevalence or staffing workload. The
multi-team clause guard is a heuristic and can miss combinations. Team labels are
project-defined demonstrations, not verified bank departments. The publisher data
audit discloses seven normalized text groups overlapping its original train/test
partitions; the official benchmark was left unchanged.

## Policy selection evidence

On the separate 1,001-request policy partition, the frozen policy accepted 830
requests (82.92% coverage), with 95.90% intent accuracy and 98.43% demo-team
accuracy. It met the minimum review-recall selection criterion in each of the
four 12-example development challenge groups. The fixed candidate search had 30
eligible combinations; it selected the one with highest supported-request
coverage. Calibration and selection details are in
[`reports/routing/report.json`](../routing/report.json).
