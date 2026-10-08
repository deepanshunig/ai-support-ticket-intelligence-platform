# Authored banking-support challenge set

`data/challenge_requests.jsonl` contains 100 manually authored English requests
for the banking-support prototype. These are invented examples, not real bank
logs, customer complaints, measured production incidents or additional Banking77
annotations. They contain no actual customer identities or transaction histories.
Example amounts, circumstances and relative dates are fictional.

The examples were written for this project using the intent names in
`configs/team_mapping.json`; they were not copied from Banking77 utterances or
derived from the official Banking77 test set. No official test examples were
opened to author this challenge set. No model was trained or evaluated during
its creation.

## Composition and partition boundary

| Group | Policy development | Held out | Total |
| --- | ---: | ---: | ---: |
| `clearly_unrelated` | 12 | 13 | 25 |
| `nearby_unsupported_banking` | 12 | 13 | 25 |
| `ambiguous_supported` | 12 | 13 | 25 |
| `multi_intent` | 12 | 13 | 25 |
| **Total** | **48** | **52** | **100** |

Records use `partition: "policy_development"` or `partition: "heldout"`.
Development records may inform the review policy. Keep held-out records out of
classifier fitting, prompt selection, threshold choice and policy tuning; use
them only after the relevant choices are frozen. They are independently worded
scenarios, not generated paraphrases of development records.

Both partitions share one JSONL file. This is a logical boundary, not a technical
access-control guarantee: any future consumer must filter the partition before
policy development. This authored held-out set is separate from, and does not
replace, the untouched official Banking77 test set.

## What the four groups mean

- `clearly_unrelated`: a request outside banking, such as cooking, astronomy or
  language editing. Some examples contain incidental words such as card or time
  that must not be mistaken for banking intent evidence.
- `nearby_unsupported_banking`: a financial or bank-adjacent request absent from
  the 77-class vocabulary, such as mortgage origination, loan restructuring,
  brokerage securities trades or insurance servicing. These examples are not
  additional supported classes. Loan applications are distinct from declined
  transfers; annuity beneficiaries are distinct from transfer beneficiaries;
  brokerage dividend tax certificates are distinct from receiving a transfer.
- `ambiguous_supported`: incomplete banking language compatible with existing
  support intents, but missing the operation, object or status needed to choose
  one reliably. These are requests to clarify, not asserted unsupported services.
- `multi_intent`: two explicitly stated, independent supported requests. Each
  record includes existing Banking77 intent names and spans different
  project-defined demonstration teams. A single-label automatic route would
  omit part of the customer's request.

Review and unsupported are different concepts. Every record has the expected
workflow action `needs_review`, but only the first two groups are authored as
outside the supported single-intent domain. Ambiguous requests need clarification;
multi-intent requests need decomposition or human handling. Neither group should
be counted automatically as unsupported just because its expected action is
review.

## Record schema

Each nonempty line is one JSON object:

| Field | Meaning |
| --- | --- |
| `id` | Stable unique example identifier. |
| `text` | The authored English customer request. |
| `group` | One of the four groups described above. |
| `partition` | `policy_development` or `heldout`. |
| `expected_action` | Always `needs_review` in this challenge set. |
| `expected_intents` | Present for multi-intent records; contains the two explicit supported intents. |
| `rationale` | Human-authored reason why review is appropriate. |

`expected_intents` is omitted for unrelated, unsupported and ambiguous examples.
Omission means that no supported single-intent ground truth is asserted; it does
not invent an unknown label or imply that an ambiguous request has no possible
supported interpretation. Multi-intent labels preserve exact vocabulary spelling,
including `Refund_not_showing_up`.

The expected action is an authored prototype handling rule. It is not an observed
customer outcome or a claim that a model can detect all these cases.

## Appropriate future evaluation

After policy freezing, report the fraction sent to review separately for each
held-out group, alongside counts and any automatic routes. Inspect errors by
reason: unrelated-domain routing, nearby unsupported-service routing, insufficient
clarification, or loss of an explicit second intent. If evaluating detection of
unsupported services, use the first two groups and report them separately from
ambiguity and multi-intent handling.

All 100 records intentionally require review. This file alone cannot measure
review precision, unnecessary-review rate, ordinary supported-request coverage,
deployment prevalence, or 77-class macro-F1. Those require a separate supported
evaluation population with an appropriate sampling design. The small balanced
challenge groups are diagnostic examples, not a representative customer stream.

No model scores, threshold recommendations or training results are included in
this deliverable. The challenge set supplements later evaluation; it must not be
used to claim production safety or banking-domain completeness.
