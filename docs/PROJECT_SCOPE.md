# Project scope

AI Support Ticket Intelligence Platform is an English-language banking-support
prototype. A customer submits a request; a classifier predicts its intent; a
versioned policy recommends a demonstration team or requests human review. Agents
correct predictions and update ticket status. The database preserves the original
prediction and each workflow event.

## Intended users and teams

- Customer: submits a request and retrieves its ticket record.
- Reviewer: confirms/corrects intent, assigns a team and updates status.
- Demo teams: Card Operations; Payments and Transfers; Cash and ATM Support;
  Account Support; General Review.

Team routing is a project-defined mapping. No real bank integration, transaction
lookup, refunds or account changes are in the prototype's scope.

## Build order

1. Repository, isolated environment, settings, logging and minimal FastAPI service.
2. Banking77 audit, documented 77 labels, team mapping and reproducible partitions.
3. TF-IDF + Logistic Regression baseline; additional Naive Bayes comparison.
4. MySQL persistence, ticket/review API, Streamlit interface and basic analytics.
5. DistilBERT comparison after the local MVP works.
6. Calibration, review policy and frozen final evaluation.
7. Focused tests, CI and containers.
8. One cloud release, documentation and demonstration.

Each step is completed and verified before beginning the next.

## Data and evaluation boundaries

Banking77 provides request text and intent labels, not urgency, identities, real
ticket histories, resolutions or resolution duration. Seeded records are demo data.
Keep fit, development, calibration and policy partitions independent; preserve the
official test for final evaluation after choices are frozen. Challenge examples
include unsupported, ambiguous and multi-intent requests with documented handling.

Report macro-F1, per-intent errors, team routing accuracy, acceptance coverage,
review workload, latency and memory. Evaluate confidence before automatic routing.
Review feedback is not instant model training or unbiased population accuracy.

## Delivery targets

- Local MVP: phases 1-4, approximately 33-42 focused implementation hours.
- Portfolio release: phases 1-8, 72-90 hours plus a 10-hour contingency.
- Optional: independently evaluated similar-request search and cited suggestions.
  Resolution retrieval needs an appropriate additional corpus.

These are planning estimates, not measured model performance or delivery promises.
