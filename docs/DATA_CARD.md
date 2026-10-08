# Banking77 data card

## Source and attribution

- Original repository: https://github.com/PolyAI-LDN/task-specific-datasets
- Pinned revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`.
- Licence: Creative Commons Attribution 4.0 (CC BY 4.0). The downloaded licence
  is retained with raw data. Attribute the source when redistributing the dataset.
- Paper: Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence
  Encoders*: https://aclanthology.org/2020.nlp4convai-1.5/
- Source statistics: 10,003 official training records; 3,080 official test records;
  77 banking intents. Actual audits and partition counts are generated under
  `reports/data/`.

## Files and provenance

`data/raw/banking77/` contains immutable source CSVs, `categories.json`, the original
licence and `provenance.json` with revision, source URLs, UTC download time and
SHA-256 checksums. Downloading never overwrites differing existing raw content.

Reproduce the pinned download:

```powershell
.\.venv\Scripts\python.exe -m ticket_intelligence.data.download --revision 57ec275d8078af65b7731c2a98be812d844a6d6b
```

## Intended task

Input is English customer-request text; the target is one documented banking intent.
The canonical category order and original spellings are preserved, including labels
whose capitalization or punctuation is unusual. Raw text is not rewritten.

The five-team routing configuration is **project-defined**. It is not bank operational
ground truth. `configs/team_mapping.json` stores an explicit mapping for all labels.
Counts: Card Operations 31; Payments and Transfers 21; Cash and ATM Support 10;
Account Support 14; General Review 1.

## Internal split protocol

Use the official training data only. A fixed-seed stratified grouped 10-fold split
allocates one fold each to development, calibration and policy validation; the
remaining seven folds are fit data. Group normalized identical texts together.
Normalization for grouping is lowercase plus collapsed whitespace; it does not
replace the text used by the model. Record actual proportions and class coverage.

The official test remains separate and does not select preprocessing, models,
calibration or thresholds. Cross-source duplicate hashes are disclosed without
changing the official test or tuning on its labels.

## Limitations

- Intent requests are not complete ticket histories or conversations.
- The dataset has no authentic priority, identity, resolution, time-to-resolution,
  SLA or support-team ground truth.
- Exact normalized deduplication does not guarantee semantic/paraphrase separation.
- Small per-intent calibration/policy samples require cautious interpretation.
- Unknown, ambiguous and multi-intent requests need separately authored challenge
  evaluation; confidence alone does not prove reliable unknown-request handling.
- Demo activity and later human corrections remain separate from held-out metrics.
