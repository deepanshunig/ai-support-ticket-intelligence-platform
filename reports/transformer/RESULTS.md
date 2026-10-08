# Frozen transformer comparison

All encoder weights remained unchanged. Only a scaler and linear classification
head were fitted on 7,000 fit examples; C was selected on 1,001 development
examples. The pinned DistilBERT revision is recorded in configs/transformer_source.json.

| Measure | TF-IDF Logistic Regression | Frozen DistilBERT + linear head |
|---|---:|---:|
| Development accuracy | 86.71% | 85.81% |
| Development macro-F1 | 86.44% | 85.45% |
| Top-three accuracy | 95.90% | 96.10% |
| Deployment model assets | 10.96 MB | 266.42 MB |
| Warmed single-request p50 | 1.11 ms | 39.72 ms |
| Warmed single-request p95 | 1.70 ms | 83.58 ms |

Logistic Regression remains selected: it wins overall intent accuracy and macro-F1
while using smaller assets and lower measured latency. The transformer's small
top-three advantage does not change that choice. This is a frozen-feature result;
it does not establish how a fully fine-tuned DistilBERT model would perform.

The encoder used four CPU threads; TF-IDF used one. These are local operational
measurements with different thread budgets, not a hardware-normalized architecture
benchmark. Transformer latency includes tokenization, encoder, pooling and head;
it uses 20 timed requests after three warmups. TF-IDF used 100 after five warmups.
Both exclude HTTP, database work and model loading.

Encoding took 268.37 seconds for fit and 40.69 seconds for development. Head
selection took 10.92 seconds across C=1,4,10; C=1 won. Masked mean pooling excludes
padding and includes special tokens. The 64-token limit truncated 18 fit requests
and one development request. See truncation.json for counts.

Offline fixtures verify padding exclusion, batch invariance, evaluation/no-gradient
mode, frozen weights, fit-only scaling, asset checksums and reload predictions.
Calibration, routing-policy and official-test data were not used in this step.

To reproduce the optional comparison after installing CPU PyTorch and Transformers:

```powershell
.\.venv\Scripts\python.exe -m ticket_intelligence.training.transformer --revision 12040accade4e8a0f71eabdb258fecc2e7e948be
```
