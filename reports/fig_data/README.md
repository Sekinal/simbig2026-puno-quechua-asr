# Figure data

Every number was copied from the evaluation outputs on the GPU servers: `scripts/compare_runs.py` and per-set corpus WER from `artifacts/preds/*.parquet`. Each one is also logged in `reports/RESULTS.md`. WER is in %. Intervals are 95% paired bootstraps: read speech resamples speakers × sentences (pigeonhole), OOD resamples clips.

Sets:
- dev_unseen / itest_unseen: held-out sentences, read speech.
- ood: the 109-clip radio drama.

Models:
- 7B "decoder" = LoRA v2.
- Whisper: clips ≤ 30 s only.

Other data:
- Probes and CKA: `reports/probe_acc{,_300m,_1b}.csv` and `reports/probe_cka{,_300m,_1b}.csv`. 7B "ed" = enc+dec adapter; 300M/1B "ed" = encoder-only adapter.
- Mechanism (coverage vs ranking): `reports/mechanism_*.csv`.
- LoRA geometry: `reports/lora_geometry.csv`.
- Verifier curves: `reports/verifier_curve*.csv` and `reports/verifier_ours_ci*.csv`.
