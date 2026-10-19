# RL and self-improvement experiment plan (qxp), 2026-10-10

Companion to `docs/LITERATURE.md` §g.11. That section holds the evidence and the bibliography; bib numbers below refer to it. This file holds the ranked table and the exact recipes.

**Status (updated 2026-10-10, late):**
- **G0, checks 1–2: done.** Oracle headroom is 32–46 % rel. on every clean set (`reports/RESULTS.md`), so RL-2 and RL-3 stay in. Checks 3–4 have not been run. Check 4 (memorisation) is replaced by the label-change rate that `pseudo_label.py --compare` reports inside RL-1.
- **RL-2: done, negative.** OOD +4.4 % rel. Not adopted.
- **RL-1: implemented** (`scripts/runs/17_expert_iteration.sh`) and queued on server A after exp15, ahead of the exp11 RL pilots.
- **All other recipes: proposed, not implemented.**
- "[ADD]" marks a flag or script that does not exist yet.
- All expected gains are **estimates**, not results.
- GPU-h means L40S hours.
- Budget assumed: 2 × L40S for about 6 days. Freeze the candidate systems by **Oct 16**, leaving 2 days for final-mode retraining, the submission and the paper.

## Baselines to beat (WER / CER, `reports/RESULTS.md`)

| System | dev-unseen (dev-mode only) | dev-spont (clean) | OOD radio (clean) |
|---|---|---|---|
| hybrid_dev (CTC v3 + zs + LoRA v1 + v2, dev-mode) | **9.11 / 1.62** | 13.47 / 2.39 | **13.82 / 2.31** |
| hybrid_final (submission `qxp_final_hybrid`) | (seen) | **13.38 / 2.35** | 14.58 / 2.41 |
| LoRA v2 (single, beam) | 13.13 / 2.32 | 14.93 / 2.59 (≤ 39 s: 14.45 / 2.44) | 15.11 / 2.53 |
| LoRA v2 + text-CPT (single) | 14.20 / 2.54 | (≤ 39 s) 14.27 / 2.42 | 14.27 / 2.22 |

**Pending runs that change the starting point:**
- exp15: a hybrid with a 4th scorer, `tc`.
- exp11: the GSPO-gold, GSPO-hybrid-verifier and MWER-gold pilots.

Read their logs before launching RL-3, RL-5 or RL-7.

## Acceptance protocol (all experiments)

1. **Clean sets only.**
   - dev_spont: its speakers are never trained on in any mode.
   - OOD radio: never tuned on.
   - dev-unseen: only for dev-mode models.
   - itest numbers are reported but never used to select.
2. **Accept a change only if all of these hold:**
   - mean clean WER improves by ≥ 3% rel. (the single-seed noise floor, Batra 2026, bib 112);
   - no clean set worsens by more than 0.3 abs WER;
   - a paired bootstrap over clips (1,000 resamples) gives a 95% CI on ΔWER that excludes 0 for the mean;
   - the gain survives **inside the hybrid** (`joint_rescore2.py`), because single-model gains often vanish after rescoring.
3. **Controls for every RL run that "wins":**
   - a shuffled-reward run (Spurious Rewards, bib 215; GRPO clipping alone can produce gains);
   - a second seed, if time allows.
4. **Logged guards in `train_rl.py`.** These already exist: `len_ratio`, `groups_kept`, `best_of_group`. Add **[ADD]** dev insertion rate and dev mean output length/reference length.
   - Abort if the dev length ratio exceeds 1.03, or the insertion rate rises by more than 20% rel. Strong bases tend to over-generate under GSPO.
   - Do **not** add decode-time repetition penalties: Quechua has productive reduplication. Use the multiplicative repetition gate in the *reward* instead.
5. **Final mode.** A winner is retrained in final mode (all labelled data except dev) and enters the hybrid with dev-mode weights. Spontaneous weights are retuned on dev_spont (`--retune-spont`), as in exp12.

## Gate G0 (CPU, about 1 h; run first)

**[ADD]** `scripts/oracle_gap.py`. It reads `artifacts/cands/pool_dev(_tc).parquet` and `artifacts/preds/hybrid_dev(_tc).parquet` and reports four things.

1. **Pooled oracle WER/CER** per clean set vs. hybrid 1-best: the selector headroom.
   - Rule: if oracle WER is within 10% rel. of hybrid on both clean sets, drop RL-2 and RL-3.
2. **Disagreement rate:** the share of clean-dev clips where the hybrid 1-best ≠ the LoRA v2 1-best. This is the signal available to RL-1, RL-4 and RL-7.
3. **Reward-spread check** on 500 training clips (250 scripted, 250 gold spontaneous):
   - Measure the share of beam-8 groups with non-zero spread of −(WER + CER), on clean audio vs. augmented audio (tempo 0.85, codec, 15 dB noise).
   - Rule: RL-3 and RL-5 need at least 30% of groups with signal *on augmented audio*.
4. **Memorisation check** on the unlabelled pool:
   - WER(LoRA v2 1-best, pseudo_v1 label) on 500 pool clips, vs. WER(LoRA v2 1-best, hybrid 1-best) on dev_spont.
   - v1 trained on silver labels and v2/tc/CTC v3 on pseudo_v1 labels of this same audio. If pool agreement is much higher than dev agreement (for example under half the WER), members re-emit memorised labels there. RL-1 and RL-7 must then use the cross-fit variant (RL-1x).

## Ranked plan

Rank = expected value × probability (my estimates). "Δ hybrid" is the relative change in mean clean WER of the final hybrid.

| Rank | ID | Experiment | Test audio? | Δ single (est.) | Δ hybrid (est.) | P | GPU-h | Server / day |
|---|---|---|---|---|---|---|---|---|
| 1 | **RL-1** | Expert iteration: hybrid-selected labels on the unlabelled pool → LoRA v3 (+ CTC v4) | no | spont −4 to −8%, OOD −2 to −5% | −1 to −4% | 0.55 | 20 (+12 CTC, +10 cross-fit) | B, D0–D3 |
| 2 | **RL-2** | Listwise consensus/MBR features + MBR selection in `joint_rescore2` | no | — | −1 to −3% | 0.5 | CPU | CPU, D0 |
| 3 | **RL-3** | MWER-trained LLM scorer over pooled candidates (augmented gold audio) → `lp_mw` feature | no | n/a | −1 to −4% | 0.35 | 10 | A, D1–D2 |
| 4 | **RL-5** | Gold GSPO/MWER with augmented "hard" rollouts + character reward (extends exp11) | no | spont/OOD −1 to −4% | 0 to −2% | 0.3 | 10 | A, D2–D3 |
| 5 | **RL-6** | Acoustic-grounding preference: same text, intact vs. masked audio (anchored conditional DPO) | no | unseen −2 to −5% | −1 to −2% | 0.25 | 8 | B, D3–D4 |
| 6 | **RL-4** | On-policy distillation from the multi-adapter ensemble (token reverse-KL) | no | −2 to −5% | 0 to −2% | 0.3 | 12 + 1 day engineering | A, D3–D4 (only if RL-1 is too weak) |
| 7 | **RL-7** | Consensus-reward GRPO on the unlabelled elderly pool vs. frozen RL-1 labels | no | 0 to −3% beyond RL-1 | 0 to −1% | 0.2 | 8 | A, D4 |
| 8 | **RL-8** | TTRL-style adaptation on the hidden-test audio | **YES (transductive)** | — | 0 to −5% | 0.3 | 4 | A, D5, **labelled contrastive only** |

Total: about 85–105 GPU-h including final-mode retrains of the winners, which fits 2 GPUs × 5 days alongside the pending runs.

**Not planned** (§g.11 evidence):
- offline text-pair DPO / IPO / KTO;
- MWER/RL on the CTC model;
- entropy or self-certainty rewards;
- LambdaMART on the existing score features;
- LLM-judge or semantic rewards;
- rewards from the policy's own majority vote.

---

## RL-1: Expert iteration (hybrid rescorer → its members) on the unlabelled pool  [inductive]

**Hypothesis.** The hybrid CTC + multi-LLM rescorer is our strongest "policy". On dev-spont it is about 7% rel. better than ROVER (13.47 vs. 14.51), and ROVER-style consensus produced the current pseudo_v1 labels. Re-labelling the pool with the hybrid and retraining the members is expert iteration (ExIt, bib 172; ReST/ReST-EM, bib 173–174; MBR/QE fine-tuning, bib 177).
- MBR/QE-FT shows the gain requires a teacher stronger than beam search.
- Distillation recovers about half of a best-of-N gain (bib 182).
- An improved member then improves the hybrid.

**Recipe** (dev mode first; paths as in `scripts/runs/12_hybrid_final.sh`):

```bash
# 0) CTC v3 posteriors of the unlabelled pool (skip if they already exist)
uv run python $R/infer_ctc.py --name pool_ctc1b_v3 --pool \
    --checkpoint $A/merged/ctc1b_v3_soup_wise0.85.pt
# 1) CTC N-best for the pool      [ADD] ctc_nbest.py --pool (mirror infer_ctc.py: pseudo_pool(man))
uv run python $R/ctc_nbest.py --post pool_ctc1b_v3 --name ctc_pool --pool
# 2) LLM N-best + every-adapter scoring (beam 4 halves the cost on 27 h)
uv run python $R/llm_candidates.py --ctc ctc_pool --name pool_unl --beam 4 --adapter zs \
    --adapter v1=$A/runs/llm7b_lora_v1/best_trainable.pt \
    --adapter v2=$A/runs/llm7b_lora_v2/best_trainable.pt \
    --adapter tc=$A/runs/llm7b_textcpt/best_trainable.pt      # drop tc if exp15 shows no gain
# 3) apply the dev-tuned spontaneous weights to the pool
#    [ADD] joint_rescore2.py --apply-pool: score pool clips with saved spont weights; write
#    preds/<name>.parquet with hyp, margin (= score(top1) - score(top2)), n_sources(top1)
uv run python $R/joint_rescore2.py --cands pool_unl --post pool_ctc1b_v3 \
    --weights hybrid_dev_tc.weights.json --apply-pool --name hyb_pool
# 4) filter + age split          [ADD] pseudo_label.py --from-hybrid hyb_pool --min-margin-quantile 0.3
#    --min-sources 2 --split-age; also drop chars/s outside 0.5-1.5x median and any 5-gram repeat
# 5) LoRA v3 from the BASE model (re-init each round: Berrebbi 2023, Singh 2023, §g.5),
#    v2 recipe + text-CPT, so the only change vs. "v2+tc" is the label source
uv run python $R/train_llm.py --run llm7b_lora_v3 --card omniASR_LLM_7B_v2 \
    --encoder frozen --decoder lora --rank 16 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/hyb_pool_old.parquet,pseudo_rest=$P/hyb_pool_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 \
    --eval-every 300 --eval-clips 150 --workers 8 \
    --text-corpus $A/lm/dev_v2/all.txt --text-weight 0.3 --text-batch 16 --grad-ckpt
# 6) add v3 as a 5th adapter (keep v1, v2: diversity), retune, compare with hybrid_dev(_tc)
uv run python $R/llm_candidates.py --ctc ctc_v3 --name pool_dev_v3 --adapter zs --adapter v1=... \
    --adapter v2=... --adapter tc=... --adapter v3=$A/runs/llm7b_lora_v3/best_trainable.pt
uv run python $R/joint_rescore2.py --cands pool_dev_v3 --post ctc1b_v3_soup_wise0.85 --name hybrid_dev_v3
```

**Variants.**
- **RL-1b (CTC v4).** `train_ctc.py` with the v3 recipe (12k steps, soup + WiSE 0.85) on `hyb_pool_*` instead of `pseudo_v1_*`. The CTC log-likelihood is the base score of the hybrid (weight 1), so a better CTC moves every decision. Cost: about 12 GPU-h. Run it only if step 6 is positive.
- **RL-1x (cross-fit), required if G0's memorisation check fires.**
  - Split pool speakers into halves A and B. Train `llm7b_lora_v2A` and `llm7b_lora_v2B` with the v2 recipe, each with pseudo labels of one half only (about 4 GPU-h each).
  - Label half B with the hybrid {zs, CTC v3, v2A}, and half A with {zs, CTC v3, v2B}. The CTC still saw both halves; accept that, or weight it lower.
  - The weights are reused from hybrid_dev; spont weights are re-tuned on dev_spont with the fold models.
- **Round 2** only if round 1 passes acceptance: re-label with a hybrid that includes v3 (and v4), then retrain. Stop after 2 rounds (Chasing COMET, bib 232).

**Success criterion.**
- (a) Single model: LoRA v3 vs. LoRA v2 + text-CPT on dev_spont (≤ 39 s and full) and OOD, with dev-unseen no worse than +0.3.
- (b) System: hybrid_dev_v3 vs. hybrid_dev(_tc) by the acceptance protocol.

**Expected (estimate).** Teacher label quality improves by about 7% rel. (hybrid vs. ROVER on dev-spont). Distillation transfers about half of it, so single-model spont −4 to −8%, OOD −2 to −5%, hybrid −1 to −4%.

**GPU-h.**
- Pool candidates: about 10–14 (27 h of audio, beam 4, 4 adapters + scoring).
- LoRA v3: about 4. Dev candidates: about 3.
- Optional: +12 (CTC v4), +10 (cross-fit).

**Risks.**
- Memorised pool labels (G0).
- The student imitates the ensemble and lowers diversity. Mitigate by adding v3 rather than replacing members.
- Errors shared by all members get confirmed. Mitigate with the margin and source-count gates; the elderly per-speaker curriculum (bib 90) can order round 2.

## RL-2: Listwise consensus features and MBR selection in the selector  [inductive, CPU]

**Hypothesis.** A learned combiner over *score-only* features gives about 1% (Wu 2022, bib 206: LambdaMART 1.92 / 4.11 → 1.90 / 4.07). The gains come from *listwise* information, such as the BERT confidence feature: 1.78 / 3.70, and Alexa 3.37% → 5.16%. Our hybrid already has the scores but uses no listwise information.

**Recipe.** **[ADD]** `joint_rescore2.py --listwise`, which appends three features per candidate:
1. `mbr`: the expected CER of the candidate against the pool, under the posterior p ∝ exp(score / T), where score is the current hybrid score. Recompute it inside `tune()`'s objective with the trial weights; T ∈ {0.5, 1, 2} is chosen on dev. This is MBR with a CER utility (Jinnai 2025, bib 167; Novosad's LM-MBR −9% rel., bib 203).
2. `nsrc`: the number of distinct systems that proposed the candidate, from the `sources` column (ctc, zs, v1, v2, tc).
3. `rate`: the z-score of characters per second against the clip's median candidate. This catches deletion and insertion outliers.

Keep the model linear with random search (≤ 6 weights, as now). Also try **[ADD]** `--decide mbr`, which outputs argmin of expected CER under the tuned posterior instead of argmax score.

Tune jointly on dev-unseen + dev_spont. Report OOD untouched. Optional check only: LightGBM LambdaMART with NDCG on WER ranks, under 5-fold CV over dev clips. Adopt it only if it beats the linear model in CV *and* on OOD.

**Success.** The acceptance protocol vs. hybrid_dev(_tc). Since no training is involved, require the gain on **both** clean sets.

**Expected.** −1 to −3% rel. Bounded by the G0 oracle gap. Cost: about 2 CPU-h.

**Risk.** More weights fit the 300-clip dev_spont. Keep the number of weights small and inspect them for stability across 2 random-search seeds.

## RL-3: Discriminative (MWER) LLM scorer over pooled candidates  [inductive]

**Hypothesis.** The `lp_*` scorers are trained with token CE on references. They were never trained to *discriminate* among the near-miss candidates the hybrid chooses between. MWER-trained LLM scorers beat likelihood scorers by 5.9–8.5% rel. (bib 208; RescoreBERT bib 207; LoRB bib 209). Training on the *pooled* list (CTC, zero-shot and LoRA beams), not on the policy's own beams, makes the candidates diverse even where the policy is confident.

**Recipe.**
- **Data:** gold training clips only: all spont_gold (×3) plus 4 h of scripted. Decode them on **augmented** audio so that errors exist: tempo 0.8–0.95, codec p = 0.5, noise 10–20 dB SNR, via `qxp_asr.augment` (G0 check 3). Candidates come from:
  - `infer_ctc.py` + `ctc_nbest.py` on the augmented copies **[ADD `--augment seed`]**, top 8;
  - `llm_candidates.py` beam 4 for zs and v2.
- **Objective:** MWER over the pooled list.
  - L = Σ_i softmax_i(lp_mw / τ) · (r̄ − r_i) + 0.1 · CE(reference), with r = −(0.5·WER + 1.0·CER) and τ = 1.
  - Optionally add an MWED term (RescoreBERT) against MWER over-confidence.
  - Init from the LoRA v2 adapter. lr 1e-5, 8 clips × ≤ 16 candidates, 600–1,000 steps, eval every 100 (hybrid-free proxy: N-best selection WER on dev).
- **Code:** **[ADD]** `train_rl.py --objective mwer --candidates <cands parquet> --augment`. Candidates are read from the table instead of from `gen.generate_nbest`; the rest of the MWER branch is unchanged. Then score with **[ADD]** `llm_candidates.py --score-only mw=...`, which scores but does not add beams, and add `lp_mw` to `joint_rescore2`.

**Success.** Hybrid with `lp_mw` (added, or replacing `lp_v2`) vs. without, by the acceptance protocol. Pay specific attention to dev-unseen, because a scorer can learn the 2k-sentence prior.

**Expected.** −1 to −4% rel. on the hybrid. GPU-h: candidates on about 7 h of augmented audio ≈ 4, training ≈ 3, dev scoring ≈ 2–3.

**Risks.**
- Sentence-prior overfitting (dev-unseen will show it).
- MWER-LoRA degrades OOD (LoRB). Mitigate by keeping the CE anchor and checking OOD.
- The augmented-audio error distribution may differ from real test errors.

## RL-5: Gold GSPO/MWER with hard rollouts and a character-level reward  [inductive]

**Hypothesis.** g.10's RL pilot (exp11) is expected to find little signal, because training audio is transcribed almost perfectly (Novosad, bib 203). Rolling out on *acoustically perturbed* gold audio creates reward variance, and tempo-down plus codec also simulate the elderly/device test shift. A character-sensitive reward gives credit inside words (Mega-ASR bib 197; CER rewards bib 202).

**Recipe** (`scripts/train_rl.py`; existing flags plus **[ADD]**):
- `--augment tempo=0.8-0.95,codec=0.3,snr=10-20`. Apply it to the rollout audio *and* the policy-gradient forward pass (same waveform). The reference stays the gold text.
- `--reward wer+cer`: r = g_rep · −(WER + CER), where g_rep = 0 → r = −1 if there is a repeated 5-gram or the hypothesis/CTC length ratio is outside [0.7, 1.3]. Normalise each term by its own group standard deviation before summing (MO-GRPO, bib 200).
- `--dynamic-sampling`: resample clips until the batch has `clips` groups with non-zero spread (DAPO, bib 230). Keep groups whose best hypothesis has error > 0 and ≤ `max_best_err` (difficulty filtering, bib 231).
- Settings:
  - objective `gspo` with `--update mask`, `--group 8` (beam), `--clips 8`, `--inner 2`, `--clip-eps 0.2`, `--ce-weight 0.1`, `--lr 1e-5`;
  - `--steps 400`, `--eval-every 50`, `--eval-clips 150`, `--eval-beam 1`;
  - also run the `mwer` objective with the same data, because MWER is cheaper and deterministic.
- Init from LoRA v2. 50% scripted / 50% gold spontaneous (existing balancing).

**Success.** Choose the best checkpoint by mean (dev-unseen, dev_spont, OOD), then apply the acceptance protocol both as a single model and inside the hybrid (add it as an adapter).

**Shuffled-reward control** if it wins (≈ 2 GPU-h).

**Expected.** Single model −1 to −4% on spont/OOD, hybrid 0 to −2%. GPU-h about 4 per run, 2 runs + control ≈ 10.

**Risks.** Over-generation (abort rule), spurious-reward gains, and a mismatch between augmented and real elderly acoustics.

## RL-6: Acoustic-grounding preference (audio-conditional, anchored)  [inductive]

**Hypothesis.** With about 2k scripted sentences the decoder over-trusts its sentence prior. The hidden test reads new sentences, and dev-unseen is where LoRA lags the hybrid most (13.13 vs. 9.11). Text-only rewards reinforce the prior (Xiong 2026, bib 156). Rewarding log π(y | audio) − log π(y | masked audio) directly increases reliance on the audio. Supporting evidence: masked-audio rejects in BPO (bib 191) and conditional DPO with an anchor (bib 192). There is no ASR precedent for this exact loss, which makes it both the risk and the novelty.

**Recipe.**
- **[ADD]** `train_llm.py --cond-pref-weight 0.5 --cond-pref-mask 0.4 --cond-pref-beta 0.1`. For each batch, ã is the waveform with 40% of frames time-masked in 2–4 contiguous spans, so content words disappear.
- Loss: CE(y | a) + λ · [ −log σ(β · ((log π(y|a) − log π_ref(y|a)) − (log π(y|ã) − log π_ref(y|ã)))) + max(0, log π_ref(y|a) − log π(y|a)) ].
  - π_ref is the frozen LoRA v2. Evaluate it by swapping adapters in `no_grad` (same pattern as `llm_candidates.set_adapter`).
  - The anchor term stops the chosen likelihood from falling (Smaug/DPOP, bib 194).
  - The encoder runs twice (a, ã), with no grad.
- Continue from LoRA v2 for 600 steps at lr 5e-5, plus a v2-recipe run from base (1,500 steps) if the continuation helps.

**Success.** Dev-unseen −3% rel. or better, dev_spont and OOD no worse than +0.3, and the gain survives in the hybrid.

**Expected.** Dev-unseen −2 to −5% single, OOD 0 to −3%, hybrid −1 to −2%. GPU-h about 8.

**Risk.** It may hurt read speech, where the LM prior legitimately helps, or encourage hallucination when the audio is unclear. Watch insertions on OOD.

## RL-4: On-policy distillation from the multi-adapter ensemble  [inductive]

**Hypothesis.** The LLM part of the hybrid is a product of experts (zs, v1, v2, tc). Per-token reverse KL to their log-linear mix, on the student's own rollouts, transfers it to one adapter with token-level credit on the erroneous suffix. Evidence: OPD for ASR, bib 185 (7.17 → 5.50); X-OPD, bib 186, where SFT and offline KD degraded a strong base and OPD did not; Thinking Machines, bib 184. The "teacher must have a gap" condition (bib 187) is borderline here, so this ranks below RL-1.

**Recipe.**
- **[ADD]** `train_rl.py --objective opd --teacher zs,v1=...,v2=...,tc=... --teacher-weights hybrid_dev_tc.weights.json`.
  - Teacher log-probs: Σ_k w_k log p_k(·|prefix), renormalised, with w_k from the hybrid `lp_*` weights.
  - Computed per teacher by adapter swap in `no_grad`, on the cached encoder (`cached_audio`). The student LoRA state is saved and restored around the swaps.
  - **[ADD]** A token-level variant of `nbest.sequence_logprobs` that returns per-token log-probs.
- Rollouts: 2 per clip (greedy + one T = 1.0 sample; **[ADD]** a sampling generator). Data: 70% unlabelled pool (elderly ×3, `--unlabeled`) + 30% gold with 0.1 CE anchor. 8 clips/step, lr 1e-5, 800 steps, eval every 100. Student init from LoRA v2 (or v3 after RL-1).

**Success.** The single model approaches the LLM-only ensemble, and the hybrid improves (acceptance protocol).

**Expected.** Single −2 to −5%, hybrid 0 to −2%. GPU-h about 12, plus about 1 day of engineering. Only worth doing if RL-1 shows that hybrid knowledge transfers but the SFT variant is too weak.

**Risk.** Too small a teacher–student gap, and engineering time.

## RL-7: Consensus-reward GRPO on the unlabelled elderly pool  [inductive]

**Hypothesis.** This is the on-policy counterpart of RL-1. On-policy updates toward the same labels should forget read speech less than SFT does (RL's Razor bib 227; Retaining by Doing bib 228: 27.8% vs. 3.4% forgetting). This matters if RL-1 helps spontaneous speech but costs dev-unseen. Label-free RL collapses when the reward comes from the policy's own votes (bib 213, 214, 216, 217), so the reference must be **frozen and independent**: the RL-1 hybrid labels, from RL-1x if the memorisation check fired.

**Recipe** (`train_rl.py`, existing hybrid mode, adapted):
- **[ADD]** `--unlabeled-ref hyb_pool.parquet`. On pool clips, r = −(0.5·WER + 1.0·CER) against the frozen hybrid label, plus 0.3 × the existing per-character CTC `verifier_reward`, both normalised by MO-GRPO.
- Gate (RESTRAIN-style, bib 216): use a pool clip only if its label has n_sources ≥ 2 and a margin above the 30th percentile. Drop the medium-agreement band if G0 shows it is noisy (DDRL, bib 223).
- **[ADD]** `--len-guard 0.7,1.3` → r = −1 outside the band, because consensus rewards favour deletions.
- `--unlabeled-frac 0.5` (gold reward on the other half), G 8, lr 5e-6, ce 0.1, at most 300 steps, eval every 25. Stop at the first eval that is not better than the previous two.
- Shuffled-reward control.

**Success.** Beats RL-1's LoRA v3 on the clean mean, or matches it with a smaller dev-unseen regression.

**Expected.** 0 to −3% beyond RL-1 (single), 0 to −1% hybrid. GPU-h about 8.

**Risk.** Sudden collapse (expect it; early stopping), and confirmation of shared errors.

## RL-8: TTRL-style transductive adaptation  [TRANSDUCTIVE: labelled contrastive run only]

**What.** RL-7's recipe with the pool replaced by the hidden-test audio of each domain. Rewards come from the hybrid consensus on the test clips themselves (TTRL bib 212; AQA-TTRL bib 225; ASR-TRA bib 201). Episodes are per domain and limited to 100 steps.

**Evaluation.** Run the identical procedure transductively on dev_spont and OOD audio to estimate the effect, then apply it to test.

**Status.** Never part of the primary submission (§g.8, MLC-SLM rules). Report it as "contrastive, uses test audio". Expected 0 to −5% (larger under shift; NTT CHiME-7 about 1% on strong combinations). GPU-h about 4.

---

## Schedule (two servers)

| Day | Server A | Server B | CPU |
|---|---|---|---|
| D0 (Oct 10–11) | finish exp15 / exp11; read the RL pilot logs | RL-1 steps 0–2 (pool candidates) | G0, RL-2 |
| D1 | RL-3 candidates on augmented gold | RL-1 steps 3–5 (LoRA v3); RL-1x folds if G0 fires | RL-1 filtering |
| D2 | RL-3 training + scoring; RL-5 if G0-3 passes | RL-1 step 6 (dev hybrid with v3) | acceptance analysis |
| D3 | RL-5 (+ control) | RL-1b CTC v4 if RL-1 passes, else RL-6 | — |
| D4 | RL-7 (if RL-1 passed) or RL-4 | RL-6 | — |
| D5 | final-mode retrain of the winners; RL-8 contrastive | final-mode retrain of the winners | final hybrid weights |
| D6 (Oct 16) | **freeze**; submission files | — | paper tables |

## What goes into the paper, whatever the outcome

1. G0 numbers: the oracle gap of the pooled candidates, and the reward spread on clean vs. augmented training audio. This empirically explains why vanilla RL has little signal for a strong low-resource system.
2. RL-1 as "expert iteration of a hybrid rescorer", with single-model and system-level deltas.
3. Every RL run with its shuffled-reward control and length/insertion curves, including negative results. No published RL or preference result exists for elderly or genuinely low-resource real-speech LLM-ASR (§g.10, §g.11.3), so negative results are informative too.
