# Results tracker

WER / CER (%), micro-averaged, official normalization (NFC, lowercase, punctuation removed).
Dev sets (see `src/qxp_asr/splits.py`): **unseen** = speaker-disjoint scripted dev clips whose
sentences are held out of all training text (the shared-task test reads new sentences);
**spont** = gold spontaneous dev (4 speakers). Model selection uses mean(unseen, spont).

## Our systems (dev)

| System | Decoding | unseen WER | unseen CER | spont WER | spont CER | Notes |
|---|---|---:|---:|---:|---:|---|
| omniASR-CTC-1B-v2 zero-shot | greedy | 30.4* | 4.7* | 35.4 | 9.7 | *old 120-sentence holdout |
| omniASR-LLM-7B-v2 zero-shot | beam (pipeline) | 25.1* | 3.9* | 14.7 | 2.7 | *old 120-sentence holdout |
| CTC-1B FT (ctc1b_v2, step 500) | greedy | 25.6 | 4.1 | 28.2 | 4.7 | still in warm-up |
| CTC-1B FT (ctc1b_v2, step 4500) | greedy | 17.5 | 3.0 | 21.5 | 3.6 | all-sentence scripted dev: 17.6 / 3.1 |
| CTC-1B FT (ctc1b_v2, final 6k) | greedy | 16.16 | 2.73 | 20.65 | 3.52 | itest-unseen 22.78/4.03, itest-spont 9.21/1.19, OOD 20.62/3.22 |
| + WiSE-FT a=0.85 | greedy | 15.98 | 2.74 | 20.49 | 3.48 | OOD 20.47/3.12 |
| + WiSE-FT a=0.7 | greedy | 16.43 | 2.81 | 20.25 | 3.42 | OOD **19.79/3.03** (robustness gain, as in Wortsman et al. 2022) |
| + WiSE-FT a=0.5 | greedy | 17.77 | 2.94 | 20.97 | 3.54 | OOD 20.77/3.00 |
| ctc1b_v2 + word 4-gram (pyctcdecode) | beam | 15.45 | 2.84 | 19.77 | 3.50 | itest-unseen 21.74; itest-spont 9.35; OOD 20.92 |
| ctc1b_v2 + char 6-gram lexicon-free (flashlight) | beam | 16.79 | — | ≥20.87 | — | worse than greedy: order 6 too weak (flashlight KenLM cap) |
| ctc1b_v2 + word-4g beam + char 10-gram N-best rescoring | beam+rescore | 15.36 | 2.84 | 19.20 | 3.41 | itest-unseen **20.61/3.82**, itest-spont 8.97/1.20, OOD 20.85/3.35 |


### Full evaluation, experiment 04 (LLM-7B) and system combination

WER / CER on full sets. OOD = organizer radio drama (never tuned on). Spontaneous numbers in
brackets exclude the 15 dev / 3 itest clips > 39 s (test clips are all <= 25.8 s); fixed 39 s
chunking made the fine-tuned LLM hallucinate on mid-word tails (fixed: pause-based chunking).

| System | dev unseen | itest unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|---|
| omniASR-LLM-7B-v2 zero-shot | 21.25 / 3.39 | 27.91 / 4.83 | 14.74 / 2.72 [14.25 / 2.45] | 7.32 / 1.05 [7.11 / 1.01] | 23.26 / 4.04 |
| LLM-7B + decoder LoRA r16 (llm7b_lora_v1) | **12.59 / 2.12** | 20.17 / **3.54** | 16.18 / 3.38 [14.41 / **2.43**] | 7.06 / 1.09 [**6.64 / 0.98**] | **14.88 / 2.73** |
| CTC-1B (ctc1b_v2) + word beam + char-10g rescoring | 15.36 / 2.84 | 20.61 / 3.82 | 19.20 / 3.41 | 8.97 / 1.20 | 20.85 / 3.35 |
| ROVER(LoRA-7B x1.5, CTC+LM x1, zero-shot 7B x0.7) | **12.32** / 2.17 | **20.00** / 3.84 | **14.51 / 2.63** | **6.90 / 1.02** | **14.43** / 2.91 |

Fair external comparison (organizer XLS-R-300M+CPT cv26-validated, best released checkpoint,
greedy; it was trained on every dev/itest *sentence*): itest unseen 17.83 / 3.21, dev unseen
14.37 / 2.26. On OOD radio the organizers report 27.4 / 4.55 for their best fine-tuned model
(Huaman et al. 2026) vs **14.4–14.9** for ours: on genuinely unseen conditions our system halves
the error of the strongest published fine-tuned baseline.

### Joint CTC + LLM-ASR rescoring (joint_v1) — best system so far

Candidates = CTC word-LM N-best (16) + LoRA-7B hypothesis + zero-shot-7B hypothesis; score =
exact CTC log-likelihood + gamma * char-10g LM + delta * length + mu * [LLM hypothesis]
(`scripts/joint_rescore.py`; Watanabe et al. 2017 hybrid CTC/attention rescoring). Tuned on dev
(scripted: unseen only): scripted gamma=1.0 delta=0.5 mu=10; spont gamma=0.7 delta=2 mu=50.

| Set | WER / CER | vs ROVER | vs organizer best (XLS-R-CPT, saw all sentences) |
|---|---|---|---|
| dev unseen | **10.54 / 1.81** | 12.32 / 2.17 | 14.37 / 2.26 |
| itest unseen | **16.52 / 3.09** | 20.00 / 3.84 | 17.83 / 3.21 |
| itest (all 909) | 13.6 / 2.7 (612: 12.16/2.49) | — | 17.85 / 3.25 |
| dev spont | **14.31 / 2.52** | 14.51 / 2.63 | leaked |
| itest spont | **6.72 / 0.91** | 6.90 / 1.02 | leaked |
| OOD | 14.88 / **2.48** | 14.43 / 2.91 | 27.4 / 4.55 (reported) |

**We now beat the strongest baseline on every held-out metric**, including the organizers' own
test split, without having seen its sentences (they had).

### Self-training (LoRA v2) and multi-candidate joint rescoring

LoRA v2 = LLM-7B decoder LoRA trained on scripted + gold spont x3 + agreement-filtered pseudo labels
(27.5 h; 50+ speakers x3) instead of raw silver labels. Full evaluation (WER / CER):

| System | dev unseen | itest unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|---|
| LoRA v1 (silver labels) | **12.59 / 2.12** | 20.17 / 3.54 | 16.18 / 3.38 | 7.06 / 1.09 | 14.88 / 2.73 |
| LoRA v2 (pseudo labels, elderly-weighted) | 13.13 / 2.32 | 20.17 / 3.68 | **14.93 / 2.59** | **6.62 / 0.97** | 15.11 / **2.53** |
| joint_v2 (CTC + v2 + zero-shot) | 10.36 / 1.82 | 16.61 / 3.02 | **14.10 / 2.48** | 6.48 / 0.85 | 15.33 / 2.48 |
| joint_v12 (CTC + v1 + v2) | 9.91 / 1.77 | 16.00 / 2.94 | 14.41 / 2.48 | 6.42 / 0.84 | 15.33 / 2.48 |
| **joint_v3 (CTC + v1 + v2 + zero-shot)** | **9.91 / 1.76** | **16.00 / 2.97** | 14.20 / 2.48 | **6.42 / 0.85** | 15.11 / **2.46** |

| CTC v3 greedy (12k steps, pseudo labels, soup + WiSE 0.85) | 14.64 / 2.55 | 21.30 / 3.83 | 19.08 / 3.27 | 8.16 / 1.06 | 19.34 / 3.05 |
| joint_v4 (CTC v3 + v1 + v2 + zero-shot) | **9.64 / 1.68** | 16.52 / 3.03 | 14.24 / 2.51 | 6.55 / 0.87 | **14.88 / 2.46** |

CTC v3 improves on CTC v2 on every set (dev-unseen 16.2→14.6, spont 20.7→19.1, OOD 20.6→19.3), so
the final run selected its recipe automatically. joint_v3 vs joint_v4 differ by 1-3 % relative
(within noise; dev favours v4, itest favours v3).

Self-training mainly helps spontaneous speech (the elderly-relevant domain); v1 and v2 are
complementary, so the final run (exp10) trains both recipes and rescoring uses both (+ zero-shot).

### Hybrid CTC + multi-LLM likelihood rescoring (exp12, joint_rescore2)

Candidates: CTC word-LM N-best + beam-8 N-best of zero-shot / LoRA v1 / LoRA v2; features: exact
CTC log-likelihood, char-10g LM, length, and the sequence log-probability of every candidate under
every LLM. Weights tuned on dev-mode systems (scripted: unseen clips), transferred to final systems
(spontaneous re-tuned on dev_spont). Submission: `artifacts/submission/qxp_final_hybrid.tsv`.

| Set | joint_v3 | **hybrid_dev** | joint_final | **hybrid_final** |
|---|---|---|---|---|
| dev unseen | 9.91 / 1.76 | **9.11 / 1.62** | (seen) | (seen) |
| itest unseen | 16.00 / 2.97 | **16.00 / 2.84** | (seen) | (seen) |
| dev spont (clean) | 14.20 / 2.48 | **13.47 / 2.39** | 13.95 / 2.48 | **13.38 / 2.35** |
| itest spont | 6.42 / 0.85 | 6.56 / 0.87 | (seen) | (seen) |
| OOD (clean) | 15.11 / 2.46 | **13.82 / 2.31** | 14.58 / 2.49 | 14.58 / **2.41** |

Recommended submission: `qxp_final_hybrid.tsv` (best or equal on both clean sets).

### Decoder text-CPT (exp13a, server B; dev protocol)

LoRA v2 recipe + text-only LM loss (weight 0.3) on dev-mode LM text (in-domain transcripts + 1.3M
respelled external tokens), interleaved with ASR batches (`qxp_asr/textlm.py`).

| System | dev unseen | itest unseen | dev spont (<=39 s) | itest spont (<=39 s) | OOD |
|---|---|---|---|---|---|
| LoRA v2 | **13.12** / 2.32 | 20.17 / 3.68 | 14.45 / 2.44 | 6.44 / 0.96 | 15.11 / 2.53 |
| LoRA v2 + text-CPT | 14.20 / 2.54 | 20.17 / **3.55** | **14.27 / 2.42** | 6.41 / 0.97 | **14.27 / 2.22** |

Text-CPT gives the largest out-of-domain gain of any single change so far (OOD WER -0.84, CER
-12 % rel.) and is neutral in-domain: a generalization lever and a diverse rescoring candidate.

### RL / self-improvement plan: gate G0 and RL-2 (reports/rl_experiment_plan.md, LITERATURE §g.11)

**G0 — selector headroom** (`scripts/oracle_gap.py --cands pool_dev --pred hybrid_dev`). Oracle
= best candidate per clip in the pooled CTC + LLM N-best (~12–16 candidates per clip):

| Set | hybrid_dev | pool oracle | headroom | CTC-only oracle | LoRA v2-only oracle |
|---|---|---|---|---|---|
| dev unseen | 9.11 | 5.27 | 42 % rel. | 10.62 | 7.95 |
| dev spont | 13.47 | 9.15 | 32 % rel. | 14.39 | 21.56* |
| OOD radio | 13.82 | 8.23 | 40 % rel. | 15.56 | 11.93 |
| itest unseen | 16.00 | 8.70 | 46 % rel. | 15.22 | 14.00 |

\* dev_spont has 15 clips > 39 s (12.6 % of its audio) that carry CTC candidates only; the test set
has none (max 25.8 s). The hybrid disagrees with the LoRA v2 1-best on 18 % (unseen), 45 % (spont)
and 40 % (OOD) of clips: enough signal for expert iteration (RL-1). The large oracle gap means a
better selector *could* help, but:

**RL-2 — listwise consensus / MBR selection** (`joint_rescore2.py --listwise`: number of proposing
systems + minimum-Bayes-risk term with tuned temperature, started from the hybrid's weights):

| Variant | dev unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|
| hybrid_dev | 9.11 / 1.62 | 13.47 / 2.39 | 6.56 / 0.87 | **13.82 / 2.31** |
| + listwise, T ∈ [e^-1, e^2.5] | 9.11 / 1.60 | 13.45 / 2.39 | 6.55 / 0.87 | 13.97 / 2.32 |
| + listwise, T ∈ [1, e^5] | 9.11 / 1.60 | **13.40 / 2.37** | **6.18 / 0.82** | 14.43 / 2.38 |

Negative: consensus helps in-domain (itest spont −5.8 % rel.) but hurts the untouched OOD set
(+4.4 % rel.), i.e. it overfits the 310-clip dev_spont. Not adopted (fails the acceptance rule: no
clean set may worsen by > 0.3). The headroom has to come from better candidates/scorers (RL-1, RL-3).

### 4th LLM scorer: text-CPT adapter in the hybrid (exp15, dev protocol)

| Hybrid (dev-mode models) | dev unseen | itest unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|---|
| 3 scorers (zs, v1, v2) = hybrid_dev | 9.11 / 1.62 | 16.00 / 2.84 | 13.47 / 2.39 | 6.56 / 0.87 | **13.82 / 2.31** |
| 4 scorers (+ tc) = hybrid_dev_tc | 9.11 / 1.61 | **14.43 / 2.56** | **13.38 / 2.35** | 6.18 / 0.83 | 14.43 / 2.42 |
| 3 scorers, spont weights tuned on dev+itest spont | 9.11 / 1.62 | 16.00 / 2.84 | 13.55 / 2.38 | (tuned) | 14.58 / 2.46 |
| 4 scorers, spont weights tuned on dev+itest spont | 9.11 / 1.61 | 14.43 / 2.56 | 13.40 / 2.35 | (tuned) | 14.35 / 2.39 |

Scripted: the text-CPT scorer gives −9.8 % rel. on itest unseen sentences (clean in dev mode: neither
its speakers nor its sentences are trained or tuned on) at equal dev unseen. Spontaneous: small
in-domain gains, OOD +0.5–0.6 WER (about 10 of ~1.7k words). Tuning the spontaneous weights on
twice as many clips does not bring OOD back to 13.82, so that number was probably a favourable draw of
the 3-scorer weights rather than something more data would recover; differences of this size on
109 OOD clips need the paired bootstrap of the acceptance protocol before a decision.

### Reward bench: which reference-free rewards rank candidates correctly? (`scripts/reward_bench.py`)

Pool `pool_dev_tc` (CTC + zs/v1/v2/tc beam candidates, clips ≤ 39 s). `pair_acc` = share of
candidate pairs with different true error that the reward orders correctly (50 = chance); this
is what GRPO/GSPO consumes. `sel` = WER if the reward alone picks the candidate.

| Reward | dev unseen pair_acc / sel | dev spont | itest spont | OOD |
|---|---|---|---|---|
| CTC log-lik (total) | 75.8 / 12.77 | 53.9 / 17.22 | 63.1 / 7.68 | 63.0 / 19.49 |
| **exp11 RL verifier** (per-char CTC + 0.3 char-LM) | 76.4 / 11.43 | **55.2** / 16.66 | 63.3 / 7.52 | 63.6 / 18.81 |
| char-10gram LM | 62.1 / 24.82 | 55.2 / 18.08 | 57.4 / 9.75 | 61.5 / 21.00 |
| LLM zero-shot log-lik | 59.3 / 20.71 | 66.2 / 13.92 | 62.0 / 7.03 | 56.5 / 23.19 |
| LLM text-CPT log-lik | 67.8 / 13.84 | 66.2 / 14.08 | 63.5 / 6.50 | 69.6 / 14.43 |
| consensus (−mean distance to other candidates) | 77.4 / 13.48 | 64.1 / 15.43 | **69.3** / 7.07 | 68.2 / 18.66 |
| number of proposing systems | 63.9 / 11.52 | 58.8 / 13.84 | 59.4 / 6.48 | 59.2 / **13.44** |
| hybrid score, tuned on labelled dev (supervised reference) | 78.6 / 9.11 | 70.8 / 12.71 | 68.9 / 6.05 | 72.7 / 14.43 |
| oracle | 100 / 5.18 | 100 / 8.11 | 100 / 2.91 | 100 / 8.08 |

Findings:
- The CTC-based verifier reward planned for the exp11 GSPO pilot is near chance on dev
  spontaneous speech (55 %) and weak on OOD (64 %); the pilot is predicted to learn little there.
- Consensus is the best reference-free signal (≈ the tuned hybrid on scripted and itest spont),
  and the LLM likelihoods are complementary to it (best on dev spont / OOD).
- No reference-free reward exceeds ~70 % on spontaneous or OOD speech. This is the bottleneck
  for RL on unlabelled audio, and why new independent verifiers (phone recognizer +
  orthographic G2P, morphological analyser) are worth testing.

**Morphological-FST verifier (negative).** The squoia Southern Quechua spell-checking FST
(`scripts/fst_accept.py`) accepts only 77 % of *gold* tokens (Puno forms such as progressive
-shan, loans and names fall outside its Cusco-leaning lexicon). As a reward, its pairwise accuracy
is 50–57 % (≈ chance) and adding it to the CTC score does not help. An off-the-shelf FST for a
neighbouring variety is not a usable verifier; a Puno-adapted lexicon would be needed.

**Synthetic speech round trip (back-translation feasibility).** MMS-TTS Cusco Quechua
(`facebook/mms-tts-quz`) reading the 241 dev unseen-sentence prompts; CTC v3 greedy:
WER 14.92 / CER 2.59 on the synthetic voice vs 14.64 on the human Puno recordings of the same
sentences. The synthetic audio is as intelligible to our model as real speech, so text-only data
can be turned into paired data: 20k external sentences (~28 h) synthesised, exp18 trains the
text-CPT LoRA recipe + synthetic pairs (weight 0.5) on server B.

**Evaluation gap: no labelled elderly speech.** Gold spontaneous data comes from 15 speakers,
almost all in their twenties or thirties (dev_spont: 298/310 clips twenties; 60+: 2 gold clips in
total). Every spontaneous *test* speaker is 60+. The elderly audio we have (~5 h: 50s–70s, ~25
speakers) is silver-labelled or untranscribed. Our spontaneous dev numbers therefore measure young
speakers only; the target population has audio but no labels. That is precisely the setting where
reference-free RL / self-training must work, and a small human-transcribed elderly set (~100
clips) would be the most valuable addition for both selection and research.

### Provided data only (exp19; decision 2026-10-10: no external text/audio)

External text was used in two places: the word 4-gram for CTC N-best and the char 10-gram of the
hybrid (both "all" = in-domain + respelled external). exp19 rebuilds both from in-domain text
(`--lm-text indomain`) and drops the text-CPT scorer; LLM candidates/scores are reused.

| Hybrid | dev unseen | itest unseen | itest scripted (all 909) | dev spont | itest spont | OOD |
|---|---|---|---|---|---|---|
| dev-mode, external LMs (hybrid_dev) | 9.11 | 16.00 | 12.81 | 13.47 | 6.56 | 13.82 |
| dev-mode, in-domain (hybrid_dev_in) | 9.82 | 17.39 | 13.80 | 13.67 | 6.20 | 14.50 |
| final, external LMs (qxp_final_hybrid) | (seen) | (seen) | (seen) | 13.38 | (seen) | 14.58 |
| final, in-domain, spont re-tuned | (seen) | (seen) | (seen) | 13.95 | (seen) | 15.33 |
| **final, in-domain, dev weights (qxp_final_indomain)** | (seen) | (seen) | (seen) | **13.93** | (seen) | **14.43** |

External text was worth ~8 % rel. on unseen read sentences and ~5 % on OOD, nothing on
spontaneous speech. Re-tuning the spontaneous weights on the 310 (young-speaker) dev_spont clips
overfits (OOD +0.9), so the final in-domain system keeps the dev-mode weights. **Recommended
submission: `qxp_final_indomain.tsv`** (provided data only).

### RL with sentence cross-fit (exp20, server B; provided data only)

Policy updates on training speakers reading held-out sentences of half 0 (+ gold spontaneous
clips); evaluation on dev + itest clips of half-1 sentences (unseen speakers *and* sentences),
dev_spont (≤ 39 s, 150 clips) and OOD. All arms start from LoRA v2, 400 steps, 4 clips × group 4,
lr 2e-5, CE anchor 0.1, greedy evaluation.

| Arm | step | unseen (other half) | dev spont | OOD |
|---|---|---|---|---|
| LoRA v2 (init) | 0 | 17.44 | 14.20 | 14.95 |
| GSPO (masked trust region) | 100 | 17.95 | 14.42 | 15.03 |
| | 200 | 18.55 | 14.27 | 15.63 |
| | 300 | 18.63 | 14.31 | 15.86 |
| | 400 | 18.03 | 14.12 | 15.41 |

| SFT control (CE on the same clips) | 0 | 17.35 | 14.20 | 14.95 |
| | 100 | 17.44 | 14.34 | 15.33 |
| | 200 | 18.29 | 14.49 | 16.09 |
| | 300 | 18.12 | 14.71 | 15.94 |
| | 400 | 17.95 | 14.34 | 16.09 |

| MWER (N-best posterior) | 0 | 17.52 | 14.20 | 14.95 |
| | 100 | 17.44 | 14.05 | 14.88 |
| | 200 | 17.78 | 14.45 | 14.73 |
| | 300 | 17.44 | 14.20 | 14.95 |
| | 400 | 17.69 | 14.12 | 15.18 |

(Step-0 unseen differs by up to 0.17 between runs: bf16 batch-composition noise.)

**Change at step 400 vs. own step 0** (unseen other half / dev spont / OOD):
- SFT: +0.60 / +0.14 / +1.14
- GSPO: +0.59 / −0.08 / +0.46
- MWER: +0.17 / −0.08 / +0.23

MWER, whose gradient sits on the tokens where hypotheses differ, is the only arm that stays within noise everywhere. None improves unseen sentences.

**SFT vs GSPO at step 400:** both are +0.6 worse on the other half's unseen sentences. Away from
the training distribution, GSPO degrades less: OOD +0.46 vs +1.14 for SFT, and dev spont −0.08
vs +0.14. This matches "RL's Razor" (on-policy RL stays closer to the base model and forgets
less than SFT; LITERATURE §g.12). Neither objective turns ~2k clips of 200 new sentences into
better recognition of *other* new sentences. LoRA v2 is already at a good point, and continuing
from it at lr 2e-5 drifts.

GSPO: no gain; unseen sentences and OOD degrade by 0.5–1.2 WER at every checkpoint (dev spont
flat). Only ~25 % of groups carried reward signal (gold spontaneous clips and many read clips are
solved by every beam), so the CE anchor on half-0 references is a large share of the update. The
SFT control, MWER and Dr. GRPO arms separate the two effects; exp21 concentrates the signal
(difficulty curriculum, read clips only, groups of 8).

### Expert iteration, learned verifiers, more RL arms (exp17, exp20b, exp21-23; provided data only)

**Expert iteration (exp17).**
- The hybrid re-labelled the unlabelled pool, but only 5.9 % of shared clips changed label (WER old → new 0.6 %). The ensemble members had memorised the pseudo_v1 labels on that audio. 17.4 h kept.
- LoRA v3 vs v2:

| System | dev unseen | itest unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|---|
| LoRA v2 | 13.12 | 20.17 | 14.93 | 6.62 | 15.11 |
| LoRA v3 | 12.68 | 20.00 | 15.03 | 6.95 | 15.03 |

- Hybrid with v3 as a 4th scorer vs hybrid_dev_in: dev unseen 9.64 vs 9.82, itest unseen 17.22 vs 17.39, dev spont 13.67 vs 13.67, OOD 14.65 vs 14.50. Not adopted.

**Dr. GRPO arm (exp20b), step 400 vs its step 0** (unseen other half / dev spont / OOD): +1.62 / +0.22 / +0.68. This is the worst arm.

**Exploration diagnostic (exp21 pre-pass, `rl_difficulty.py`).**
- On 1,966 held-out-sentence clips, LoRA v2's beam-8 contains a transcript better than its top-1 for only 54 clips (2.7 %).
- Mean err (WER+0.5CER): top-1 0.132 vs beam oracle 0.120.
- With beam groups the policy almost never produces a better answer, so there is little for RL to reinforce. The exp21 training itself ran out of memory at group 8.

**Sequence-level learned verifier (exp22 cross-fit folds + exp23 `train_verifier.py`).**
- Fold models: dev unseen 15.9 / 14.4, dev spont 15.4 / 15.3. Candidate lists with signal: 4,985 scripted and 450 spontaneous clips.

| Verifier | pair acc (dev unseen / dev spont / OOD) | selection WER (same sets) |
|---|---|---|
| before training | 70.4 / 67.2 / 70.9 | 13.3 / 14.2 / 15.0 |
| after 1,500 steps | 70.1 / 65.7 / 69.2 | 12.4 / 15.0 / 15.9 |

- As a 4th hybrid scorer it makes things worse (dev unseen 10.09, dev spont 13.90, OOD 15.26). Negative.

**Word-level verifier (`word_features.py` + `word_verifier.py`).**
- Method: confusion network over the pooled candidates; gradient-boosted P(word correct); features are word-agnostic (votes, posteriors, system 1-bests, lexical plausibility); cross-fit dev ↔ itest, OOD fully held out.
- Error detection on hybrid 1-best words: **AUC 0.83 dev+itest, 0.82 OOD** (0.79 when trained on test-like clips only).
- Word-level combination:
  - Trained on all clips, it gains only on *seen* sentences: dev 10.26 → 7.41, itest 12.08 → 8.91.
  - It loses on unseen sentences: dev 9.82 → 10.80, itest 17.39 → 17.74. It learned to prefer memorised prompts.
  - Trained on test-like clips only (unseen sentences + spontaneous), the best switching threshold is "never switch": dev unseen 9.82 → 9.91, itest unseen = , spont = , OOD 14.50 → 14.58.
- **Detection works, correction does not.** When a hybrid word is wrong, the right word is rarely among the competing alternatives, or the alternatives cannot be told apart. Uses that need only detection:
  - masking unreliable words in pseudo-labels;
  - flagging words for human (community) verification.

### Exploration sweep and mixed-rollout RL (short experiments, 2026-10-11)

`scripts/pass_at_k.py`, 150 clips per set (≤ 20 s), k = 8 per source. "beats" = share of clips
where some hypothesis has lower WER+0.5CER than the beam top-1 (clips that carry RL signal);
"oracle" = corpus WER if the best hypothesis of the source were picked.

| Model / set | top-1 | beam: beats / oracle | T=1 samples | perturbed views (aug) | union (beam+T0.7/1/1.3+aug) |
|---|---|---|---|---|---|
| LoRA v2 / held-out sentences | 14.86 | 6.0 % / 13.86 | 14.0 % / 12.19 | 16.7 % / 11.35 | 24.0 % / 8.35 |
| LoRA v2 / dev unseen | 13.26 | 10.0 % / 10.80 | 18.7 % / 8.84 | 16.0 % / 8.51 | 25.3 % / 7.04 |
| zero-shot / held-out sentences | 21.54 | 6.7 % / 19.87 | 20.0 % / 16.53 | 30.0 % / 13.19 | 32.7 % / 11.35 |
| zero-shot / dev spont | 11.25 | 15.3 % / 9.84 | 32.0 % / 8.43 | 30.7 % / 8.48 | 48.0 % / 6.36 |

**Beam search is a poor explorer.**
- Only ~2.8 of 8 beam hypotheses are distinct; the rest are the same text tokenised differently.
- Temperature sampling and greedy decodes of perturbed audio views (speed 0.85–1.1, noise) find better transcripts 2–5× more often.

**Mixed-rollout MWER (exp24).** Groups of 2 beam + 3 samples + 3 perturbed views, scored on the clean audio; held-out half 0, 150 steps.

| step | unseen (other half) | dev spont | OOD |
|---|---|---|---|
| 0 | 17.35 | 14.20 | 14.73 |
| 50 | 17.18 | 14.01 | 15.03 |
| 150 | 17.44 | 14.16 | 15.11 |

- Neutral. Signal-carrying groups rose to 50–100 %, but the gradient vanishes (norm 1.0 → 0.005).
- Reason: MWER redistributes probability in proportion to the clean-audio likelihood, and the better transcripts found under perturbation have almost none.
- More fundamentally, on labelled clips the reward-optimal answer is the reference itself, so RL can at most match SFT on the same clips, which did not help either (exp20).
- RL can only beat supervision on *unlabelled* audio, where the open problem is selecting the better candidate without a reference.

### Perturbed-view candidates in the hybrid (exp25) — adopted

zs and v2 greedily decode 6 perturbed views per clip (speed 0.85–1.1, noise; `llm_candidates
--views`). The new candidates are scored by every adapter on the CLEAN audio. The hybrid gets a
view-agreement feature (`--view-feature`); its weights are searched from hybrid_dev_in, ending
almost unchanged with naug ≈ 0.03–0.17.

| Hybrid (provided data only) | dev unseen | itest unseen | itest scripted (all) | dev spont | itest spont | OOD |
|---|---|---|---|---|---|---|
| dev mode, hybrid_dev_in | 9.82 | 17.39 | 13.80 | 13.67 | 6.20 | 14.50 |
| dev mode, + view candidates (hybrid_dev_aug) | **9.73** | **16.87** | **13.52** | **13.52** | 6.21 | **13.67** |
| final, qxp_final_indomain | (seen) | (seen) | (seen) | 13.93 | (seen) | 14.43 |
| **final, + view candidates (qxp_final_indomain_aug)** | (seen) | (seen) | (seen) | **13.76** | (seen) | **14.12** |

- The pool oracle improves: dev unseen 5.27 → 4.91, dev spont 9.15 → 8.98.
- Perturbing the input surfaces correct transcripts that beam search never proposes, and the clean-audio hybrid scores can recognise them. Voting across views alone could not, because it hurt OOD in the 150-clip test.
- No clean set gets worse. **Recommended submission: `qxp_final_indomain_aug.tsv`** (provided data only).

### Breakthrough probes: significance, test-time scaling, robust verification (2026-10-12)

**Paired bootstrap of the view candidates** (`scripts/paired_bootstrap.py`, dev-mode hybrid_dev_aug vs hybrid_dev_in, 2,000 resamples):

| Set | Clips changed | WER change | 95 % CI |
|---|---|---|---|
| itest unseen | 6 / 297 | −0.52 | [−1.13, −0.09] |
| dev spont | 10 / 310 | −0.15 | [−0.30, −0.04] |
| OOD | 12 / 109 | −0.83 | [−1.86, +0.08] (P(worse) = 0.03) |
| dev unseen | 3 / 278 | −0.09 | [−0.27, 0] |

- Final mode: dev spont −0.17 [−0.34, −0.05]; OOD −0.30 [−0.79, +0.08].
- The gains are real but come from only ~3.5 % of clips.

**Test-time scaling** (`scripts/view_scaling.py`, fixed hybrid_dev_aug weights; `reports/view_scaling_pool_dev_aug.csv`). Hybrid / oracle WER by number of views:
- 0 views: dev unseen 10.09 / 5.45, itest unseen 17.30 / 9.04, OOD 14.65 / 8.46.
- 6 views: dev unseen 9.73 / 4.91, itest unseen 16.87 / 8.61, OOD 13.67 / 7.33.
- Both curves fall with more views and have not saturated at 6. The hybrid–oracle gap is the verification headroom.

**Scaling to 12 views** (`reports/view_scaling_pool_dev_aug12.csv`; views 7–12 = speed 0.8 / 1.15 and more noise levels):

| views | hybrid: dev unseen / dev spont / OOD | oracle: dev unseen / dev spont / OOD |
|---|---|---|
| 0 | 10.09 / 12.79 / 14.65 | 5.45 / 8.19 / 8.46 |
| 6 | 9.73 / 12.61 / 13.67 | 4.91 / 7.66 / 7.33 |
| 12 | 9.91 / 12.69 / 13.82 | 4.91 / 7.45 / 7.02 |

- **Exploration scales, verification does not.** The oracle keeps improving up to 12 views; the hybrid saturates at about 6 views and then slips slightly.
- The submission keeps 6 views.

**Search-then-distill with perturbed views (exp26).**
- The hybrid labels with views change only 6.9 % of the pseudo_v1 labels (WER old → new 0.7 %; 17.9 h kept).

| System | dev unseen | itest unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|---|
| LoRA v2 | 13.1 | 20.2 | 14.9 | 6.6 | 15.1 |
| LoRA v4 | 13.4 | 20.0 | 15.3 | 6.4 | 15.3 |

- Neutral, like v3. The scorers (v1, v2, CTC v3) memorised these clips, so they prefer the old labels even when the search proposes alternatives.
- Cross-fitted labelling with unbiased scorers runs in exp27.

**Acoustic word features** (`word_features.py --acoustic`: change in the exact CTC log-likelihood of the transcript when a competing word fills its slot): word-verifier AUC 0.788 vs 0.790 without, and still no combination gain. The hybrid already selects by whole-sequence CTC likelihood, so the local change adds no new information.

**Bagged hybrid weights** (`joint_rescore2 --bag 8`: weights averaged over 8 bootstrap tunings on dev): 0.1–0.3 worse on every set, not significant (P(worse) 0.67–0.89). Not adopted.

**Second CTC verifier** (`joint_rescore2 --extra-post ctc1b_v2_wise0.85_all`: log-likelihood under the CTC v2 model, trained without pseudo-labels, as an extra feature): the search keeps its weight at 0 in both domains, and outputs are identical. It is redundant with the primary CTC score.

**Robust verification** (`scripts/robust_scores.py`, log-probs averaged over 3 perturbed views, 150 clips per set; `reports/robust_dev.csv`):
- On its own, view averaging improves a single adapter's ranking. LoRA v2 pairwise accuracy: dev unseen 70.4 → 71.5, itest unseen 70.3 → 72.4, dev spont 67.1 → 69.1. Selection WER on dev unseen: 13.42 → 11.95.
- Added to the hybrid it is redundant: λ chosen on dev = 0, and every λ > 0 hurts.
- The verification ceiling remains ≈ 71 % on spontaneous speech and 75–78 % elsewhere (hybrid pairwise accuracy).

### Where is the remaining error? (panel gates, 2026-10-12)

**Consensus errors.** Share of the hybrid's (hybrid_dev_aug) substitution errors whose wrong word is also output by ≥ 3 of 4 other systems (zero-shot, CTC v3, LoRA v1, v2):

| Set | Share of errors | WER points |
|---|---|---|
| dev unseen | 75 % | 7.3 |
| itest unseen | 65 % | 11.0 |
| dev spont | 56 % | 7.6 |
| OOD | 53 % | 7.3 |

**Errors across readings of the same sentence.** 146 held-out sentences have ≥ 2 readings in dev+itest (321 clips, 145 error events):

| Error type | Share |
|---|---|
| Same error in another reading (systematic) | 15 % |
| Word correct in all other readings (reading-specific) | 72 % |
| Different error on the same word elsewhere | 13 % |

Most residual read-speech errors are therefore specific to one speaker's reading, and the systems agree on the deviating word. The most likely cause is speakers deviating from the prompt (misreadings, dropped or changed suffixes); the reference is the prompt text. This label noise also affects the test (prompt references). It cannot be fully separated from speaker-specific acoustics without a native listener.

**Minimal-pair discrimination.** Candidate pairs that differ in exactly one letter of one word, scored pairwise:

| Set | Pairs | CTC | LoRA v1/v2 | zero-shot | hybrid |
|---|---|---|---|---|---|
| dev unseen | 2,510 | 93 % | 94 % | 83 % | 96 % |
| dev spont | 3,191 | 77 % | 80–81 % | 81 % | 85 % |
| OOD | 1,097 | 81 % | 84–86 % | 76 % | 86 % |

The models hear one-letter contrasts well in read speech. The overall 71–78 % ranking ceiling comes from other pair types (insertions/deletions, multi-word differences). Acoustic headroom is mainly in spontaneous speech.

**Rule-edit share** (reference word one q/k, ejective, aspiration or vowel edit from the hypothesis word): 13 % dev unseen, 21 % itest unseen, 14 % dev spont, 20 % OOD. Any one-letter difference: 36–53 %. Targeted rule-edit candidates are below the panel's 40 % gate, so they are not pursued.

### Human-in-the-loop pilot (non-native reviewer, 2026-10-12)

`make_review.py` → `review_tool.html` (local, CTC word timestamps) → `review_score.py`.
- **Queue:** 70 untranscribed pool clips (elderly speakers first) and 30 dev-spontaneous clips with hidden references, shuffled; the verifier flags up to 4 uncertain words per clip.
- **Reviewer:** a native Spanish speaker who does not speak Quechua.
- **Work so far:** 31 clips, 32 flagged words, about 40 min (≈ 75 s per word, including listening).

**Dev (hidden references), 7 words:**
- The reviewer changed 71 % of the flagged words and answered "can't tell" on 14 %.
- On the 4 words whose reference was among the options, the reviewer got 0 right and the model's original word 3. All the reviewer's changes moved away from a correct word, on q/k, ejectives and suffixes (-cha/-chiq, -paq/-ta).
- 2 words had no correct option.

**Conclusion:** verifier-guided review needs speakers of the language; a non-speaker's corrections add errors. The 24 corrected pool transcripts are **not** used for training. The simulated budget (Table 4 of the paper) assumes a reviewer who knows the language.

### Paired bootstrap of every ablation-ladder step (dev mode, 2,000 resamples, 2026-10-12)

WER change (new − baseline) with 95 % CI; bold = interval excludes 0.

| Step | dev unseen | itest unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|---|
| LoRA v1 vs zero-shot | **−8.66** [−10.7, −6.8] | **−7.74** [−9.5, −5.9] | +1.44 [−0.03, +3.98] | −0.26 [−0.88, +0.41] | **−8.38** [−10.9, −6.1] |
| LoRA v2 vs LoRA v1 | +0.54 [−0.36, +1.45] | 0.00 [−1.29, +1.23] | −1.25 [−3.99, +0.33] | −0.44 [−1.09, +0.07] | +0.23 [−0.94, +1.41] |
| hybrid vs LoRA v1 | **−2.77** [−4.07, −1.62] | **−2.78** [−4.39, −1.13] | **−2.51** [−5.38, −0.81] | **−0.86** [−1.54, −0.26] | −0.38 [−1.28, +0.54] |
| hybrid vs LoRA v2 | **−3.30** | **−2.78** | **−1.25** | −0.42 [−0.92, +0.07] | −0.60 [−1.78, +0.47] |
| hybrid vs CTC v3 | **−4.82** | **−3.91** | **−5.41** | **−1.97** | **−4.83** |
| + views vs hybrid | **−0.09** [−0.27, 0] | **−0.52** [−1.13, −0.09] | **−0.15** [−0.30, −0.04] | +0.02 [−0.07, +0.14] | −0.83 [−1.86, +0.08] |
| full system vs LoRA v1 | **−2.86** | **−3.30** [−4.97, −1.64] | **−2.66** | **−0.84** | **−1.21** [−2.32, −0.08] |
| full system vs LoRA v2 | **−3.39** | **−3.30** [−4.72, −1.87] | **−1.41** [−2.24, −0.70] | −0.40 [−0.92, +0.10] | **−1.44** [−2.74, −0.23] |
| full system vs zero-shot | **−11.52** | **−11.04** | **−1.22** | **−1.11** | **−9.59** |

- The full system is significantly better than every single model on every set.
- LoRA adaptation is the dominant gain on read speech and OOD. On the young-speaker dev spont set it is borderline *worse* than zero-shot (+1.44).
- Hybrid rescoring alone is not significant on OOD (−0.38); the views make the combined gain significant there.

**Clustered bootstrap** (`paired_bootstrap.py --cluster`: speakers resampled for spontaneous sets, sentences for unseen read sets; OOD has no speaker labels and stays clip-level). Dev spont has 4 speakers and itest spont 6, so these intervals are coarse.

| Comparison | dev spont (4 spk) | itest spont (6 spk) | dev / itest unseen (241 / 248 sentences) |
|---|---|---|---|
| LoRA v1 vs zero-shot | +1.44 [−3.62, +2.71] | −0.26 [−0.48, +0.15] | **−8.66 / −7.74** |
| hybrid vs LoRA v1 | −2.51 [−3.43, +0.72] | −0.86 [−1.49, +0.20] | **−2.77 / −2.78** |
| + views vs hybrid | −0.15 [−1.45, 0.00] | +0.02 | −0.09 [−0.27, 0] / **−0.52** [−1.17, −0.09] |
| full system vs LoRA v1 | **−2.66** [−3.69, −0.72] | −0.84 [−1.52, +0.23] | **−2.86 / −3.30** |
| final mode: views vs adopted baseline | −0.17 [−0.21, 0.00] | (seen) | (seen) |

Read-speech conclusions hold under sentence resampling. Spontaneous gains are mostly not significant at the speaker level: there are too few speakers to tell.

### Hybrid ablation and weight-search noise (`scripts/hybrid_ablation.py`, `reports/hybrid_ablation.csv`, 2026-10-12)

Features built once on `pool_dev_aug`. Each variant is re-tuned from scratch with 3 seeds (400 random + 300 local trials); the table gives mean WER. The pool oracle is unchanged by feature ablations.

| Variant | dev unseen | itest unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|---|
| full | 9.64 | 16.99 | 13.52 | 6.33 | 14.22 |
| − CTC log-likelihood | 10.77 | 18.67 | 13.48 | 6.35 | 14.43 |
| − char 10-gram LM | 10.30 | 17.94 | 13.86 | 6.45 | 14.22 |
| − length | 9.85 | 16.84 | 13.86 | 6.34 | 13.92 |
| − all LLM likelihoods | 10.36 | 17.04 | 14.62 | 6.55 | 16.57 |
| − zero-shot likelihood | 9.67 | 17.01 | 13.49 | 6.32 | 14.43 |
| − LoRA v1 likelihood | 10.06 | 16.81 | 13.52 | 6.36 | 14.55 |
| − LoRA v2 likelihood | 9.58 | 17.07 | 13.60 | 6.37 | 14.17 |
| − view-count feature | 10.00 | 17.54 | 13.78 | 6.32 | 14.02 |
| − perturbed-view candidates | 9.73 | 17.13 | 13.74 | 6.33 | 15.28 |
| − CTC candidates | 9.64 | 17.16 | 13.69 | 6.37 | 14.15 |
| − LLM beam candidates | 10.00 | 17.39 | 13.73 | 6.42 | 14.05 |
| CTC candidates only | 15.33 | 22.90 | 16.99 | 7.20 | 19.56 |

- **Weight-search noise:** sd over seeds of the full model is 0.03–0.13 WER. Variants reach up to 0.61 (OOD without views).
- **Reported numbers are one draw.** The submitted dev-mode hybrid (searched from hybrid_dev_in's weights) has OOD 13.67; from-scratch searches average 14.22.
- **Most important features:** the LLM likelihoods, especially out of domain (OOD +2.35, dev spont +1.10 without them). The CTC log-likelihood (+1.1 / +1.7) and the char LM (+0.7 / +1.0) matter on read speech.
- **Candidates:** view candidates matter most on OOD (+1.06). LLM beam candidates matter on read speech (+0.4). CTC candidates alone are far worse.
- **Prior subtraction:** the tuned read-speech weight of the zero-shot likelihood is negative (−0.70 in hybrid_dev_aug), but removing it changes little.

### Verifier quality: two verifier families (`scripts/verifier_curve.py`, `reports/verifier_curve.csv`)

- **Blend:** the hybrid score blended toward the oracle (optimistic).
- **Noisy oracle:** z(−error) + σ·N(0,1), mean of 5 draws, so mistakes fall on random pairs.
- At our own pairwise accuracy the noisy oracle reproduces our selection WER on 3 of 4 sets: dev unseen ≈ 9.5–10 vs 9.73, dev spont ≈ 12.4 vs 12.61, OOD ≈ 13.3 vs 13.67. itest unseen is the exception (≈ 13.5 vs 16.87).
- **+5 points of pairwise accuracy** over ours, interpolated:
  - dev spont: 12.61 → ≈ 10.8–11.0 (−13 to −14 %);
  - OOD: 13.67 → ≈ 10.7–11.0 (−20 to −22 %);
  - itest unseen: 16.87 → 9.1–13.7;
  - dev unseen: 9.73 → 4.9–8.1.

### Test sentences vs the prompt pool (`scripts/pool_overlap.py --preds zs_llm7b_all`)

Normalized character distance of zero-shot hypotheses to the closest training-pool sentence (median; share < 0.25):

| Group | median | share < 0.25 |
|---|---|---|
| dev seen | 0.028 | 98.5 % |
| dev unseen | 0.489 | 0.7 % |
| test scripted | 0.495 | 4.3 % |

**Corpus counts** (manifest):
- Gold spontaneous: 1,074 clips, 5.18 h, 20 speakers (dev 4, itest 6, train 10). Another 148 "pending" clips have text.
- Without gold transcription: 7,749 clips, 37.0 h; 5,651 of these have silver labels.

CER for the abstract: submission OOD 14.12 / 2.27; dev-mode hybrid_dev_aug on the 909-clip itest scripted set 13.52 / 2.63.

Ages of the 1,074 gold spontaneous clips (manifest): twenties 896, thirties 173, forties 2, fifties 1, sixties 2. So 1,069 are from speakers in their twenties or thirties.

### Matched control: perturbed views vs temperature samples (exp30, `reports/ctrl_*.csv`, 2026-10-13)

Evaluation clips only (1,298). The same hybrid (`hybrid_ablation.py --only full`) is tuned from scratch, 3 seeds; cells are mean ± sd.
- **base:** pool_dev_in candidates, 13.6 per clip.
- **+views:** + 6 greedy perturbed views of zs and v2, 14.8 per clip.
- **+samples:** + 6 temperature-1 samples of the clean audio from zs and v2, 16.8 per clip.

| Pool | dev unseen | itest unseen | dev spont | itest spont | OOD | oracle (same order) |
|---|---|---|---|---|---|---|
| base | 10.09 ± 0.09 | 17.54 ± 0.35 | 13.76 ± 0.15 | 6.28 ± 0.12 | 14.63 ± 0.31 | 5.45 / 9.04 / 9.45 / 3.02 / 8.46 |
| + views | **9.64** ± 0.09 | **16.99** ± 0.13 | **13.52** ± 0.03 | 6.33 ± 0.05 | **14.22** ± 0.12 | 4.91 / 8.61 / 8.98 / 2.90 / 7.33 |
| + samples | 9.79 ± 0.10 | 17.25 ± 0.61 | 13.68 ± 0.08 | 6.36 ± 0.19 | 14.73 ± 0.08 | 4.91 / 8.70 / 8.73 / 2.60 / 7.25 |

- Samples add more candidates and reach the same or a better oracle, but the hybrid gains little from them (OOD 14.73, no better than the base pool's 14.63).
- Views gain on every set except itest spont. Their advantage is verifiability, not candidate count.

**Re-decode with pause-based chunking (exp30).** The stored zs/v1 predictions predated fix 4631bb8. Only the spontaneous sets change:
- zero-shot: dev spont 14.74 → 14.31; itest spont 7.32 → 7.25.
- LoRA v1: dev spont 16.18 → 14.79; itest spont 7.06 → 6.81.

New prediction files: `zs_llm7b_pc`, `llm7b_lora_v1_pc`. The clip-level bootstrap rows above that compare against LoRA v1 or zero-shot on spontaneous sets used the old predictions.

### Cross-fit relabelling (exp27a/a2 + exp27b, 2026-10-13)

Unbiased hybrid: zero-shot + fold LoRA (never heard the labelled speakers); CTC only proposes candidates.
- vs pseudo_v1: 3,864 shared clips, **16.4 % changed** (WER old → new 1.8 %). With in-sample scorers (exp17/26) it was 5.9–6.9 %.
- Hybrid labels: 7,055 clips → 4,024 kept (16.5 h; 50+ speakers 2.9 h).
- **LoRA v5** (v2 recipe on these labels) vs v2, clip bootstrap:
  - dev unseen +0.36 [−0.72, +1.51]; itest unseen 0.00;
  - **dev spont −0.72** [−1.24, −0.21]; itest spont +0.37 [−0.14, +0.89];
  - OOD +0.38 [−1.02, +1.74].
  - Neutral to mixed: the only significant gain is on the tuning set dev spont. Not adopted.

### Exact LoRA soup (exp28b, rank 64 = mean of the v1–v4 updates, `scripts/lora_soup.py`)

| System | dev unseen | itest unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|---|
| LoRA v1 (pause chunking) | 12.59 | 20.17 | 14.79 | 6.81 | 14.88 |
| LoRA v2 | 13.12 | 20.17 | 14.93 | 6.62 | 15.11 |
| LoRA v3 | 12.68 | 20.00 | 15.03 | 6.95 | 15.03 |
| **soup (v1–v4)** | **12.50** | **19.22** | **14.77** | **6.55** | **14.73** |

- The soup is the best single model on every set.
- vs v1: itest unseen −0.96 [−1.91, 0.00], itest spont −0.26 [−0.59, +0.05], others n.s.
- vs v2: dev unseen −0.62, itest unseen −0.96 [−1.98, 0.00], others n.s.
- **As a hybrid scorer (exp31, `reports/abl_pool_dev_aug_soup*.csv`):** 3 from-scratch searches, mean ± sd, vs the full hybrid 9.64 / 16.99 / 13.52 / 6.33 / 14.22:
  - soup added: 9.70 / 16.99 / 13.60 / 6.32 / 14.25;
  - soup replacing v1 + v2: 9.79 / 16.70 ± 0.69 / 13.62 / 6.35 / 14.33.
  - Neutral. Not adopted; no final-mode v3/v4 retraining needed.

### TTS-likelihood verifier: OmniVoice zero-shot and fine-tuned (exp29, 2026-10-12)

`scripts/tts_score.py`: masked-token likelihood of the clip's own audio tokens (HiggsAudioV2, 8 codebooks) given each candidate text, with bidirectional attention, 4 masks of 50 %.
- **Bug found first:** without the bidirectional attention mask the model ran causally. The reference text then scored the same as an unrelated sentence or empty text, at a loss of about ln 1024.
- **With the fix:** the reference text clearly beats an unrelated sentence (OOD first-codebook loss 2.61 vs 3.00). Near-miss candidates are another matter.
- **exp29:** LoRA fine-tune (rank 16, OmniVoice recipe) on the 19,065 labelled `train` clips (27.6 h; dev/held-out/OOD/test excluded), 3,000 steps. Held-out loss was best at step 1,000 (3.88).

Pairwise accuracy on all 109 OOD clips (same pool as the hybrid):

| Scorer | pairwise | top-1 WER |
|---|---|---|
| OmniVoice zero-shot (lp_tts / first codebook) | 53.4 / 55.6 | 24.8 / 23.5 |
| OmniVoice fine-tuned, step 1,000 | 55.3 / 55.5 | 23.0 / 23.9 |
| OmniVoice fine-tuned, step 3,000 | 55.4 / 55.5 | 23.3 / 23.9 |
| omniASR-LLM-7B zero-shot likelihood | 59.3 | 23.0 |
| LoRA v1 / v2 likelihood | 71.3 / 73.5 | 14.6 / 14.7 |

- Random and contiguous-span masks, mask ratios 0.5–1.0, and the first codebook alone all stay near chance (zero-shot smoke test on 20 OOD clips: 42–53 %).
- **Negative:** text-conditioned TTS likelihood does not discriminate near-miss transcripts here, even after fine-tuning on our data. It is not used.

### Final systems (exp10, "final" mode: all labelled data except dev)

joint_final = joint rescoring of final CTC (v3 recipe, 12k steps, soup + WiSE 0.85) with final
LoRA v1 + v2 and zero-shot LLM-7B; scripted weights fixed from dev-mode tuning, spontaneous
re-tuned on dev_spont. Submission: `artifacts/submission/qxp_final.tsv` (GPU box).

| Set | joint_v3 (dev-mode models) | joint_final | Clean in final mode? |
|---|---|---|---|
| dev spont | 14.20 / 2.48 | **13.95 / 2.48** | yes (speakers never trained on) |
| OOD radio | 15.11 / 2.46 | **14.58 / 2.49** | yes |
| dev unseen | 9.91 / 1.76 | 7.95 / 1.46 | no (sentences now trained on) |
| itest unseen | 16.00 / 2.97 | 10.78 / 2.11 | no (speakers + sentences trained on) |
| itest spont | 6.42 / 0.85 | 5.05 / 0.58 | no |

### Like-for-like comparison with the organizer baseline (same sentence knowledge)

The organizers' cv26 models were trained on every scripted sentence (split by speaker only), and
our `dev_scripted` / `itest_scripted` are exactly their cv26 dev / test splits. In "org" mode our
LM is built from all scripted sentences too (`scripts/build_lm.py --mode org`), and LM weights are
tuned on all dev clips (`scripts/rescore.py --tune-on all`). Acoustic model: ctc1b_v2 (dev mode —
it never heard the held-out-sentence clips, so this still slightly favours the organizers).

| Split (organizers' cv26) | Organizer XLS-R-300M+CPT (best released, greedy) | Ours: CTC-1B + org LM (word beam + char-10g rescoring) |
|---|---|---|
| dev (948 clips) | 15.96 / 2.58 | **6.33 / 1.58** (670 clips) and **5.54 / 1.25** (278 held-out) |
| test (909 clips) | 17.85 / 3.25 | **7.96 / 2.11** (612 clips) and **9.04 / 2.04** (297 held-out) |

Under matched information our system roughly halves the baseline's error on the organizers' own
test split. (Their models are decoded greedily; only their text outputs were stored, so our LM
could not be applied to them.) Without any sentence knowledge (dev protocol, the setting that
matches the hidden test), see the tables above.

Observations: LM gains are moderate (5–10 % rel.) and do not transfer to the OOD radio domain
(vocabulary not covered by the LM text); WiSE-FT is the only technique so far that improves OOD.

## Published / organizer benchmarks (different test sets — not directly comparable)

| System | Scripted WER/CER | Spontaneous WER/CER | Source |
|---|---|---|---|
| XLS-R-300M + CPT, validated (organizers) | 1.19 / 0.19 | 13.61 / 1.73 | organizer repo `results/summary.csv` (seen-sentence scripted test) |
| XLS-R-300M + CPT, + silver (organizers) | 2.11 / 0.30 | 3.15 / 0.41 | same; spontaneous test is 97 % silver (omniASR-7B) labels |
| omniASR-LLM-300M-v2 zero-shot | 7.07 / 0.89 | 11.98 / 1.54 | organizer repo |
| omniASR-LLM-7B-v2 zero-shot | 26.6 / 6.2 | 11.1 / 1.9 | Huaman et al. 2026 (AmericasNLP) |
| omniASR-CTC-1B qxp single-language FT | — / 1.61 | — | Omnilingual ASR paper §5.7.5 (Meta's qxp split) |
| omniASR-LLM-7B-v2 zero-shot | — / 1.32 | — | Omnilingual ASR paper (Meta's qxp split) |

## Organizer checkpoints re-evaluated on our dev sets (greedy)

| Checkpoint | dev_scripted WER / CER | dev_spont WER / CER | Caveat |
|---|---|---|---|
| whisper-base-qxp-finetuned | 15.1 / 2.5 | 35.2 / 19.1 | trained on all dev *sentences* |
| xls-r-cpt-qxp-cv26-validated | 16.0 / 2.6 | 7.1 / 1.2 | trained on all dev sentences **and 307/310 dev_spont clips** |
| wav2vec2-base-qxp-finetuned | 21.7 / 3.5 | 40.6 / 7.4 | older split |

Leakage: the organizer cv26 spontaneous train split contains 307/310 of our dev_spont and 290/304
of our itest_spont clips (their XLS-R gets 1.4 % WER on itest_spont), and every scripted sentence.
Spontaneous comparisons on our dev are therefore meaningless; scripted ones favour them (sentence
memorisation that cannot help on the hidden test, which reads new sentences). A fair comparison
requires retraining their recipe on our split — queued.

## Statistical analysis for the paper (2026-10-13; `scripts/paper_stats.py` → `reports/paper_stats.md/.csv`)

**Method** (`src/qxp_asr/stats.py`, unit-tested including a null-coverage simulation):
- Paired bootstrap of corpus-WER differences, 10,000 resamples, 95 % percentile intervals.
- Read speech: speakers × sentences resampled jointly (pigeonhole bootstrap, Owen 2007).
- OOD: clips resampled (no speaker labels).
- Spontaneous: exact sign test over speakers. The clip interval is descriptive only, because 4 and 6 speakers are too few for a cluster bootstrap; with 4 speakers p < 0.05 is impossible.
- Benchmark (public / organizer) outputs get the same text normalization as in `bench_all_systems.py`.

**Confirmatory family, Holm-corrected:**

| Test | Δ WER | 95 % CI | p | Holm p |
|---|---|---|---|---|
| full vs LoRA v1, itest unseen | −3.30 | [−6.51, −0.79] | 0.011 | **0.045** |
| full vs LoRA v1, OOD | −1.21 | [−2.35, −0.15] | 0.039 | 0.116 |
| views vs none, itest unseen | −0.52 | [−1.77, 0.00] | 0.36 | 0.36 |
| views vs none, OOD | −0.83 | [−1.80, +0.08] | 0.080 | 0.160 |

- Views lower WER on every read and OOD set, in all 3 weight searches and in both designs (ablation; restricted-pool control). Only one test is significant: removing view candidates from the tuned pool, OOD +1.06 [+0.22, +1.97].
- Views vs an equal number of samples: n.s. (OOD −0.50 [−1.56, +0.48]).
- Finding 1, sentence reuse: CTC + 4-gram vs greedy is −8.1 (dev seen) vs +4.0 [+1.1, +7.2] (dev unseen). Interaction −12.1 [−17.9, −7.3] (dev), −12.9 [−18.7, −7.4] (itest).
- Organizer XLS-R CPT validated, normalized: 17.46 / 3.18 on the 909-clip itest scripted set (17.85 / 3.25 without our normalization). Our dev-mode system: −3.94 [−6.66, −1.18]; on the unseen third, −0.87 [−5.14, +3.57], n.s.

**Training seeds (exp32a/b): LoRA v2 recipe, seeds 0 / 1 / 2:**

| Set | seed 0 | seed 1 | seed 2 | range |
|---|---|---|---|---|
| dev unseen | 13.12 | 13.66 | 12.86 | 0.80 |
| itest unseen | 20.17 | 20.09 | 20.26 | 0.17 |
| dev spont | 14.93 | 14.55 | 14.77 | 0.38 |
| itest spont | 6.62 | 6.83 | 7.09 | 0.47 |
| OOD | 15.11 | 14.43 | 15.63 | 1.20 |

Single-model differences of up to about 1 WER on OOD are within training noise.

**RL arms re-decoded with infer_llm (exp32b).** LoRA v2 decoded twice is identical on read speech and OOD (decoding is deterministic). Δ vs LoRA v2 (unseen half 1 / dev spont ≤ 39 s / OOD); all intervals include 0:
- SFT: +0.68 / +0.31 / +1.06 [0.00, 2.11]
- GSPO: +0.68 / +0.22 / +0.38
- MWER: +0.26 / +0.02 / +0.23
- Dr. GRPO: +1.62 [−0.21, 3.85] / +0.14 / +0.38
- MWER with mixed rollouts (step 150): +0.26 / +0.02 / +0.23. The point estimates equal MWER's by coincidence; the outputs differ on 214 clips and the intervals differ.

**Verifier.** The hybrid's pairwise accuracy (clip bootstrap) is 78.3 [76.0, 80.5] dev unseen, 76.6 [74.5, 78.6] itest unseen, 71.2 [69.0, 73.4] dev spont, 74.8 [70.4, 78.7] OOD. At these accuracies the noisy-oracle WER lies inside our WER's interval on 3 of 4 sets; itest unseen is the exception.

**Exploration, paired (pass@k rerun with per-clip output, `artifacts/cands/passk_v2_h0.parquet`; reproduces 6.0 / 14.0 / 16.7 / 24.0 %).** Exact McNemar test on the same 150 held-out-sentence clips:
- samples (T = 1) vs beam: 17 vs 5 discordant clips, p = 0.017;
- views vs beam: 21 vs 5, p = 0.0025;
- views vs samples: 9 vs 5, p = 0.42;
- union vs views: 11 vs 0, p = 0.001.

**Statistical audit (second pass) applied to the paper.**
- The confirmatory family is stated as designated after the experiments, not preregistered.
- The comparator is "best component on dev unseen" (LoRA v1). The soup is better but is not a component.
- Stars are labelled unadjusted and design-based.
- The soup gain vs v1 is n.s. (itest unseen −0.96 [−2.56, +0.28]).
- The views claim is hedged to "consistent sign, not significant".
- The 72 % interval was dropped (events cluster by sentence and speaker).
- Exploration rates use McNemar instead of Wilson.

### Internal-LM (prior) subtraction for the hybrid (exp34a, 2026-10-14)

- **Method:** every candidate of `pool_dev_aug` is also scored by zs / v1 / v2 on 1 s of near-silence (`llm_candidates --blank-audio`). The negated priors are added as features `lp_ilm_*`, so a positive weight subtracts the prior. The hybrid is re-tuned from scratch with 3 seeds.
- **Smoke test:** the priors are finite and correlate only weakly with the audio scores (r = 0.13).
- **Mean WER:** 9.76 / 17.10 / 13.31 / 6.12 / 14.10 (dev unseen / itest unseen / dev spont / itest spont / OOD), vs the full hybrid 9.64 / 16.99 / 13.52 / 6.33 / 14.22.
- **Paired, design-based** (`compare_runs.py`): nothing significant.
  - read: +0.12;
  - spont: −0.22 and −0.21 (itest clip interval [−0.49, +0.06]);
  - OOD: −0.13 [−1.06, +0.86].
- **Neutral.** Not adopted (at most a small, unconfirmed gain on spontaneous speech).

### Training-seed ensemble (exp35 + exp37, 2026-10-14)

- **Seed soup** (exact soup of LoRA v2 seeds 0/1/2, rank 48): 13.04 / 20.35 / 14.51 / 6.74 / 14.88 (dev unseen / itest unseen / dev spont / itest spont / OOD).
  - vs v2: all differences n.s. (dev spont −0.41, 2/4 speakers better; OOD −0.23 [−1.28, +0.82]).
- **Seeds 1 and 2 as extra hybrid scorers** (3 from-scratch searches): 9.70 / 17.13 / 13.53 / 6.36 / 14.12.
  - vs the full hybrid: all |Δ| ≤ 0.14, n.s. (OOD −0.10 [−0.49, +0.35]).
- **Neutral.** With the 4-adapter soup, the ILM priors and a second CTC model, this is the fourth time that adding scorers of the same family does not move the hybrid.

### Encoder + decoder LoRA (exp34b, the exp08 recipe, never run before; 2026-10-14)

Rank-16 LoRA on the 7B encoder as well as the decoder; same data as LoRA v2; lr 2e-4, 60 s × 4 batches, 1,500 steps. Best dev step 900 (score 12.94; last 12.98).

| System | dev unseen | itest unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|---|
| LoRA v2 (decoder only) | 13.12 | 20.17 | 14.93 | 6.62 | 15.11 |
| **LoRA enc+dec** | **10.89** | **16.00** | 14.57 | 6.79 | 15.41 |

Paired, design-based:
- dev unseen −2.23 [−5.20, −0.21], p = 0.03.
- **itest unseen −4.17 [−7.32, −1.46], p = 0.006.**
- Spontaneous n.s. (dev 2/4, itest 4/6 speakers better).
- OOD +0.30 [−1.49, +2.15].

As a single model it is better on itest unseen than the full hybrid (16.87). Acoustic adaptation to the read-speech speakers is what the decoder-only LoRA misses; spontaneous and OOD speech gain nothing.

Next:
- exp39 (server A): final-mode enc+dec LoRA.
- exp40 (server B): enc+dec as candidate generator and scorer in the hybrid (dev mode, evaluation clips).

**Encoder+decoder LoRA, seed 1 (exp42):** 10.54 / 15.30 / 14.53 / 7.27 / 14.95 (dev unseen / itest unseen / dev spont / itest spont / OOD).
- vs LoRA v2: dev unseen −2.59 [−5.49, −0.39], p = 0.025; **itest unseen −4.87 [−8.37, −1.92], p = 0.002**; spontaneous and OOD n.s.
- vs seed 0: no significant difference on any set (itest unseen −0.70 [−2.64, +1.05]).
- The read-speech gain of acoustic adaptation replicates across seeds.

**Incident (2026-10-14).**
- exp39 (final enc+dec training) finished at 17:26 UTC. exp41b failed one minute later: server A's `llm_candidates.py` lacked `--encoder-rank`, because the update had only been copied to server B. Watchers only matched "Traceback", so the usage error went unnoticed. GPU A was idle until 00:03.
- B was idle 19:55–00:04 after exp42.
- No results were lost. Since then, code hashes are checked on each server before launches, and watchers also report processes that exit without their completion marker.

### Why does encoder adaptation help? Error types and verifier accuracy (2026-10-15)

**Error types** (`scripts/error_types.py`, `reports/error_types.csv`). Itest unseen, WER points per class:

| System | q/k | ejective | aspiration | other 1-letter | suffix (same stem) | other word | ins | del | WER |
|---|---|---|---|---|---|---|---|---|---|
| zero-shot | 3.04 | 2.00 | 0.78 | 10.43 | 2.17 | 7.39 | 1.65 | 0.43 | 27.91 |
| LoRA v1 (dec) | 1.57 | 1.83 | 0.26 | 6.78 | 2.26 | 5.65 | 1.39 | 0.43 | 20.17 |
| LoRA v2 (dec) | 1.30 | 2.09 | 0.43 | 6.35 | 2.26 | 5.83 | 1.65 | 0.26 | 20.17 |
| enc+dec seed 0 | 1.13 | 1.22 | 1.04 | 5.91 | 1.83 | 4.00 | 0.78 | 0.09 | 16.00 |
| enc+dec seed 1 | 0.96 | 1.39 | 0.70 | 5.74 | 2.00 | 3.65 | 0.70 | 0.17 | 15.30 |

- **Decoder adaptation fixes spelling:** one-letter errors incl. q/k fall from 13.5 to about 7.7.
- **Encoder adaptation fixes mishearing:** "other word" 5.8 → 3.7–4.0, insertions 1.65 → 0.7–0.8, ejectives 2.1 → 1.2–1.4.
- **Suffix errors** (same stem, wrong ending) do not move at any stage (2.2 → 2.3 → 1.8–2.0).
- Dev unseen shows the same pattern.

**Verifier pairwise accuracy** (`verifier_curve.py --tag`), with vs without the enc+dec scorer on the same clips:

| Set | without | with |
|---|---|---|
| dev unseen | 78.3 [76.0, 80.5] | 79.7 [77.7, 81.8] |
| itest unseen | 76.6 [74.5, 78.6] | 78.6 [76.5, 80.7] |
| dev spont | unchanged (lp_ed weight 0) | |
| OOD | unchanged | |

- WER change predicted by the local slope of the two synthetic families:
  - itest unseen: −2.0 (noisy oracle) / −0.9 (blend); observed −1.39.
  - dev unseen: −1.3 / −0.4; observed −0.53.
- The two families bracket the observed effect of a real verifier improvement.

### Final-mode submission with the enc+dec scorer (exp39 + exp41b, 2026-10-15) and seed soup as scorer (exp43/43a)

- **`qxp_final_ed.tsv`** (hybrid_final_ed: final enc+dec LoRA as a 4th scorer with the hybrid_dev_ed weights; ed views count in the view feature; ed-only candidates dropped).
  - It differs from `qxp_final_indomain_aug.tsv` on **25 of 659 test lines**, all scripted: the spontaneous weight of lp_ed is 0.
  - Clean final-mode sets are unchanged: dev spont 13.76 / OOD 14.12 for both.
  - Final enc+dec single model vs final LoRA v2: dev spont −0.19, OOD −0.15 (n.s.).
  - Evidence for the scripted change comes from dev mode: itest unseen −1.39 [−3.08, +0.16].
  - **Recommended submission** unless a later run improves on it.
- **Soup of the two enc+dec seeds as the 4th scorer** (exp43a, 3 searches): 9.26 / 15.86 / 13.51 / 6.28 / 13.97, vs seed 0 as scorer 9.29 / 15.71 / 13.49 / 6.29 / 13.95. All |Δ| ≤ 0.14. Neutral; not adopted.

### What else to adapt? (exp44a, exp45; 2026-10-15)

| System | dev unseen | itest unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|---|
| enc+dec LoRA (exp34b) | 10.89 | 16.00 | 14.57 | 6.79 | 15.41 |
| + connector, qxp language embedding, all norms (exp44a) | 11.07 | 15.91 | 14.62 | 6.69 | 15.41 |
| enc+dec, cross-fit elderly pseudo-labels, spontaneous up-weighted (exp45) | 11.07 | 16.35 | 14.17 | 6.98 | 15.03 |

- **exp44a:** trains `encoder_proj` (8.4 M), `lang_embeddings` (qxp row) and 282 Layer/RMSNorms (1.2 M) fully, lr 5e-5. vs enc+dec: all |Δ| ≤ 0.18, n.s. The never-trained small modules add nothing once encoder and decoder carry LoRA.
- **exp45:** vs enc+dec: dev spont −0.40 [−1.00, +0.24] (3/4 speakers better), itest spont +0.19, OOD −0.38 [−1.99, +1.17], read +0.2/+0.35. All n.s.
  - vs LoRA v2: still −3.83 on itest unseen [−6.98, −1.14].
  - Elderly pseudo-labels do not move spontaneous speech, even for the encoder.

### Encoder-only LoRA: the encoder is the site that matters (exp44b, 2026-10-15)

Rank-16 LoRA on the encoder only; the decoder is frozen (no decoder LoRA). Same data and steps as enc+dec. Best dev score 13.55.

| System | dev unseen | itest unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|---|
| LoRA v2 (decoder only) | 13.12 | 20.17 | 14.93 | 6.62 | 15.11 |
| **encoder only** | 11.70 | 16.43 | 14.75 | 6.70 | 14.95 |
| encoder + decoder | 10.89 | 16.00 | 14.57 | 6.79 | 15.41 |

- vs v2: itest unseen −3.74 [−7.22, −0.72], p = 0.02; dev unseen −1.43 [−4.46, +0.94].
- vs enc+dec: +0.43 [−0.97, +1.89] itest unseen, n.s. everywhere.
- **Error types** (itest unseen; `reports/error_types_enc.csv`): encoder-only one-letter (q/k + other) 7.48, other word 4.00, insertions 0.78. These equal enc+dec (7.04 / 4.00 / 0.78). Decoder-only v2: 7.65 / 5.83 / 1.65.
- **Correction to the earlier decomposition:** the decoder is not the spelling site. Encoder adaptation alone fixes spelling and misheard words; decoder-only LoRA fixes spelling only. Once the encoder is adapted, the decoder LoRA adds little.

**exp46 (RL from enc+dec, encoder LoRA frozen):**
- The GSPO arm finished.
- The SFT control ran out of memory: the frozen encoder LoRA had requires_grad=True, so SFT backpropagated through the 7B encoder. Fixed in train_rl.py; rerun as exp46b.
- B was idle 09:25–15:46 UTC, unnoticed because the local watcher had been stopped for memory.

### RL from the encoder-adapted model (exp46/46b) and LoRA rank 64 (exp47), 2026-10-15

**RL from enc+dec, encoder LoRA frozen** (sentence cross-fit, half 0, 400 steps). Paired vs the enc+dec start (rl_unseen / rl_spont / OOD):
- SFT: 0.00 [−1.30, +1.12] / +0.23 / +0.76 [−0.16, +1.71].
- GSPO: +1.45 [−0.61, +3.48] / +0.25 / **+2.11 [+0.78, +3.51], p = 0.002**.
- GSPO vs SFT: OOD +1.36 [+0.30, +2.54], p = 0.01.
- RL still does not help new sentences, and GSPO hurts OOD. The earlier weak "RL forgets less than SFT" direction (from LoRA v2) does not replicate.

**Enc+dec LoRA rank 64** (rank 16 before): 10.98 / 14.78 / 14.55 / 7.09 / 14.95.
- vs rank 16: itest unseen −1.22 [−3.12, +0.55], n.s.; dev unseen +0.09; OOD −0.45.
- Lowest single-model itest-unseen WER so far, but within seed spread (rank-16 seeds: 16.0 / 15.3).

### LoRA rank scan for enc+dec (exp47, exp50) and rank 64 as 4th scorer (exp48/48a), 2026-10-16

| rank | dev unseen | itest unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|---|
| 16 (seed 0 / seed 1) | 10.89 / 10.54 | 16.00 / 15.30 | 14.57 / 14.53 | 6.79 / 7.27 | 15.41 / 14.95 |
| 64 | 10.98 | 14.78 | 14.55 | 7.09 | 14.95 |
| 128 | 10.89 | 15.65 | 14.65 | 7.55 | 15.63 |

- No trend. Dev unseen is identical across ranks, and itest unseen moves within seed spread: r128 vs r16 −0.35 [−2.58, +1.86]; r128 vs r64 +0.87 [−0.51, +2.62].
- Rank 32 (exp50) is still running.
- **Rank 64 as the 4th hybrid scorer** (3 searches; vs rank 16 as scorer): 9.38 / 16.00 / 13.50 / 6.19 / 14.27 vs 9.29 / 15.71 / 13.49 / 6.29 / 13.95. Slightly worse (OOD +0.33 [−0.03, +0.79]); not adopted.
- The final-mode rank-64 model (exp49, trained speculatively) is unused. **The submission stays `qxp_final_ed.tsv`.**

## Why encoder adaptation helps: ranking, not coverage (`scripts/adaptation_mechanism.py`, 2026-10-16)

Shared pool `pool_ctrl_ed`: every candidate is scored by every model; clips ≤ 39 s. CSVs: `reports/mechanism_{coverage,ranking,pairs}.csv`.

**Coverage: the oracle of each model's own beam-8 list.** Itest unseen: zs 20.2, v1 14.1, v2 14.0, ed 14.0. Dev unseen: 14.0 / 7.7 / 7.95 / 7.95. Spontaneous and OOD likewise within about 0.6.
- The decoder-only model's N-best already contains transcripts as good as the encoder-adapted model's. The frozen encoder did not discard the information.

**Ranking: pairwise accuracy of each model's own likelihood / WER of its top pick on the shared pool.**

| Set | zs | v1 | v2 | ed |
|---|---|---|---|---|
| dev unseen | 67.2 / 20.6 | 77.6 / 12.4 | 78.2 / 13.1 | **82.7 / 10.7** |
| itest unseen | 64.6 / 27.7 | 75.4 / 19.9 | 75.7 / 20.0 | **80.9 / 16.3** |
| dev spont | 71.0 | 69.3 | 69.3 | 71.2 |
| itest spont | 66.9 | 67.3 | 67.6 | 70.2 |
| OOD | 61.3 | 74.4 | 76.3 | 77.1 |

- Encoder adaptation raises verification accuracy by about 5 points on read speech with the same coverage. On its own it beats the tuned hybrid (76.6–78.6).
- Gains are small where the encoder saw no matching audio (spontaneous, OOD).

**Which distinctions** (read speech; pairwise accuracy v2 → ed by the type of difference between the two candidates):

| Pair type | pairs | v2 → ed |
|---|---|---|
| several words | 36,571 | 65.0 → 73.2 (+8.2) |
| other word | 1,440 | 85.3 → 92.6 (+7.3) |
| ejective | 375 | 80.8 → 87.7 (+6.9) |
| q/k | 611 | 88.9 → 93.6 (+4.7) |
| suffix | 984 | 94.9 → 96.6 (+1.7) |
| other one-letter | 4,256 | 93.3 → 94.7 (+1.4) |
| aspiration | 275 | 89.8 → 88.0 (−1.8) |

- Spontaneous: small and mixed (several words +2.0; vowels −7.6).
- OOD: about 0 or negative (aspiration −14.8 on 54 pairs).

**Intervals (added 2026-10-16; pairs of one clip are dependent, so clusters are resampled):** read speech by speakers × sentences (pigeonhole), spontaneous and OOD by clips.
- All read pairs: 69.7 → 76.9, +7.3 [+3.6, +11.7].
- Several words: +8.2 [+4.0, +13.4]. Other word: +7.3 [+2.2, +12.6].
- Ejective: +6.9 [−2.1, +18.1]. q/k: +4.7 [−2.4, +13.5]. Aspiration: −1.8 [−13.4, +10.3]. None of the one-letter contrasts is significant (199 / 252 / 148 clips).
- Spontaneous, all pairs: +1.9 [+0.4, +3.5]. OOD, all pairs: +1.2 [−0.9, +3.3].
- **Correction:** the per-contrast story below (ejectives, q/k) is a direction, not a finding. The paper now says only that one-letter gains are positive but not significant. Probes (exp56) test it at the representation level.

**Interpretation:** the "verification bottleneck" of the paper is largely acoustic discrimination between plausible hypotheses: multi-word alternatives and, as a non-significant trend, Quechua-specific contrasts (ejectives, uvular vs velar). Decoder adaptation cannot supply it, and external verifiers did not either. Encoder adaptation on matching audio does, but only for that audio.

**Encoder layer norms** (squared ‖ΔW‖ share by quarter of the 128-layer 7B encoder): enc-only 0.21 / 0.28 / 0.31 / 0.21; enc+dec 0.26 / 0.30 / 0.28 / 0.16, i.e. spread across depth. enc+dec puts 4.5× more update into the encoder than into the decoder. Causal test: exp53 (halves).

## Where to adapt: encoder scale, encoder layers, and inside the encoder (exp53, exp54, exp56; 2026-10-16)

All runs use the v2 data and recipe, 1,500 steps, rank 16, one seed. Intervals are from `scripts/compare_runs.py`: read speech by speaker × sentence, spontaneous by speaker sign test, OOD by clips.

### Encoder scale (exp54)
omniASR-LLM-300M / 1B / 7B share the **same decoder** (12 layers, 4096 wide). Only the encoder differs (24 × 1024, 48 × 1280, 128 × 2048).

| WER itest unseen / OOD | zero-shot | decoder LoRA | encoder LoRA | enc+dec LoRA |
|---|---|---|---|---|
| 300M encoder | 26.1 / 23.3 | 19.8 / 20.2 | 25.4 / 21.3 | 20.6 / 19.8 |
| 1B encoder | 26.0 / 20.4 | 20.9 / 18.1 | 21.6 / 18.6 | 20.4 / 18.1 |
| 7B encoder | 27.9 / 23.3 | 20.2 / 15.1 | **16.4** / 15.0 | **16.0** / 15.4 |

Encoder-only LoRA minus decoder-only LoRA, itest unseen:
- 300M: +5.6 [+2.2, +10.0] (worse).
- 1B: +0.7 [−2.1, +4.0].
- 7B: −3.7 [−7.2, −0.7].

Enc+dec minus decoder-only, itest unseen: 300M +0.8 (n.s.), 1B −0.4 (n.s.), 7B −4.2. On dev unseen, enc+dec beats decoder-only at 1B (−2.7 [−5.7, −0.6]) and at 300M (−2.3, p = 0.08).

- **The "adapt the encoder" result is a property of the 7B encoder, not of LLM-ASR in general.** With the decoder adapted, all three encoder sizes land at about 20 on itest unseen. Only the 7B encoder turns adaptation into a further 4 points.
- Confound: rank-16 encoder LoRA has about 10× fewer parameters on the 300M encoder than on the 7B one. The learning rate was tuned at 7B.

### Which encoder layers (7B, encoder-only LoRA, decoder frozen)

**Trained on half of the layers (exp53).**
- Bottom half (layers 0–63): 16.1. This equals all layers (16.4; −0.35 [−2.0, +1.2]).
- Top half (layers 64–127): 20.6, i.e. no read-speech gain over decoder LoRA v2. It is worse than the bottom half: +4.5 [+1.2, +8.1].
- OOD: top half 14.4, bottom half 15.7 (n.s.).

**Causal block ablation of the trained all-layer adapter (exp56).** The LoRA update is kept only in the given layers and zeroed elsewhere. Itest unseen / OOD:

| update kept in | WER | Δ vs full adapter |
|---|---|---|
| none (zero-shot) | 27.9 / 23.3 | |
| 0–31 | 22.4 / 19.8 | |
| 32–63 | 23.2 / **15.3** | |
| 64–95 | 25.4 / 21.2 | |
| 96–127 | 27.2 / 24.2 | |
| 0–63 | 19.1 / 15.3 | +2.7 [+0.8, +5.1] |
| 64–127 | 24.9 / 22.7 | |
| 32–127 (bottom quarter removed) | 20.1 / 14.3 | +3.7 [+1.0, +7.3] |
| 0–95 (top quarter removed) | **16.2** / 14.8 | −0.3 [−1.6, +1.0] |
| all (encoder-only LoRA) | 16.4 / 15.0 | |

- The top quarter's update, the **largest** by norm (geometry: rel. size grows with depth), is causally unnecessary. Removing it changes nothing.
- The bottom quarter is necessary for the read-speech gain (+3.7 without it) but not sufficient alone (22.4).
- The read-speech gain is non-additive: single quarters recover little, while 0–95 recovers all of it.
- **OOD is different:** layers 32–63 alone recover the whole OOD gain (15.3 vs 23.3 zero-shot, −7.9 [−10.1, −5.8]; full 15.0). The read-speech gain needs the bottom quarter as well. Two separable effects: a mid-encoder shift that transfers out of domain (it may be the orthographic/language convention that decoder LoRA also learns; OOD 15.1), and a low-level acoustic adaptation that helps only matching (Common Voice) audio.

### Inside the encoder (exp56, `scripts/encoder_probe.py`)

**Linear CKA**, pretrained vs enc+dec LoRA (dev frames), with the same for the two enc+dec seeds as a noise floor:

| layer | 0 | 8 | 16 | 32 | 64 | 96 | 120 | 127 |
|---|---|---|---|---|---|---|---|---|
| zs vs ed | 1.00 | 0.98 | 0.84 | 0.84 | 0.83 | 0.73 | 0.74 | 0.47 |
| ed vs ed seed 1 | 1.00 | 0.95 | 0.65 | 0.67 | 0.65 | 0.65 | 0.58 | 0.31 |

OOD frames: zs vs ed 0.85 at layer 16, falling to 0.56 at layer 127.

- The representation changes from layer ~16 on and most at the top.
- The two seeds differ from each other **more** than either differs from the pretrained encoder (0.65 vs 0.84; roughly 0.84², as for independent perturbations of the same base). Each seed moves the representation in its own direction, consistent with the low cross-seed subspace overlap of the LoRA updates (0.02–0.08). Both reach the same WER.
- Descriptive only: CKA is sensitive to a few large dimensions.

**Linear probes** on force-aligned stops. Logistic regression trained on dev clips and tested on itest speakers, with balanced test classes. Control: shuffled labels, always about 0.50.

| contrast (test items) | best pretrained layer | pretrained, layers ≥ 56 | adapted, layers ≥ 56 |
|---|---|---|---|
| ejective vs plain (892) | 0.953 (L32) | 0.94 | 0.95 |
| aspirated vs plain (704) | 0.963 (L48) | 0.95 | 0.965–0.976 |
| q vs k, plain (4,494) | 0.969 (L80) | 0.967–0.969 | 0.974–0.976 |

- The **pretrained 7B encoder already encodes all three Quechua contrasts linearly at 94–97%**, from about layer 24 on. Adaptation adds 1–2 points in the upper layers (McNemar p < 0.05 from layer ~48–56 on for all three; items within a clip are dependent, so these p-values are optimistic).
- This matches "coverage, not ranking" at the representation level. The information is there before adaptation; adaptation makes it slightly more accessible. It is not a new phonological category.
- Aspiration also improves at the representation level, although aspiration pairs did not improve in ranking (n.s. either way).

### Synthesis
- Encoder adaptation does not teach the 7B encoder new contrasts; they are already linearly present.
- It acts in the lower and middle layers: the bottom quarter is necessary, the top quarter dispensable.
- It improves the ranking of the hypotheses the model already reaches.
- It pays off only for the large encoder.
- Its in-domain (read-speech) and out-of-domain effects live in different layers.

## Transfer, capacity control, subspace surgery, strong augmentation (exp55, 57, 58, 51; 2026-10-17)

### Whisper large-v3 (exp55; no Quechua in pretraining, Spanish language token)
Same data and recipe as omniASR. One seed; a second seed of enc/dec is running (exp61). Clips ≤ 30 s.

| itest unseen / OOD WER | zero-shot | decoder LoRA | encoder LoRA | enc+dec LoRA |
|---|---|---|---|---|
| Whisper large-v3 | 141 / 254 (hallucinates) | 34.1 / 32.8 | 27.7 / 24.0 | 25.0 / 23.8 |

- Encoder minus decoder: itest unseen −6.4 [−13.0, −1.1], dev unseen −9.2 [−14.9, −3.5], OOD −8.8 [−11.4, −6.2]. Spontaneous: better for 4/4 and 6/6 speakers (itest sign test p = 0.03).
- Enc+dec minus encoder: −2.7 [−6.7, +0.8] (n.s.).
- **Adapting the encoder transfers to another model family, and there it also wins out of domain.**

### Capacity control (exp57): parameter-matched encoder LoRA on the small encoders
Encoder-only LoRA at about 71M parameters (300M encoder at rank 160, 1B at rank 64), vs 75M for 7B at rank 16.

| itest unseen | decoder LoRA | encoder LoRA r16 | encoder LoRA, matched |
|---|---|---|---|
| 300M encoder | 19.8 | 25.4 | 22.9 (+3.0 vs decoder [−0.1, +6.3]) |
| 1B encoder | 20.9 | 21.6 | 21.4 (+0.5 [−2.5, +4.0]) |

- More capacity helps the 300M encoder (−2.5 [−5.8, −0.1] vs r16) but does not close the gap to decoder LoRA.
- **In the omniASR family the encoder advantage appears only at 7B, and LoRA capacity does not explain it.** With Whisper (a 635M encoder) it is large.
- So it is not encoder size as such. A candidate explanation, untested: the omniASR decoder already knows qxp (the zero-shot models cover it), so decoder LoRA mostly fixes orthography and goes far for every encoder. Whisper's decoder knows no Quechua, so its decoder LoRA is weak and the encoder carries the adaptation.

### Spectral truncation and seed consensus (exp58; weight surgery, no retraining)
Encoder-only LoRA truncated per projection to its top-k singular directions (share of update energy: top-1 0.61, top-2 0.74, top-4 0.85, top-8 0.93):

| k | 1 | 2 | 4 | 8 | 16 (trained) |
|---|---|---|---|---|---|
| itest unseen | 17.8 | 17.0 | **16.35** | 16.5 | 16.4 |
| OOD | 15.6 | **14.35** | 14.7 | 14.8 | 14.95 |

- **Rank 4 of 16 keeps the whole gain** (−0.1 [−1.0, +0.8]). Top-2 keeps most of it (+0.6, n.s.) and is the best OOD value (−0.6, n.s.). The useful encoder change is about 4-dimensional per projection.

Exact average of the two enc+dec seeds:
- Itest unseen 15.65, between the seeds (16.0, 15.3). OOD 15.0 (seeds 15.4, 15.0). All n.s.
- Truncating the average back to rank 16 gives identical WER (15.65), so the directions where the seeds disagree are irrelevant.
- Average with the encoder cut to rank 4: 16.3.
- **The consensus is no better than the seed mean.** The averaged-away seed-specific directions were neither harmful nor useful.

### Strong (SimCLR-style) augmentation (exp51)
Enc+dec with reverb, band-limiting, wider speed and noise:
- Itest unseen 17.2 vs 16.0 (+1.2 [−0.7, +3.7]).
- OOD 14.6 vs 15.4 (−0.8 [−2.2, +0.5]).
- Spontaneous unchanged.
- Neutral to slightly worse in domain, slightly better out of domain; nothing significant.

## Replications and the information-vs-plasticity question (exp59–61; 2026-10-18/19)

**Whisper seed 1 (exp61):**
- Encoder minus decoder LoRA: itest unseen −4.6 [−12.8, +1.7] (n.s.), dev unseen −8.0 [−13.5, −2.3], OOD −7.6 [−11.0, −4.4].
- itest spontaneous: better for 6/6 speakers (p = 0.03).
- Means over the two seeds: decoder 34.1 / 32.8, encoder 28.7 / 24.6 (itest unseen / OOD).
- Seed 0's itest gain (−6.4) was the larger of the two.

**Layer ablation on both enc+dec seeds (exp60):**
- Top quarter removed: +0.1 for both seeds (n.s.).
- Bottom quarter removed: +4.1 [+1.6, +7.1] and +3.7 [+1.4, +6.7].
- The encoder-only finding replicates on two other adapters.
- Layers 32–63 alone: OOD 16.2 for both seeds, worse than the full adapter. The "OOD block" result holds only for the encoder-only adapter; with a decoder LoRA, the decoder already supplies the OOD gain.

**Probes on the smaller pretrained encoders (exp59; best layer, held-out speakers, shuffled controls ≈ 0.50):**

| contrast | 300M | 7B |
|---|---|---|
| ejective vs plain | 0.938 | 0.953 |
| aspirated vs plain | 0.956 | 0.963 |
| q vs k (plain) | 0.939 | 0.969 |

- The 300M encoder encodes the contrasts almost as well as the 7B one; the largest gap is 3 points, for q/k.
- Its encoder LoRA adds about 1 point, as at 7B.
- So the absence of an encoder-adaptation gain at 300M is not a lack of phonetic information. Together with the capacity control, this points to plasticity, or to what the decoder already knows, as the explanation.

**1B probes (exp59, completed):** ejective 0.941, aspirated 0.966, q/k 0.964 (pretrained, best layer; controls ≈ 0.50). The information is present at every scale.

**CKA, pretrained vs encoder LoRA, dev frames:**
- 300M: ≥ 0.97 up to layer 19 of 24, then 0.93 → 0.87 at the top.
- 1B: ≥ 0.98 up to layer 39 of 48, then 0.92 → 0.81.
- 7B (enc+dec adapter): 0.84 from layer 16 on, 0.47 at the top.
- **Same information, different plasticity:** adaptation moves the small encoders only in their top layers, but moves the 7B encoder from its lower-middle layers on, which is where the ablations put the causal effect. In the paper, §5 "Model family and scale".
