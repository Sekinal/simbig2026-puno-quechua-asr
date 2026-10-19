# SIMBig 2026 Shared Task 1 (Puno Quechua ASR): Literature and Model Survey

Compiled 2026-10-08. Final results are due **2026-10-18 (AoE)**. The abstract is due 2026-10-11 via CMT. Hardware: 1x L40S 48 GB, 12 CPU cores, 70 GB RAM, about 7 GPU-days.

Sources were checked on 2026-10-08:
- Every Hugging Face (HF) model ID in this document was checked against `https://huggingface.co/api/models/<id>`. Any ID that returned HTTP 200 is marked ✔.
- Every arXiv ID was checked against its arXiv abstract page.
- "Expected gain" figures are **my estimates** from the cited evidence. They are not guarantees.

---

## 0. Key findings from inspecting the task, data and baseline

These findings shape the whole plan.

1. **The organizers' scripted test is closed-vocabulary.** The baseline repo's splits (`data/splits_cv26/*`, made by `make_split.py`) are **speaker-disjoint but not sentence-disjoint**. In the scripted test split, **909 of 909 utterances have a transcript that also occurs in train**. The scripted corpus has only **2,051 distinct texts (2,067 `sentence_id`s)** and **4,595 distinct words**. A sentence-level or n-gram LM built from the training transcripts is therefore close to an oracle for scripted speech, if the official hidden test is built the same way (it was released 2026-09-30).
   - Action: look at the official test (clip counts, durations, file-name scheme) to infer whether its sentences come from the known 2,071-sentence CV v27 text corpus.
   - Never use test labels or test metadata text for training or decoding. That would be cheating and would also be caught by the system-paper review.
2. **Orthography normalization matters a great deal.** The organizers released `data/shared-task/validated_sentences_curated.tsv`:
   - It respells 627 of the 2,067 sentences in Puno orthography: three vowels, `huq`/`nuqa`/`qulqi`, `n` before `p`, joined `-hina`, and so on.
   - Text is lowercased, punctuation is removed, and the charset is `a–z ñ '`.
   - Models trained on raw CV text, or on Cusco/Ayacucho-spelled external text, will lose WER on spelling alone.
   - This probably explains why the organizer paper reports omniASR-LLM-7B-v2 zero-shot at 26.6% WER / 6.2% CER on scripted speech, while the repo reports omniASR-LLM-300M-v2 at 7.07% WER.
   - Every LM corpus and every pseudo-label must be passed through a **Puno-orthography normalizer** derived from the curated file.
3. **The spontaneous "test" in the baseline split is mostly silver.** In `splits_cv26/spontaneous/test.tsv`, **531 of 546 clips are `type=silver`** (omniASR-LLM-7B auto-transcripts) and only 15 are gold. The headline "3.15% spontaneous WER with silver" is therefore largely agreement with omniASR-7B output. It is not gold accuracy.
   - If the official spontaneous test references are silver as well, then **imitating omniASR-LLM-7B-v2** wins. That means distillation on its outputs, plus including it in the ensemble.
   - If the references are gold, real accuracy wins.
   - Find out which is true: check the official test metadata, or ask the organizers (elh97@cam.ac.uk). Hedge both ways (see plan step 3).
4. **omniASR has already seen CV qxp data, and Meta has fine-tuned on it.** The Omnilingual ASR paper (Table 20/21, §5.7.5) fine-tunes language-specific CTC models on **qxp (9.9 h train)**. The resulting CERs:
   - 1.70 / 1.61 / 1.81 for 300M / 1B / 3B when seeded from an omniASR CTC checkpoint (5k steps, lr 1e-5).
   - 2.08 / 1.68 / 1.47 when seeded from the w2v2 SSL checkpoint (30k steps).
   - 4.49 / 2.94 / 2.71 for the un-fine-tuned omniASR CTC models.
   - 1.32 for omniASR-LLM-7B zero-shot.

   qxp is supported as `qxp_Latn` in `omnilingual_asr.models.wav2vec2_llama.lang_ids`. This is the strongest available prior for this exact language and corpus.
5. **The 66 h figure breaks down as follows:**
   - Scripted CV v27: 31.2 h validated, 22,727 clips, 77 speakers. About 3.7 h more is unvalidated. Its prompt text is known, so it can be verified by forced alignment.
   - Spontaneous CV SPS 5.0: 7,305 clips and 35.4 h, of which 5.2–5.5 h is validated gold. About 30 h is unvalidated and was silver-labelled by omniASR-7B.
   - Spontaneous transcripts include disfluency tags (`[ruido]`, `{aa}`, `[um]`, …). The ASR target column `transcript_for_asr` removes them.

---

## (a) Ranked plan (most likely to win, with budget)

GPU budget is about 7 × 24 h on the L40S. Rough allocations are given in brackets.

| # | Step | Why (evidence) | Expected effect | GPU |
|---|------|----------------|-----------------|-----|
| 0 | **Evaluation hygiene.** Write a normalizer identical to the organizers' (lowercase, strip punctuation, keep `'`, `ñ`; Puno respellings from the curated TSV). Score with `jiwer`, exactly as `eval/omnilingual/compute_wer.py` does. Build two dev sets: a speaker-disjoint scripted dev, and a **gold-only** spontaneous dev alongside a silver one. Profile the official test set. | Points 1–3 above. Normalization alone can be worth several WER points. | Prevents the largest avoidable losses | 0 |
| 1 | **Main acoustic model: fine-tune `omniASR_CTC_1B_v2`** (975M) with fairseq2 `ctc-finetune-recommendation.yaml`. Settings: lr 1e-5, bf16, 5k–15k steps, scripted-gold + spontaneous-gold + silver. Set the vocabulary/tokenizer to the omniASR SentencePiece tokenizer (default), or swap in a 30-char Puno vocabulary. | The omniASR paper's qxp CTC-seed FT gives 1.61 CER at 1B, beating XLS-R-style w2v2 seeds at ≤1B. The baseline XLS-R-300M+CPT reaches 0.19% CER / 1.19% WER scripted. | Scripted ≈ baseline or better (≤1.2% WER before LM). Spontaneous should beat the 13.6% gold-only baseline. | ~20 h |
| 2 | **Word n-gram LM + CTC beam search** (KenLM + pyctcdecode, or torchaudio/flashlight). Components: (i) a scripted sentence LM (4–6-gram on 2,051 curated sentences, heavy weight); (ii) a spontaneous LM (train transcripts + external Southern Quechua text normalized to Puno spelling); (iii) a char/subword fallback for OOVs (flashlight lexicon-free, or Morfessor-segmented KenLM). Tune α/β per domain on dev. Also use the spontaneous **prompt** text as hotwords when it is available. | Standard large wins for CTC in low-resource settings (MMS, HF n-gram blog, Whisper-LM up to 51% rel. in-domain). Scripted is a closed set (point 1). | Scripted: 30–80% rel. WER reduction if test sentences ∈ known set. Spontaneous: 5–15% rel. | CPU |
| 3 | **Refresh pseudo-labels for the 30 h unvalidated spontaneous audio, and verify the ~3.7 h unvalidated scripted clips.** Re-decode the spontaneous audio with an ensemble of the step-1 model + LM and omniASR-LLM-7B-v2. Keep utterances by confidence and agreement. For unvalidated scripted clips the prompt text is known: keep them when CTC forced-alignment/posterior scores are good. Retrain once or twice (noisy student). **Hedge:** if test refs are silver, weight omniASR-7B outputs more heavily (distillation). | Organizer result: silver data took spontaneous from 13.6 → 3.15 (mostly on silver refs). Noisy-student/IPL literature: 10–35% rel. | Spontaneous: largest lever. 10–30% rel. on gold refs; more on silver refs. | ~25 h |
| 4 | **Second and third diverse systems for ensembling.** (a) Reproduce the organizers' XLS-R-300M + CPT (scripts exist, about 1 day). (b) `facebook/w2v-bert-2.0` CTC (580M; HF Trainer; strong low-resource encoder). (c) Optionally, fine-tune `omniASR_LLM_1B_v2` / `omniASR_LLM_300M_v2` (an encoder-decoder with an internal LM; zero-shot already 7.07% WER scripted). | Diversity is what makes ensembling pay off. w2v-BERT 2.0 and MMS/omniASR win at small data (Nahabwe 2025). XLS-R-128 was best on Quechua in ML-SUPERB 2023. | Each system alone ≈ step 1. Mainly feeds step 5. | ~40 h |
| 5 | **Combine:** (i) checkpoint averaging (last 3–5 checkpoints) per model; (ii) frame-level CTC posterior averaging for models that share a vocabulary and frame rate (all w2v2-family models are 20 ms); (iii) ROVER / word-level voting over the systems' 1-best LM-decoded outputs, including omniASR-LLM-7B-v2 zero-shot for spontaneous speech. | Model soups / SWA. ROVER typically gives 5–15% rel. over the best single system. | 5–15% rel. | ~5 h |
| 6 | **Continued pretraining (CPT) on all qxp audio** (all 66 h of train audio; adding the unlabelled test audio only if the rules explicitly permit it). Apply it to XLS-R-300M (already done by the organizers) and optionally to `omniASR_W2V_1B` with the organizers' `temp_sampled_audio_pretraining` task. | Organizers: 42% rel. scripted gain for XLS-R. Robust wav2vec2, DeHaven & Billa, Nowakowski: CPT is the most effective adaptation for new languages. Expected to be smaller for omniASR, which already saw qxp. | 0–20% rel. depending on base | ~20 h |
| 7 | **Augmentation:** speed perturbation 0.9/1.0/1.1, SpecAugment (masking is already in the w2v2/omniASR FT configs), light MUSAN noise/RIR for spontaneous. Optionally use extra in-variety supervised speech: the **Quechua Collao emotion corpus** (15 h read speech, 7 speakers from Puno/Cusco/Apurímac, CC BY 4.0), after re-spelling to Puno orthography. | Standard. Most useful for robustness and spontaneous speech. | 0–10% rel. | ~10 h |
| 8 | Low priority or likely not worth it: LLM generative error correction (open LLMs barely edit low-resource text; GPT-class models help but break reproducibility and risk over-correction); test-time adaptation (SUTA); TTS augmentation (MMS-TTS has `quz`/`quy` but no `qxp`); Whisper fine-tuning (no Quechua prior; organizers' Whisper-base reached 17.1% on spontaneous). | See §(c). | ≤ few % rel., risky | — |

**Suggested schedule (Oct 8–18):**
- D0–1: step 0, LM tooling, launch step 1.
- D1–2: XLS-R-CPT repro, w2v-BERT run.
- D2–3: step 2 tuning on dev.
- D3–5: step 3 pseudo-label iteration (retrain the omniCTC-1B).
- D5–6: step 5 ensemble, final α/β/ROVER tuning on dev.
- D7: inference on test, paper and video.
- Keep 1 day of slack. Submit a safe system early, for example the baseline repro + LM.

**Rules to confirm with the organizers:** whether external data or pretrained models are allowed (the baseline itself uses external pretrained models and omniASR silver labels), whether test audio may be used for unsupervised CPT/TTA, and whether the spontaneous references are gold or silver.

---

## (b) Model candidates

✔ = HF ID verified (HTTP 200) on 2026-10-08.

| Model | HF ID | Params | Quechua coverage | Fine-tuning path | Fits 1×L40S? | Verdict |
|---|---|---|---|---|---|---|
| **omniASR CTC 1B v2** | `facebook/omniASR-CTC-1B` ✔. The v2 weights are at `dl.fbaipublicfiles.com/mms/omniASR-CTC-1B-v2.pt` and download automatically by card name `omniASR_CTC_1B_v2`. | 975M | **qxp_Latn plus 40 other Quechua codes.** Meta FT'd qxp: 1.61 CER. | fairseq2 recipe `workflows/recipes/wav2vec2/asr` (`ctc-finetune*.yaml`). Community HF/transformers ports exist (`aadel4/omniASR-CTC-1B-v2`, `bezzam/omniasr-ctc-300m-v2`); the transformers PR #43265 is still OPEN. | Yes (full FT bf16) | **Primary** |
| omniASR CTC 300M / 3B / 7B v2 | `facebook/omniASR-CTC-300M` ✔, `-3B` ✔, `-7B` ✔ | 325M / 3.08B / 6.5B | same | same | 300M yes. 3B is tight (needs 8-bit Adam or a partial freeze). 7B is no for full FT. | 300M for fast iteration; 3B if time allows |
| **omniASR LLM 300M/1B/3B/7B v2** | `facebook/omniASR-LLM-300M` ✔, `-1B` ✔, `-3B` ✔, `-7B` ✔ (v2 by card name `omniASR_LLM_*_v2`) | 1.63B / 2.28B / 4.38B / 7.80B | qxp (lang-conditioned); 7B zero-shot 1.32 CER on Meta's qxp test | `llm-finetune.yaml` (lr 5e-5, FSDP, 20k steps). No LoRA in the repo. The `Unlimited` variants cannot be fine-tuned. | 300M/1B yes. 7B inference only (~17 GB). | 7B-v2 = pseudo-labeller and ROVER member; 1B-v2 FT = ensemble member |
| omniASR LLM 7B ZS | card `omniASR_LLM_7B_ZS` | 7.81B | in-context examples | none (few-shot prompting) | inference | Not needed: we have plenty of data |
| omniASR W2V SSL 300M/1B/3B/7B | `facebook/omniASR-W2V-300M` ✔, `-1B` ✔, `-7B` ✔ | 317M / 966M / 3B / 7B | pretrained on 4.3M h incl. qxp | CPT then CTC FT (`ctc-from-encoder.yaml`) | 300M/1B yes | For CPT experiments |
| MMS-1B-all | `facebook/mms-1b-all` ✔ | ~1B + 2M per adapter | adapters `quz`, `quy` and others; **no qxp** | HF adapter FT (`blog/mms_adapters`) | Yes | Superseded by omniASR. Organizers: 35% WER zero-shot. `robertp408/mms-1b-all-qxp` reached 19.5% WER on a small FT. |
| MMS-1B / MMS-300M (SSL) | `facebook/mms-1b` ✔, `facebook/mms-300m` ✔ | 1B / 300M | SSL incl. Quechua | CTC FT | Yes | Possible diversity member |
| XLS-R 300M / 1B / 2B | `facebook/wav2vec2-xls-r-300m` ✔, `-1b` ✔, `-2b` ✔ | 315M / 965M / 2.2B | SSL (128 langs, incl. Quechua via BABEL/CV) | HF or fairseq CTC. The organizers' CPT recipe is in fairseq. | 300M/1B yes; 2B tight | 300M+CPT = organizer baseline (reproduce as member) |
| **w2v-BERT 2.0** | `facebook/w2v-bert-2.0` ✔ | 580M | SSL on 4.5M h, 143+ langs | HF `Wav2Vec2BertForCTC` (blog `fine-tune-w2v2-bert`) | Yes | **Strong diversity member** |
| XEUS | `espnet/xeus` ✔ | ~577M | SSL on 1M h, 4,057 langs | ESPnet | Yes | Optional. SOTA on ML-SUPERB, but ESPnet-only tooling. |
| mHuBERT-147 | `utter-project/mHuBERT-147` ✔ | 95M | 147 langs | HF | Yes | Too small to win |
| SeamlessM4T v2 large | `facebook/seamless-m4t-v2-large` ✔ | 2.3B | no Quechua ASR target | seq2seq FT | tight | TalTech used it in ML-SUPERB 2.0. Its encoder = w2v-BERT 2.0. Use that instead. |
| Whisper large-v3 / turbo | `openai/whisper-large-v3` ✔, `openai/whisper-large-v3-turbo` ✔ | 1.55B / 809M | **No Quechua** (detected as other languages, 147% WER on Quechua TTS) | HF Seq2Seq FT or LoRA; Whisper-LM for n-gram fusion | Yes | Weak prior. QUESPA (IWSLT 2026) FT'd v3 on 48 h. Organizers' Whisper-base reached 3.8% / 17.1%. Low priority. |
| OWSM v4 | `espnet/owsm_v4_medium_1B` ✔, `espnet/owsm_ctc_v4_1B` ✔ | 1B | YODAS-based, many langs | ESPnet FT | Yes | Low priority |
| NVIDIA Canary-1B-v2 / Parakeet-TDT-0.6B-v3 | `nvidia/canary-1b-v2` ✔, `nvidia/parakeet-tdt-0.6b-v3` ✔ | 978M / 600M | 25 European langs only | NeMo (new tokenizer needed) | Yes | Not recommended |
| Qwen3-ASR 1.7B / 0.6B | `Qwen/Qwen3-ASR-1.7B` ✔, `-0.6B` ✔ | 1.7B / 0.6B | 30 langs + 22 Chinese dialects; no Quechua | LLM-ASR FT | Yes | Not recommended |
| Voxtral Mini / Small | `mistralai/Voxtral-Mini-3B-2507` ✔, `mistralai/Voxtral-Small-24B-2507` ✔, `mistralai/Voxtral-Mini-4B-Realtime-2602` ✔ | ~4.7B / 24B | 8–13 major langs | LoRA | Mini yes | Not recommended |
| IBM Granite Speech | `ibm-granite/granite-speech-3.3-8b` ✔, `-3.3-2b` ✔, `ibm-granite/granite-speech-4.1-2b` (listed) | 2–8B | EN/FR/DE/ES/PT/JA | LoRA | Yes | Not recommended |
| Other 2025–26 releases checked | `microsoft/VibeVoice-ASR` (50+ langs), `CohereLabs/cohere-transcribe-03-2026`, `nvidia/nemotron-3.5-asr-streaming-0.6b` (40 locales), `FunAudioLLM/Fun-ASR-Nano-2512`, `Cactus-Compute/whistle` (7 langs) | — | none cover Quechua | — | — | Not relevant. **omniASR is the only 2025–26 open model with native qxp.** |

Existing qxp checkpoints on HF (useful to sanity-check or seed):
- `QuechuaBase/xls-r-cpt-qxp*` (organizer CPT/FT checkpoints, cv26 variants)
- `QuechuaBase/whisper-base-qxp-finetuned`
- `QuechuaBase/wav2vec2-base-qxp-finetuned`
- `ctaguchi/ssc-qxp-mms-*` (MMS adapters from the 2025 Mozilla spontaneous task; about 50% WER)
- `robertp408/mms-1b-all-qxp`

---

## (c) Technique catalog (with citations)

### C1. Backbones and adaptation
- **Omnilingual ASR** (Omnilingual ASR team / Keren et al., 2025, arXiv:2511.09690, https://arxiv.org/abs/2511.09690; code https://github.com/facebookresearch/omnilingual-asr). It covers 1,600+ languages. SSL is trained on 4.3M h, and CV v22 is in the training data. §5.7.5 gives qxp-specific FT results (CER 1.61 at 1B CTC-seed). The recommended FT recipe is lr 1e-5, 5k steps when seeded from a CTC checkpoint, and 30k steps when seeded from SSL. *Why it matters:* it is the best prior for exactly this language. Expected to match or beat the XLS-R+CPT baseline before any LM is added.
- **MMS** (Pratap et al., 2023, arXiv:2305.13516). This is the 1B-all model with per-language adapters (quz, quy). Its decoding uses n-gram LMs trained on Common Crawl. *Why:* it is a fallback/diversity encoder and shows the size of the n-gram LM gain. It is superseded by omniASR.
- **XLS-R** (Babu et al., 2021, arXiv:2111.09296). *Why:* the organizer baseline. ML-SUPERB 2023 found XLS-R-128 the best SSL encoder on Quechua (Shi et al., 2023, arXiv:2310.05513).
- **w2v-BERT 2.0** (Seamless Communication, 2023, arXiv:2308.11596 / 2312.05187; w2v-BERT: Chung et al., 2021, arXiv:2108.06209; HF blog https://huggingface.co/blog/fine-tune-w2v2-bert). *Why:* strong at 1–30 h (Nahabwe et al., 2025, arXiv:2512.10968). Its architecture (Conformer) differs from w2v2/omniASR, which makes it a good ensemble member.
- **XEUS** (Chen et al., 2024, arXiv:2407.00837). It beats MMS-1B and w2v-BERT 2.0 on ML-SUPERB. *Why:* an optional extra encoder.
- **Whisper** (Radford et al., 2022, arXiv:2212.04356). It has no Quechua. *Why:* low priority. A 2026 Frontiers study found Whisper-v3 mis-identifies Quechua as 6 other languages, at 147% WER (https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1909076/full).
- **Continued pretraining (CPT):** Robust wav2vec 2.0 (Hsu et al., 2021, arXiv:2104.01027); DeHaven & Billa (2022, arXiv:2207.00659), where CPT beats or complements semi-supervised training in low-resource settings; Nowakowski et al. (2023, arXiv:2301.07295), where CPT was the most effective adaptation of XLSR-53 to Ainu; the organizers, with 42% relative gain on scripted speech (arXiv:2605.28253). *Expected:* large for XLS-R, smaller for omniASR (which already saw qxp).
- **LoRA / parameter-efficient FT** (Hu et al., 2021, arXiv:2106.09685); depth-aware adaptation for low-resource multilingual ASR (arXiv:2602.01008). *Why:* only needed for 3B/7B models on 48 GB. Full FT of ≤1B is better here.
- **Data-size findings:** Liang & Levow (2025, FieldMatters, arXiv:2506.17459) found MMS best at <1 h and XLS-R at parity above 1 h. Nahabwe et al. (2025, arXiv:2512.10968) found that n-gram LMs help when the text and acoustic domains match. *Why:* we have about 60 h, so all encoders are viable and the LM text must match the domain.

### C2. Language-model decoding (highest-ROI inference trick)
- **KenLM** (Heafield, 2011, WMT, https://aclanthology.org/W11-2123/). Use it to build modified-Kneser-Ney ARPA/binary LMs. Use `--discount_fallback` for tiny corpora.
- **pyctcdecode** (Kensho, https://github.com/kensho-technologies/pyctcdecode). A word-level KenLM CTC beam search with **hotword boosting**. HF guide: "Boosting Wav2Vec2 with n-grams in Transformers" (von Platen, 2022, https://huggingface.co/blog/wav2vec2-with-ngram). *Why:* a drop-in for HF CTC models. Hotwords can carry the spontaneous prompt's words.
- **Flashlight / torchaudio CTC decoder** (Kahn et al., 2022, arXiv:2201.12465; Yang et al., 2022, arXiv:2110.15018). Supports a lexicon-constrained or **lexicon-free** decoder with token-level KenLM. *Why:* a char/subword LM avoids OOVs in an agglutinative language, where 77% of words are hapaxes.
- **Lexicon-free char LMs:** "Who Needs Words?" (Likhomanenko et al., 2019, arXiv:1904.04479). Character LMs nearly match word LMs and handle OOVs.
- **Subword (morph) LMs for agglutinative languages:** Hirsimäki et al. (2006, Computer Speech & Language 20(4), "Unlimited vocabulary speech recognition with morph language models applied to Finnish"); Smit, Virpioja & Kurimo (2021, Computer Speech & Language 66:101158, "Advances in subword-based HMM-DNN speech recognition across languages"); **Morfessor 2.0** (Smit et al., 2014, EACL demos, https://aclanthology.org/E14-2006/). *Why:* Quechua has about 3 morphemes per word. Morph or BPE n-grams (with word-boundary markers) generalize to unseen inflections in spontaneous speech. *Expected:* a few % rel. over word LMs on OOV-heavy spontaneous speech.
- **Quechua morphology tools:** the SQUOIA toolkit / Rios (2015, PhD thesis, U. Zurich; https://github.com/a-rios/squoia) has a Cusco-Quechua finite-state analyzer. *Why:* it can supply linguistically informed segmentation for the LM and for Puno re-spelling rules.
- **LM fusion for seq2seq models:** Whisper-LM (de Zuazo et al., 2025, arXiv:2503.23542; code https://github.com/hitz-zentroa/whisper-lm). It reports up to 51% relative WER reduction in-distribution with n-grams for low-resource languages. *Why:* it gives the scale of gains when LM text matches the test domain, which is exactly the scripted case here.
- **Closed-set sentence snapping** (data-centric). If a scripted hypothesis is within a small edit distance of one of the 2,051 known sentences, replace it with that sentence. Alternatively, put a very high weight on a sentence-level LM. This works because Common Voice re-records the same prompts across speakers (Ardila et al., 2020, arXiv:1912.06670). *Expected:* scripted WER toward ~0 if test sentences ∈ known set. Gate it with a threshold tuned on dev, so that novel sentences are not damaged.

### C3. Pseudo-labelling / self-training
- **Noisy Student for ASR** (Park et al., 2020, arXiv:2005.09629); **Iterative Pseudo-Labeling** (Xu et al., 2020, arXiv:2005.09267); **slimIPL** (Likhomanenko et al., 2021, arXiv:2010.11524); **Momentum PL** (Higuchi et al., 2021, arXiv:2106.08922); **Self-training for E2E ASR** (Kahn et al., 2020, arXiv:1909.09116), which covers confidence filtering and ensembles of pseudo-labels; **Self-training + pre-training are complementary** (Xu et al., 2021, arXiv:2010.11430); **multilingual PL** (Lugosch et al., 2022, arXiv:2111.00161); **cross-lingual transfer + IPL**, which reports 35% error reduction (arXiv:2305.13652); **iterative confidence-filtered self-training**, which reports 14.9% relative (arXiv:2308.05269). *Why:* 30 h of untranscribed spontaneous speech is 5–6× the gold spontaneous data. The organizers' silver run is the largest single gain in their paper (77% relative on spontaneous). *Expected:* 10–30% relative on gold spontaneous refs over a gold-only model, and more if refs are silver.
- **Practical recipe:**
  1. Label with an ensemble: our fine-tuned CTC model + LM, plus omniASR-LLM-7B-v2.
  2. Keep clips where the systems agree (pairwise WER < x) or where CTC confidence is high.
  3. Retrain from the CTC checkpoint, adding SpecAugment and speed-perturbation noise.
  4. Iterate once or twice.

### C4. Augmentation
- **SpecAugment** (Park et al., 2019, arXiv:1904.08779). Already present in w2v2/omniASR fine-tuning as time/channel masking. Tune `mask_prob`.
- **Speed perturbation** (Ko et al., 2015, Interspeech, "Audio augmentation for speech recognition"). Typically 3–8% relative in hybrid and E2E ASR.
- **MUSAN noise** (Snyder et al., 2015, arXiv:1510.08484) and RIR. Mainly helps spontaneous and OOD speech.
- **TTS augmentation:** Rossenbach et al. (2020, arXiv:1912.09257); text generation + TTS (Huang et al., 2023, arXiv:2305.16333); "Frustratingly Easy Data Augmentation for Low-Resource ASR" (Ibaraki et al., 2025, arXiv:2509.15373); Quechua-specific: Zevallos et al. (2022, Interspeech, "Data Augmentation for Low-Resource Quechua ASR Improvement", DOI 10.21437/Interspeech.2022-770), which reports 8.73% relative WER reduction with synthetic text + TTS on Siminchik. *Why:* low priority here, because real data is plentiful and there is no qxp TTS (MMS-TTS has `facebook/mms-tts-quz`/`-quy`). Could help the LM side via text generation.
- Data augmentation for ML-SUPERB 2.0 (Wang et al., 2025, arXiv:2505.24200).

### C5. Ensembling, averaging, MBR
- **ROVER** (Fiscus, 1997, IEEE ASRU, "A post-processing system to yield reduced word error rates: Recognizer Output Voting Error Reduction"). Typically 5–15% relative over the best system when systems are diverse. Implementations: SCTK `rover`, or a Python alignment-and-vote script.
- **Weight / checkpoint averaging:** Model soups (Wortsman et al., 2022, arXiv:2203.05482) and SWA (Izmailov et al., 2018, arXiv:1803.05407). A free gain of 0–5% relative and more stable results.
- **CTC posterior (logit) averaging** across models with the same output units and frame rate. Alternatively, run a joint beam search that sums log-probs.
- **MBR decoding** (Goel & Byrne, 2000, Computer Speech & Language 14(2), "Minimum Bayes-risk automatic speech recognition"). MBR over the n-best of each system and the union directly targets WER. It is a cheap complement to ROVER.
- **System selection:** TalTech won ML-SUPERB 2.0 by picking per language among SeamlessM4Tv2, MMS-1B-all and MMS-zeroshot on dev (Alumäe et al., 2025, arXiv:2506.01458). *Why:* choose systems per domain (scripted vs spontaneous) on dev.

### C6. Post-correction and test-time methods (low priority)
- **LLM generative error correction (GER):** HyPoradise (Chen et al., 2023, NeurIPS, arXiv:2309.15701); Whispering-LLaMA (Radhakrishnan et al., 2023, EMNLP, arXiv:2310.06434); task-activating prompting (Yang et al., 2023, arXiv:2309.15649); Denoising LM (Gu et al., 2024, arXiv:2405.15216). On low-resource languages, Yun et al. (2026, arXiv:2605.19711) found for Frisian that GPT-5.1 GER cut WER below the 5-best oracle, while **open Qwen3 models left 97–99% of sentences unchanged**. *Why it is low priority:* no capable open Quechua LLM exists, and proprietary APIs hurt reproducibility, which is a stated task goal. There is a real risk of over-correcting to Cusco/Ayacucho spellings. If tried, use it on spontaneous n-best only, with a constrained edit budget.
- **Test-time adaptation:** SUTA (Lin et al., 2022, Interspeech, arXiv:2203.14222), which applies entropy minimization per utterance. *Why:* small gains (about 1–5% relative) under domain shift. CV test speakers are in-domain, so the expected gain is small.
- **Multimodal in-context learning for low-resource ASR** (arXiv:2601.05707), and the omniASR-7B-ZS context-example model. Not needed with 60 h of in-language data.
- **Task arithmetic with support languages** (Rafkin, DeGenaro & Yang, 2026, arXiv:2601.07038; code https://github.com/ddegenaro/mozilla-asr-challenge). This was a 2025 Mozilla spontaneous-task submission that merged Whisper LoRA task vectors from related languages. It reports up to 10% relative. Low priority given omniASR's native qxp.

---

## (d) Quechua text and speech resources

All external text must be **re-spelled to Puno orthography** (`validated_sentences_curated.tsv` conventions: 3 vowels, `huq`, `nuqa`, `n`+`p`, `-hina` joining, lowercase, no punctuation) before it goes into an LM.

### Text (for LMs)
| Resource | Variety | Size (approx.) | License | URL |
|---|---|---|---|---|
| **Curated CV qxp scripted sentences** (organizers) | Puno | 2,067 sentences / 8,272 tokens | CC0 (CV) | `data/shared-task/validated_sentences_curated.tsv` in https://github.com/QuechuaBase/asr-puno-quechua |
| CV qxp spontaneous `transcript_for_asr` (gold + silver train) | Puno | ~7k utterances | CC0 | Mozilla Data Collective, SPS 5.0 (ID `cmu60wldk00jcnq07eaxdrdl5`) |
| CV qxp text corpus (`validated_sentences.tsv` in CV release; 2,071 sentences) | Puno | small | CC0 | MDC Scripted 27.0 (ID `cmu62ctgp00o9nq07canzmu07`) |
| Organizer OOD radio transcripts | Puno | 110 lines | Apache-2.0 repo | `data/additional_data/additional_data_qxp.tsv` |
| **Llamacha/monolingual-quechua-iic** (QuBERT corpus; Zevallos et al., 2022, DeepLo) | Southern (quz/quy) | 175k lines / ~11.6 MB | Apache-2.0 | https://huggingface.co/datasets/Llamacha/monolingual-quechua-iic |
| Quechua Wikipedia (`wikimedia/wikipedia`, `20231101.qu`) | Southern, mostly Collao-like spelling | 24k articles / 7.7 MB parquet | CC BY-SA 4.0 | https://huggingface.co/datasets/wikimedia/wikipedia |
| FineWeb-2 (Penedo et al., 2025, arXiv:2506.20920) `quz_Latn`, `quy_Latn`, `quh_Latn` | Cusco / Ayacucho / S. Bolivian | 1.5k / 2.9k / 0.1k docs (2.7 / 3.8 / 0.5 MB) | ODC-By | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| GlotCC-V1 (Kargaran et al., 2024, arXiv:2410.23825) `quz-Latn`, `quy-Latn` | | ~0.2 MB each | CC0 / CommonCrawl terms | https://huggingface.co/datasets/cis-lmu/GlotCC-V1 |
| MADLAD-400 (Kudugunta et al., 2023, arXiv:2309.04662) `qu` | mixed | small | ODC-By | https://huggingface.co/datasets/allenai/MADLAD-400 |
| AmericasNLP 2021 Quechua–Spanish (JW300 quy, MINEDU dict, misc dict; Mager et al., 2021) | Ayacucho/Cusco | ~125k sentence pairs | JW300 is restricted (withdrawn from OPUS for copyright); MINEDU is public | https://github.com/AmericasNLP/americasnlp2021/tree/main/data/quechua-spanish |
| IWSLT 2026 QUE–SPA data: JW300 + Hinantin text in `additional_mt_text`; ~8 h synthetic-translation set | quz/quy | MT text | CC BY-NC-ND 3.0 (audio) | https://github.com/johneortega/IWSLT2026_Quechua_data |
| FLORES-200 `quy_Latn` | Ayacucho | ~2k sentences | CC BY-SA 4.0 | https://huggingface.co/datasets/facebook/flores |
| Cusco Quechua Bible text (e.g., `pollitoconpapass/quz-bible-v2`) | Cusco | ~30k verses | unclear (Bible Society copyright); eBible has no quz/quy/qxp | https://huggingface.co/datasets/pollitoconpapass/quz-bible-v2 |
| Quechua treebank / SQUOIA (Rios) | Cusco | small | research | https://github.com/a-rios/squoia |

**Expected LM value:**
- In-domain CV text dominates for scripted speech.
- For spontaneous speech, interpolate the CV spontaneous transcripts (highest weight) with QuBERT + Wikipedia + FineWeb-2 quz, after re-spelling. Tune the interpolation weights on dev perplexity.

### Extra speech (check the rules before using)
| Resource | Variety | Size | License | URL |
|---|---|---|---|---|
| **Quechua Collao emotion corpus** (Paccotacya-Yanque et al., 2022, *Scientific Data* 9:778) | Collao: speakers from Puno, Cusco and Apurímac | 15.25 h, 12,420 segments of 2,070 read items, 7 speakers | CC BY 4.0 | https://doi.org/10.6084/m9.figshare.20292516 |
| Siminchik / IWSLT unconstrained ASR set (Cardenas et al., 2018) | Southern (radio, spontaneous) | ~48 h transcribed (of 97 h) | CC BY-NC-ND 3.0 (training use legally grey) | https://github.com/johneortega/IWSLT2026_Quechua_data |
| Huqariq (Zevallos et al., 2022, LREC, arXiv:2207.05498) | several Peruvian languages incl. Quechua | 220 h total | check | arXiv |
| omnilingual-asr-corpus (Meta) | Quechua configs only `qug`, `qum`, `quv` (not Southern) | — | CC BY 4.0 | https://huggingface.co/datasets/facebook/omnilingual-asr-corpus |
| ⚠ `Epiph0nE/quechua-puno-tts` | — | 22,727 rows = a re-host of the CV qxp scripted set | violates MDC "no re-share" terms | Do not use (it is also not new data) |

---

## (e) Quechua and shared-task literature, and public results

| Work | Result relevant to us |
|---|---|
| **Huaman, Gamarra Lafuente, Cordova, Korhonen (2026)**, "Building Community-Centred NLP Resources for Puno Quechua", AmericasNLP 2026 (https://aclanthology.org/2026.americasnlp-6.13/; arXiv:2605.28253) | The organizers' benchmark. Zero-shot: omniASR-LLM-7B-v2 26.6/6.2 (scripted WER/CER), 11.1/1.9 (spontaneous), 23.7/4.1 (OOD); MMS-1b-all 35.0/5.3. Fine-tuned: XLS-R+CPT with V+S 2.11/0.30 scripted and 3.15/0.41 spontaneous, but 27.4/4.55 OOD; Whisper-base 3.81 / 17.1; wav2vec2-base 7.37 / 13.9. Silver labels came from omniASR_LLM_7B. 46-char vocab. |
| **Huaman, Huaman, Huaman, Quispe (2025)**, "Quechua Speech Datasets in Common Voice: The Case of Puno Quechua", SIMBig 2025, Springer CCIS (https://link.springer.com/chapter/10.1007/978-3-032-20322-9_13; arXiv:2510.13871) | Dataset creation: sentence sources were bilingual-education materials, and the paper describes the Pontoon/CV workflow. |
| Organizer baseline repo (https://github.com/QuechuaBase/asr-puno-quechua) | `results/summary.csv`: ft_cpt_validated 1.19/0.19 (scripted), 13.61/1.73 (spontaneous); ft_cpt_silver 2.11/0.30, 3.15/0.41; ft_xlsr_validated 2.06 / 13.58; ft_xlsr_silver 4.36 / 6.68; omniASR LLM-300M-v2 7.07 / 11.98; omniASR CTC-300M-v2 24.47 / 20.22. |
| SIMBig 2026 shared-task page (https://simbig.org/SIMBig2026/en/sharedtask.html); CFP (https://linguistlist.org/issues/37/2673/) | Metrics are WER/CER. Results are submitted as a TSV. There is a 4–8 page LNCS paper and a 15-min video. Prize: USD 100 + registration + CCIS publication. **No public leaderboard found.** |
| Omnilingual ASR (arXiv:2511.09690), §5.7.5, Tables 20–21 | qxp single-language CTC FT: CER 1.70/1.61/1.81 (CTC seed, 300M/1B/3B), 2.08/1.68/1.47 (W2V seed); base omniCTC 4.49/2.94/2.71; omniLLM-7B 1.32. These are the only other public qxp numbers, on Meta's own split. |
| Mozilla Common Voice Spontaneous Speech ASR Shared Task 2025 (https://community.mozilladatacollective.com/shared-task-mozilla-common-voice-spontaneous-speech-asr/) | qxp was one of 5 **unseen** languages (Task 4, no training data). No findings paper or leaderboard was located. Example entry: Rafkin et al. (2026, arXiv:2601.07038). HF models `ctaguchi/ssc-qxp-mms-*` report about 50% WER / 9% CER. |
| `robertp408/mms-1b-all-qxp` (Mozilla.ai blueprint) | MMS-1b-all FT on 1,502 qxp clips: 19.5% WER (normalized) / 2.7% CER on 562 clips. |
| Cardenas, Zevallos, Baquerizo, Camacho (2018), "Siminchik: A Speech Corpus for Preservation of Southern Quechua", LREC 2018 CCURL workshop (http://lrec-conf.org/workshops/lrec2018/W14/pdf/4_W14.pdf) | 97 h of Southern Quechua radio. First ASR/LM baselines. |
| Zevallos, Bel, Cámbara, Farrús, Luque (2022), "Data Augmentation for Low-Resource Quechua ASR Improvement", Interspeech 2022 (https://www.isca-archive.org/interspeech_2022/zevallos22_interspeech.html) | Synthetic text + TTS: 22.75% WER, 8.73% relative gain (wav2letter++). |
| Zevallos, Camacho, Melgarejo (2022), "Huqariq", LREC 2022 (arXiv:2207.05498) | Peruvian multi-language speech corpus. |
| Zevallos et al. (2022), "Introducing QuBERT: A Large Monolingual Corpus and BERT Model for Southern Quechua", DeepLo (https://aclanthology.org/2022.deeplo-1.1/) | Source of the monolingual text corpus. |
| Ebrahimi et al. (2023), "Findings of the Second AmericasNLP Competition on Speech-to-Text Translation", PMLR 220:217–232 (NeurIPS 2022 comp.) (https://proceedings.mlr.press/v220/ebrahimi23a.html) | ASR + ST for 5 languages including Quechua. XLS-R fine-tuning dominated. |
| Romero, Gómez, Torre (2024), "Automatic Speech Recognition Advancements for Indigenous Languages of the Americas", *Applied Sciences* 14(15):6497 (arXiv:2404.08368; model `ivangtorre/wav2vec2-xlsr-300m-quechua` ✔-listed) | XLS-R 300M/1B with Bayesian HPO. Quechua CER 12.14 (Siminchik-type data). |
| Ortega, Zevallos et al. (2026), "Team QUESPA System Submission for the IWSLT 2026 Dialectal and Low-resource Speech Translation Task", IWSLT 2026 (https://aclanthology.org/2026.iwslt-1.5.pdf) | Whisper-large-v3 FT on 48 h Quechua ASR data (ESPnet). Releases a Collao corpus and text. Cites ML-SUPERB 2023 finding that XLS-R-128 is best on Quechua. |
| Paccotacya-Yanque et al. (2022), *Scientific Data* (https://pmc.ncbi.nlm.nih.gov/articles/PMC9789950/) | 15 h Collao read speech (CC BY 4.0). Speakers include Puno natives. |
| Shi et al. (2023), "Findings of the 2023 ML-SUPERB Challenge" (arXiv:2310.05513) | Quechua among the evaluated languages; XLS-R-128 strongest SSL on it. |
| Chen et al. (2025), "The ML-SUPERB 2.0 Challenge" (arXiv:2509.07139); Alumäe et al. (2025, arXiv:2506.01458) | Multilingual low-resource ASR; per-language system selection with MMS / SeamlessM4T. |

---

## (f) Benchmark & leaderboard check (Oct 2026)

Checked 2026-10-08. The question was whether any open model released through Oct 2026 is a stronger base than omniASR for qxp. HF IDs were re-checked with `https://huggingface.co/api/models/<id>` (✔ = HTTP 200; the license is the one the API reports). Numbers marked "my aggregation" were computed by me from the leaderboard's raw CSVs.

### f.1 What each benchmark says

**1. HF Open ASR Leaderboard** (https://huggingface.co/spaces/hf-audio/open_asr_leaderboard; paper arXiv:2510.06961, https://arxiv.org/abs/2510.06961). The current version is "02-10-2026", per the space's `init.py`.
- **English track** (raw CSV https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results, `english_short_latest.csv`). Best open models by average WER:
  - `microsoft/Phi-4-multimodal-instruct`: 5.02 (MIT, 8 langs)
  - `ibm-granite/granite-speech-3.3-8b`: 5.26 (Apache-2.0)
  - `Qwen/Qwen3-ASR-1.7B-hf`: 5.75 (52 langs)
  - `CohereLabs/cohere-transcribe-03-2026`: 6.26 (14 langs)
  - `nvidia/canary-qwen-2.5b`: 6.37
  - `openai/whisper-large-v3`: 7.84
  - `facebook/omniASR-LLM-7B-v2`: 8.78
  - `facebook/omniASR-CTC-7B-v2`: 12.01
- **Multilingual track** (per-language CSVs at https://huggingface.co/datasets/hf-audio/multilingual_evals: es, pt, fr, de, it, nl, hi, hy; FLEURS, MCV and MLS WER).
  - Best open models by mean WER over es/pt/fr/de/it/nl (my aggregation): `mistralai/Voxtral-Small-24B-2507` 3.74, `CohereLabs/cohere-transcribe-03-2026` 3.84, `openai/whisper-large-v3` 4.57, `nvidia/canary-1b-v2` 4.97, `Qwen/Qwen3-ASR-1.7B-hf` 5.52, omniASR-LLM-7B-v2 5.69, omniASR-CTC-1B-v2 8.26.
  - **Armenian (`multilingual_hy.csv`) is the only track that resembles a long-tail language.** Results as FLEURS / MCV WER:
    - omniASR-LLM-7B-v2: **3.61 / 10.04**
    - omniASR-CTC-3B-v2: 6.49 / 12.0
    - omniASR-CTC-1B-v2: 9.55 / 16.83
    - Gemini-3.5-transcribe: 4.79 / 11.74
    - gpt-transcribe: 7.23 / 12.21
    - mms-1b-all: 13.43 / 26.57
    - whisper-large-v3: 43.03 / 57.04
  - Only two systems beat omniASR-LLM-7B-v2 on Armenian. ElevenLabs Scribe v2 (proprietary) scores 3.22 / 7.6. A **dedicated monolingual fine-tuned** `nvidia/stt_hy_fastconformer_hybrid_large_pc` scores 6.8 / **4.94** on MCV. In other words, the large multilingual models rank by Common Voice coverage, and only in-language fine-tuning beat them in-domain.
  - Takeaway: the leaders of this board are tuned for high-resource European languages. **None of the top open models (Phi-4-MM, Granite, Qwen3-ASR, Cohere, Canary/Parakeet, Voxtral, Whisper) covers Quechua.**

**2. ML-SUPERB / ML-SUPERB 2.0**
- **XEUS** (arXiv:2407.00837, Tables 3–4, https://arxiv.org/pdf/2407.00837v2). These numbers come from a frozen encoder with a light CTC head, not full fine-tuning. On ML-SUPERB monolingual ASR (10 min / 1 h CER):
  - XEUS: 30.3 / 25.1
  - MMS-1B: 33.3 / 25.7
  - w2v-BERT 2.0 v2: 41.0 / 29.2
  - XLS-R 1B: 40.5 / 30.9

  On multilingual ASR, XEUS and MMS-1B split the two settings: XEUS 21.1 / 20.1, MMS-1B 21.3 / 18.1. FLEURS CER: w2v-BERT 2.0 v2 8.7, XEUS 8.9, MMS-1B 9.2, XLS-R 1B 9.6.
- **ML-SUPERB 2.0 Challenge** (Interspeech 2025; arXiv:2509.07139, Table 2, https://arxiv.org/pdf/2509.07139).
  - Frozen-SSL baselines, standard CER: XEUS 40.2 (best SSL), w2v-BERT 46.7, XLSR-53 51.9, MMS-1B 57.1. Whisper-v3 zero-shot scored 43.2.
  - The winner (TalTech, arXiv:2506.01458) selected among SeamlessM4T / MMS systems per language.
  - **No ML-SUPERB 2026 / 3.0 challenge exists.** Interspeech 2026 instead runs MLC-SLM-2 (multilingual conversational speech, 21 language/accent categories) and an Audio-Encoder Capability challenge, neither relevant here (https://dataoceanai.github.io/Interspeech2026-Audio-Encoder-Challenge/).
  - omniASR W2V encoders have **never been entered in ML-SUPERB**, so no head-to-head SSL probe exists.

**3. FLEURS / FLEURS-R / Common Voice**
- **FLEURS has no Quechua.** The 103 configs of `google/fleurs` were checked via datasets-server: there is no `qu*` config. FLEURS-R is a restoration of the same set.
- **The only public Quechua-family CERs from a general model** are in Meta's per-language table for omniASR-LLM-7B, on Meta's internal mixed eval sets (https://github.com/facebookresearch/omnilingual-asr/blob/main/per_language_results_table_7B_llm_asr.csv):
  - **qxp_Latn: 1.4 CER (9.9 h train)**
  - quy_Latn: 2.7 (174 h)
  - quz_Latn: 0.2 (141 h)
  - quh_Latn: 0.5
  - 40+ other `qu*`/`qv*`/`qx*` codes: mostly ≤ 5
- Common Voice 22 (76 langs, CER, test): omniASR-LLM-7B 6.4, CTC-1B 14.8, Whisper-large-v3 55.6 (arXiv:2511.09690, Table 5).

**4. Omnilingual ASR paper comparisons** (arXiv:2511.09690, https://arxiv.org/pdf/2511.09690)
- **Whisper** (Table 5, FLEURS-81 test CER): 300M-CTC 11.8, 1B-CTC 8.6, 3B-CTC 7.8, 7B-CTC 7.3, 1B-LLM 6.6, 7B-LLM 5.6, Whisper-large-v3 22.6. The 7B-LLM beats Whisper-v3 on 65 of 81 languages.
- **USM** (Table 6, FLEURS-102 test CER): 7B-LLM 6.2 (6.1 with LM fusion), USM-M 6.5, USM 6.9, Maestro-U 8.7.
- **MMS** (Table 7): 7B-LLM 6.2 vs MMS multi-domain + LM 6.3 on FLEURS-102 CER; 1.9 vs 2.1 on MMS-Lab-1143.
- **The paper has no Gemini or ElevenLabs comparison.** The only Gemini number found is the Armenian leaderboard row above.
- Resource buckets (Table 8): mean CER for mid-resource languages (10–50 h; qxp sits at the low/mid boundary) is 3.0 for 7B-LLM and 4.4 for 7B-CTC.
- **Follow-ups:** none. The last commit to the GitHub repo is 2025-12-30 (the per-language CSV). Releases are 0.1.0 (2025-12-12, v2 checkpoints) and 0.2.0 (2025-12-22, `LLM_Unlimited` variants).
  - The HF API lists **no omniASR v3, no larger CTC, and no new LLM variants** from `facebook` as of 2026-10-08. Every 2026 `omniASR` repo on HF is a community port (ONNX, MLX, CoreML, sherpa-onnx) or a language fine-tune.
  - HF transformers support is still community-only (see §(b)).

**5. 2026 models claiming broad coverage or strong low-resource fine-tuning**
- **GigaAM-Multilingual** (`ai-sage/GigaAM-Multilingual` ✔, MIT, 240M / 600M Conformer, HuBERT-style SSL on 2M h of 70+ languages; arXiv:2607.10371, https://arxiv.org/abs/2607.10371). It beats Omnilingual-1B on FLEURS Kyrgyz / Kazakh / Uzbek: WER 5.5 / 4.4 / 7.3 vs 8.1 / 6.6 / 15.4. It has no Quechua and no Americas focus, and was fine-tuned only for 5 Eurasian languages. Not a candidate.
- **BuzzASR** (arXiv:2609.09554, https://arxiv.org/abs/2609.09554). Whisper-large-v3 fine-tuned per FLEURS language. It beats Omni-1B on 60 of 102 languages, but only on FLEURS languages, which exclude Quechua.
- Other new or broad models, none of which covers Quechua with open weights:
  - **Qwen3-ASR** (52 langs).
  - **Qwen3.5-Omni** (113 langs; arXiv:2604.15804). No open weights were found on HF; the open `Qwen/Qwen3-Omni-30B-A3B-Instruct` handles 19 speech-input languages.
  - **Gemma 4** (`google/gemma-4-E4B-it` ✔, 140 text languages). Weak ASR: English average 12.66, multilingual 8.83 (my aggregation).
  - **NVIDIA Canary-1B-v2 / Parakeet-v3 / Granary**: 25 European languages.
  - **Voxtral**: 8–13 languages.
  - **Cohere Transcribe**: 14 languages.
  - **IBM Granite-speech 4.x / 5.0**: ≤ 6 languages.
  - Kyutai (en/fr), ByteDance Seed-ASR (closed), AI4Bharat IndicConformer (Indic only).
  - Meta's 2026 entry on the leaderboard, `meta/muse-voice-transcribe`, is **proprietary** (25 langs).
- **Mozilla CV Spontaneous Speech ASR shared task 2025** (qxp was a Task-4 unseen language). **No leaderboard or findings paper has been published.** Winners were scheduled for 2025-12-12 (https://community.mozilladatacollective.com/shared-task-mozilla-common-voice-spontaneous-speech-asr/). The task baseline was MMS fine-tuning. The one public system paper (Whisper task arithmetic) reports that "the task baseline based on fine-tuning the MMS model outperforms most of our systems" (arXiv:2601.07038, https://arxiv.org/abs/2601.07038).

**6. Head-to-head fine-tuning studies, 2025–26 (10–200 h per language)**

| Study | Data | Result (lower is better) | Fine-tuned omniASR included? |
|---|---|---|---|
| Ethio-ASR (arXiv:2603.23654, https://arxiv.org/abs/2603.23654) | ~190 h/lang, 5 Ethiopian langs (WAXAL) | FT **w2v-BERT 2.0 30.48** < MMS-1B 31.20 < MMS-300M 33.99 < AfriHuBERT 35.08 avg WER. Zero-shot omniASR (which saw part of WAXAL): LLM-7B-v2 32.21, LLM-1B-v2 33.88, CTC-3B-v2 37.01, CTC-1B-v2 40.51. | **No** (zero-shot only) |
| Garhwali, 5 seeds (arXiv:2608.10670, https://arxiv.org/abs/2608.10670) | VAANI official splits | **w2v-BERT 2.0 47.0** < MMS-1B 49.0 < XLS-R-300M 50.2 < HuBERT-L 60.9 < Whisper-v3 66.4 WER | No |
| Nahabwe et al. (arXiv:2512.10968, https://arxiv.org/abs/2512.10968) | 1–400 h, 13 African langs | **MMS-1b-all best on average at 10–20 h** (still best average at 50 h); w2v-BERT data-efficient at the lowest budgets; XLS-R scales best with more data; Whisper-small mid-resource | No |
| Southern Bantu (arXiv:2606.31642, https://arxiv.org/abs/2606.31642) | community corpus, 6 langs | w2v-BERT beats Whisper by 3–4 WER on Nguni; Whisper better on Sotho-Tswana | No |
| **Omnilingual ASR §5.7.5** (Table 21) | 5–10 h, 11 langs incl. **qxp 9.9 h** | qxp CER: CTC-seed 1.70 / **1.61** / 1.81 (300M / 1B / 3B); W2V-seed 2.08 / 1.68 / **1.47**; zero-shot omniCTC 4.49 / 2.94 / 2.71; **zero-shot omniLLM-7B 1.32** | Yes (vs its own seeds only) |
| German tokenizers on omniASR W2V + CTC (arXiv:2610.11646, https://arxiv.org/abs/2610.11646) | 40 fine-tunes | In-domain, the char, BPE and syllable vocabularies tie on WER/CER; small syllable-aware vocabularies help under dialect or spontaneous shift | Yes (same backbone) |

**No published study fine-tunes omniASR and w2v-BERT 2.0 / XEUS / MMS on the same data.** The evidence is indirect:
- Among non-Meta encoders, w2v-BERT 2.0 wins most 2026 head-to-heads.
- omniASR wins every zero-shot and long-tail comparison.
- omniASR is the only family with qxp in its supervised training, with published qxp fine-tuning results.

### f.2 Candidate table

| Rank | Model (HF ID) | Params | License | Benchmark evidence (cited above) | Quechua coverage | 1×L40S 48 GB? |
|---|---|---|---|---|---|---|
| **1** | **omniASR CTC v2**: `facebook/omniASR-CTC-1B` ✔, `facebook/omniASR-CTC-3B` ✔ (v2 by card name) | 975M / 3.08B | Apache-2.0 | qxp FT CER 1.61 (1B, CTC seed) and 1.47 (3B, W2V seed); FLEURS-81 CER 8.6 / 7.8 vs Whisper-v3 22.6; Armenian FLEURS WER 9.55 / 6.49 | **qxp_Latn supervised (9.9 h CV)** + 40 other Quechua codes | 1B: yes (full FT). 3B: tight. AdamW needs ~16 B/param ≈ 49 GB before activations, so freeze lower layers, use an 8-bit optimizer, or offload. |
| **2** | **omniASR LLM v2**: `facebook/omniASR-LLM-7B` ✔ (zero-shot / LoRA), `facebook/omniASR-LLM-1B` ✔ (full FT) | 7.80B / 2.28B | Apache-2.0 | qxp CER **1.32** (paper) / 1.4 (repo CSV) zero-shot, better than single-language FT CTC-1B (1.61) on Meta's split; FLEURS-102 CER 6.2 (USM-M 6.5, MMS 6.3); best open model on Armenian (3.61 / 10.04 WER), ahead of Gemini-3.5-transcribe | qxp_Latn (language-conditioned) | 7B inference ~17 GB. **LoRA r=8 on omniASR_LLM_7B_v2 was trained with fairseq2 on one 48 GB RTX 6000 Ada** (arXiv:2607.12468, https://arxiv.org/abs/2607.12468), but LoRA is not in the official repo. 1B: full FT is feasible with bf16 and checkpointing. |
| **3** | **w2v-BERT 2.0**: `facebook/w2v-bert-2.0` ✔ | 580M | MIT | Best FLEURS CER among SSL encoders (8.7); wins the 2026 FT head-to-heads vs MMS-1B, XLS-R and Whisper (Ethio-ASR 30.48 vs MMS-1B 31.20; Garhwali 47.0 vs 49.0) | None listed in the card; 4.5M h, 143+ langs (Quechua inclusion unknown) | Yes (HF `Wav2Vec2BertForCTC`) |
| 4 | XEUS: `espnet/xeus` ✔ | 577M | **CC-BY-NC-SA-4.0** | Best frozen-SSL baseline in ML-SUPERB 2.0 (CER 40.2); ML-SUPERB 1 h mono CER 25.1 | **qxp, quz, quy and others in pretraining** (card) | Yes, but ESPnet-only. No full-FT evidence against omniASR. |
| 5 | MMS-1B-all: `facebook/mms-1b-all` ✔ | ~1B | CC-BY-NC-4.0 | Best average at 10–20 h in Nahabwe; Mozilla 2025 baseline beat most Whisper systems; Armenian 13.43 / 26.57, behind omniCTC-1B 9.55 / 16.83 | quz / quy adapters, no qxp | Yes |
| — | Leaderboard leaders: `microsoft/Phi-4-multimodal-instruct` ✔, `Qwen/Qwen3-ASR-1.7B` ✔, `CohereLabs/cohere-transcribe-03-2026` ✔, `nvidia/canary-1b-v2` ✔, `mistralai/Voxtral-Small-24B-2507` ✔, `openai/whisper-large-v3` ✔ | 1–24B | MIT / Apache / CC-BY | Lead English and European tracks only; Whisper-v3 collapses on the tail (Armenian 43 / 57 WER; CV22 CER 55.6) | **None** | Mostly yes; irrelevant |
| — | `ai-sage/GigaAM-Multilingual` ✔ | 240M / 600M | MIT | Beats Omni-1B on 3 Central-Asian languages | None | Yes; irrelevant |

### f.3 Verdict

1. **Nothing released through Oct 2026 beats omniASR as a base for qxp.** Every model that tops the Open ASR Leaderboard lacks Quechua. The only tail-language track there, Armenian, is led among open models by omniASR-LLM-7B-v2. Meta has released no omniASR v3, larger CTC model or new LLM variant since the December 2025 v2 / Unlimited drop.
2. **Ranked bases for winning:**
   - **(1) omniASR CTC v2, fine-tuned from the CTC checkpoint** with an n-gram LM. Use 1B now. Try 3B, preferably W2V-3B-seeded with a longer schedule (best qxp CER, 1.47), only if a memory-saving setup fits 48 GB.
   - **(2) omniASR LLM v2.** Its 7B zero-shot qxp CER (1.32) already beats Meta's own fine-tuned 1B CTC. Use 7B as pseudo-labeller and ROVER member, and try **LoRA-tuning 7B** (shown feasible on a 48 GB card) or fully fine-tuning LLM-1B-v2. On present evidence this is the most likely single strongest system, if the orthography mismatch (organizer zero-shot: 26.6 WER / 6.2 CER) is removed by fine-tuning.
   - **(3) w2v-BERT 2.0 CTC** as the diversity member. It is the most consistent winner among non-omniASR encoders in 2026 fine-tuning studies, but it has never been tested against a fine-tuned omniASR and has no confirmed Quechua pretraining.
3. **XEUS** is the only other encoder with qxp in pretraining. It is optional: the license is non-commercial, it requires ESPnet, and the only evidence is from frozen-encoder probes.
4. **Our zero-shot omniCTC-1B gap.** We measure 31 % WER / 5 % CER zero-shot on our scripted dev, while Meta reports 2.94 % CER on its qxp test. That difference points to normalization and orthography, not the model; see §0, point 2.

---

## (g) Squeezing performance: ranked techniques

Compiled 2026-10-09. This section surveys 2019–2026 work for the regime we are actually in. That regime:
- Fine-tuned omniASR-CTC-1B-v2 (character CTC, 30 symbols): greedy **17.5 / 3.0** WER / CER on unseen-sentence scripted dev, and **21.5 / 3.6** on gold spontaneous dev.
- Zero-shot omniASR-LLM-7B-v2 reaches **14.7 / 2.7** on spontaneous.
- WER is dominated by single-character errors inside long agglutinative words.
- The test set reads new sentences, its spontaneous speakers are all 60+, its clips are silence-trimmed, and it was recorded on personal devices.
- We have 1× L40S and about 8 days, i.e. roughly 190 GPU-h.

It does not repeat §(a)–(f); where a technique is already listed there, only new evidence and the concrete recipe are added.

**Ranking principle (user priority, 2026-10-09).** The goal is a *generalizable* model: robust to new speakers, new sentences, elderly voices, new devices and new domains.
- Techniques are therefore ranked by gains expected to transfer, with out-of-distribution (OOD) evidence quoted wherever a paper reports it.
- Techniques that mainly fit this benchmark, or that use the test audio transductively, are collected separately in §g.8 with reasons.

**Conventions.**
- "rel." means relative WER reduction.
- "Reported" numbers are copied from the linked page.
- "Expected" gains and GPU-hours are **my estimates for our setup**. Assume about 9 GPU-h per 10k-step CTC-1B fine-tune and about 3 GPU-h to decode 30 h with LLM-7B.
- Numbers marked UNVERIFIED were seen only second-hand.

### g.1 Elderly-speech adaptation (spontaneous test = 60+ speakers only)

**Background: where the elderly penalty shows up.**
- The penalty is large in *spontaneous* elderly speech and small in read speech.
- In DementiaBank Pitt, the same Conformer system has 39.1 eval WER on elderly participants vs 18.3 on the (younger) investigators (Wang et al., 2022, Interspeech, arXiv:2206.13232, https://arxiv.org/abs/2206.13232).
- On SeniorTalk (55.5 h of spontaneous Mandarin from speakers aged 75–85):
  - Whisper-large-v3 goes from 57.74 zero-shot to 23.84 CER fine-tuned.
  - A strong in-language Paraformer-large only moves 14.91 → 14.41 (Chen et al., 2025, arXiv:2503.16578, https://arxiv.org/abs/2503.16578, Table 6).
- Fine-tuning Whisper on Dutch JASMIN-CGN cut WER by 65% rel. for native elderly speakers (Shekoufandeh, Boersma & van den Bosch, 2025, arXiv:2502.17284, https://arxiv.org/abs/2502.17284, abstract).
- *Implication:* in-domain elderly data, real or simulated, is what moves the needle. A strong base model alone is not enough.

**E1. Speaker-rate-matched speed/tempo perturbation of non-elderly speech (CUHK recipe).** [general: needs no test audio]
- *Citation:*
  - Geng, Xie, Liu, Yu, Hu, Liu & Meng (2020), "Investigation of Data Augmentation Techniques for Disordered Speech Recognition", Interspeech, https://www.isca-archive.org/interspeech_2020/geng20_interspeech.html
  - Liu et al. (2022), "Recent Progress in the CUHK Dysarthric Speech Recognition System", arXiv:2201.05845, https://arxiv.org/html/2201.05845v2
  - Wang, Deng, Geng, Ye, Hu, … , Liu & Meng (2022), Interspeech, arXiv:2206.13232
  - Hu et al. (2024), IEEE/ACM TASLP, arXiv:2407.13782, https://arxiv.org/abs/2407.13782
- *Mechanism:*
  - Normal (younger or control) speech is slowed down per target speaker. The factor is the mean healthy phone duration divided by the target speaker's mean phone duration, measured from forced alignment. This turns young speech into pseudo-elderly speech.
  - The atypical speech itself is speed-perturbed with {0.9, 1.1} or {0.85…1.15}.
  - Speed perturbation beat both VTLP and tempo-only perturbation (Geng 2020).
- *Reported:*
  - Geng 2020: up to 2.92 abs (9.3% rel.) WER, reaching 26.37% on UASpeech (abstract).
  - Liu 2022: training data 30.6 h → 130.1 h; speed perturbation alone 31.45 → 29.47 WER, all augmentation combined 28.53; growing further to 207.5 h added only 0.1–0.2.
  - Wang 2022 (elderly Pitt, speaker-dependent perturbation of investigator speech plus SpecAugment, 15.7 h → 58.9 h, together with other changes): overall 35.9 → 24.8; elderly Eval 39.1 → 25.9. This total includes NAS, LHUC and system combination, so the augmentation gain is not isolated.
  - Hu 2024: the augmented sets (Pitt 27.2 → 58.9 h; JCCOCC MoCA 32.4 → 156.9 h) underlie every elderly system they report.
- *Recipe for us:*
  - Measure characters per voiced second from CTC alignments for the elderly clips (0.05 h gold + 1.7 h silver + 0.3 h unlabelled) and for each non-elderly spontaneous speaker.
  - Speed- or tempo-perturb every non-elderly spontaneous clip by that speaker's ratio, roughly {0.8, 0.85, 0.9}. Use tempo (sox `tempo`) if pitch should be kept.
  - Keep the standard {0.9, 1.0, 1.1} for scripted data.
  - Do this on the fly inside the next retrain (the pseudo-label round, §g.5); it needs no extra run.
- *Expected:* 3–8% rel. on elderly spontaneous dev, about 0 on read speech.
- *Cost:* 0 extra GPU-h if folded into a retrain (data loading only).
- *Risk:* low. Resampling-based speed factors below about 0.85 shift formants unnaturally; for stronger slow-downs use pitch-preserving tempo.

**E2. Upweight elderly clips, then a short in-domain stage.** [general]
- *Citation:* Hu et al. (2024, TASLP, arXiv:2407.13782): two-stage fine-tuning (out-of-domain first, then in-domain elderly) was consistently best.
- *Recipe:* repeat the gold and silver elderly clips 3–5× per epoch, each copy with its own speed factor (0.85–1.1) and an optional VTLP warp (0.9–1.1). Then run a final 1–2k-step stage on spontaneous-only data with elderly clips upweighted.
- *Expected:* 2–5% rel. on elderly spontaneous.
- *Cost:* about 2 GPU-h.
- *Risk:* overfitting to the handful of elderly training speakers. Hold out elderly speakers by speaker for dev, and stop on spontaneous dev.

**E3. Confidence-curriculum pseudo-labelling of elderly audio.** [general]
- *Citation:* Deng, Xie, Hu et al. (2026), "Confidence Score Guided Incremental and Speaker Adaptive Pseudo-Labeling for Semi-Supervised Elderly Speech Recognition", arXiv:2606.16546, https://arxiv.org/abs/2606.16546
- *Mechanism:* pseudo-labels are added in order from high to low confidence, ranked within each speaker, and regenerated over 4 iterations (keeping 80%).
- *Reported:* Whisper-medium with LoRA, 10% of speakers labelled.
  - Vanilla pseudo-labelling with no filtering: Pitt 23.62 WER / MoCA 32.52 CER.
  - The proposed method: 21.50 / 30.25, i.e. −1.45 abs (6.21% rel.) and −2.27 abs (6.98% rel.).
  - The 100%-labelled topline: 20.68 / 28.68.
- *Recipe:* this is folded into §g.5 S1. Rank unlabelled spontaneous clips per speaker, and put the 0.3 h of unlabelled elderly audio first.
- *Expected:* see §g.5. *Risk:* unfiltered pseudo-labels hurt (23.62 vs 21.50).

**E4. Device and silence-condition robustness.** [general]
- *Citation:* Vu, Zeng, Xu & Chng (2019), "Audio Codec Simulation based Data Augmentation for Telephony Speech Recognition", APSIPA ASC, http://www.apsipa.org/proceedings/2019/pdfs/216.pdf
- *Reported:* training on heavily codec-distorted simulated data cut WER on real telephony test sets by 7.28–12.78% vs the baseline (abstract).
- *Recipe:*
  - With p≈0.3, pass clips through a random low-bitrate codec (Opus 8–24 kbps, AMR-WB, MP3 32–64 kbps, via ffmpeg) or a band-limit plus small RIR.
  - Randomly trim leading and trailing silence to 0–0.1 s, since test clips are silence-trimmed (FINDINGS.md item 2).
  - Randomly crop long spontaneous clips toward the test length distribution (mean 9.1 s vs 15–17 s in train). Cut at CTC-alignment word boundaries so the transcript can be cut too.
- *Expected:* 0–5% rel. on test; not measurable on our dev unless dev audio is processed the same way.
- *Cost:* 0 extra GPU-h (folded into a retrain).
- *Risk:* low. The evidence is telephony, not smartphones.

Not transferable to the generalization goal (see §g.8): per-test-speaker adaptation such as f-LHUC with spectral-basis speaker features (Geng et al., 2024, arXiv:2407.06310, https://arxiv.org/abs/2407.06310: up to 5.32 abs, 18.57% rel.), and the test-time part of structured speaker-deficiency adaptation (Hu et al., 2024, arXiv:2412.18832, https://arxiv.org/abs/2412.18832: up to 1.50 abs / 6.94% rel. on Pitt). Both need test-speaker audio.

Zero-shot prompt-expert adaptation (MOPSA, Deng et al., 2025, Interspeech, arXiv:2505.24224, https://arxiv.org/abs/2505.24224: 4.21% rel. WER / 5.40% rel. CER, abstract) is per-utterance and in principle general. It needs Whisper-style prompt training, and the gain is too small for 8 days.

Elderly-voice TTS augmentation:
- Reported: Lee et al., 2026, arXiv:2604.24770, https://arxiv.org/html/2604.24770v1. Common Voice 70+ 4.1 → 2.2 WER; Korean VOTE400 11.6 → 5.2.
- Why not here: there is no qxp TTS (§C4).

### g.2 Decoding for an agglutinative language

§C2 already covers KenLM, pyctcdecode, flashlight lexicon-free decoding, Morfessor and Whisper-LM. The new evidence below is mostly from wav2vec2-family CTC models on Finnish, Estonian, Indic, Quechua and Common Voice Spontaneous languages.

**D1. Broad n-gram LM with CTC: word-level plus open-vocabulary (character or subword) units, interpolated over corpora.** [general, if the text is broad]
- *Reported:*
  - **Estonian, character XLS-R CTC with shallow fusion of a 4-gram over 40k SentencePiece unigram units** (closest to our setup): Common Voice 12.55 → 10.59 (XLS-R-300M) and 10.13 → 8.81 (XLS-R-1B), i.e. 13–16% rel. Per-corpus Kneser-Ney LMs were interpolated with dev-tuned weights. A Transformer LM gave "minor or no improvements". (Olev & Alumäe, 2022, Baltic J. Modern Computing 10(3), https://doi.org/10.22364/bjmc.2022.10.3.14)
  - **Colloquial Finnish, wav2vec2-Large CTC (100 h) with a word 4-gram:** dev WER 22.50 → 20.34 but CER only 6.08 → 5.83; test 24.03 → 21.75 (CER 7.02 → 6.80). LM gains are mainly word-level. (Moisio et al., 2022, Language Resources and Evaluation, arXiv:2203.12906, https://arxiv.org/abs/2203.12906)
  - **Mozilla CV Spontaneous Speech 2025 languages, XLS-R + KenLM 4-gram (α=0.5, β=1.0, beam 100):** helped 19 of 21 languages for XLS-R and 17 of 21 for MMS; where it hurt, the loss was ≤1 abs. Examples: Scots 0.302 → 0.244 (−19.2%), Wixárika 0.560 → 0.506. (Bhandari, Kumar & Mulhern, 2026, ComputEL-9, https://aclanthology.org/anthology-files/pdf/computel/2026.computel-1.5.pdf)
  - **IndicWav2Vec (flashlight, lexicon-based word 6-gram):** MSR Tamil greedy 22.1 → transcript-only LM 14.9 → plus large external text 13.6. Telugu 22.9 → 14.4 → 11.0. Tuning grid α∈[−4,4], β∈[0,5], beam 64; test beam 1024. (Javed et al., 2022, AAAI, arXiv:2111.03945, https://arxiv.org/abs/2111.03945)
  - **Bengali/Bhojpuri (MADASR, ASRU 2023), 20-gram BPE-level KenLM fused into Whisper:** 16.81 → 14.85 and 16.25 → 14.19 dev WER. (Transsion TSUP, arXiv:2307.11778, https://arxiv.org/abs/2307.11778)
- *Domain-mismatch evidence (generalization risk):*
  - **Quechua, about 3 h, wav2vec2:** no LM 81.35 → transcript LM 65.73 → larger LM adding religious/government text **66.81**. A bigger but mismatched LM did not help. (Liu, Venkateswaran, Le Ferrand & Prud'hommeaux, 2024, Findings of ACL, https://aclanthology.org/2024.findings-acl.13.pdf)
  - **Nahabwe et al. 2025 (arXiv:2512.10968, https://arxiv.org/abs/2512.10968):** a 5-gram hurt some languages, e.g. Bemba XLS-R 49.63 → 60.72 and Bambara w2v-BERT 36.8 → 50.2.
  - **Whisper-LM (arXiv:2503.23542, https://arxiv.org/abs/2503.23542):** n-gram fusion hurt OOD on Spanish MLS (−17%) and Spanish FLEURS (−8%), while LLM fusion was smaller but consistently positive.
  - **AmericasNLP 2022 (Romero et al., arXiv:2404.08368, https://arxiv.org/abs/2404.08368):** KenLM "did not lead to significant improvement and even degraded performance for some languages".
- *Units for open vocabulary:*
  - Finnish hybrid systems with 400-unit BPE and VariKN n-grams "up to 10-gram scale" (Rouhe, Virkkunen, Leinonen & Kurimo, 2022, Interspeech, https://www.isca-archive.org/interspeech_2022/rouhe22_interspeech.pdf).
  - Inuktitut, where over 60% of tokens are OOV (Gupta & Boulianne, 2020, LREC, https://aclanthology.org/2020.lrec-1.307.pdf): dev WER word LM 108.7 → Morfessor 80.7 → syllables with in-unit word-boundary markers 74.3. Predicting boundaries afterwards was worse (75.6).
  - Malayalam: syllable units win once the test set's OOV rate is moderate, but the word LM wins when it is under 6% (Manohar, Jayan & Rajan, 2023, arXiv:2301.06736, https://arxiv.org/abs/2301.06736).
- *Recipe for us:*
  1. **Never** train the LM on dev or test sentences. Build three Kneser-Ney LMs (scripted transcripts, spontaneous transcripts, Puno-respelled external text) and interpolate them with weights tuned for perplexity on the unseen-sentence dev. The external weight will probably land around 0.2–0.4; cap it if spontaneous dev degrades.
  2. First pass, character track: lexicon-free character 8–10-gram (flashlight).
  3. First pass, subword track: SentencePiece unigram or Morfessor units, 1–4k, as a 6–8-gram with in-unit word-boundary marking (`▁` prefix or `+m+`), pruning singletons at order ≥5.
  4. Also a word 4-gram in pyctcdecode, with `unk_score_offset` raised from −10 to about −4, since 22.5% of spontaneous tokens are unseen. This is a tool-behaviour tip, not from a paper.
  5. Tune α∈[0,2] in steps of 0.1–0.25 and β∈[−2,3] at beam 64–128, separately for read and spontaneous speech. Decode the final run at beam 500–1000. Report CER alongside WER.
- *Expected:* read speech 8–15% rel. (17.5 → about 15–16); spontaneous 3–8% rel. Expect less than the XLS-R literature, because omniASR's encoder already has an implicit LM, much as MMS gained less than XLS-R in Bhandari 2026.
- *Cost:* CPU plus about 2 GPU-h for logit dumps.
- *Risk:* over-weighting in-domain text helps dev and hurts new domains. Choose the weights that are best on *both* dev sets, not on either one alone.

**D2. Second pass: N-best rescoring with a neural character/subword LM, then CER-loss MBR.** [general]
- *Reported:*
  - Finnish Morfessor hybrid, 50-best: n-gram 25.41 → LSTM + n-gram 20.57 → Transformer-XL 20.04. With 1000-best, 19.3 → 18.38. BERT did not help. (Jain, Rouhe, Grönroos & Kurimo, 2020, Interspeech, https://www.isca-archive.org/interspeech_2020/jain20b_interspeech.pdf)
  - Transformer rescoring after an n-gram adds only 0.1–1.0 abs on IndicWav2Vec (Tamil 14.9 → 14.8).
  - English CTC: 11 rescoring strategies using only CTC or acoustic scores never beat greedy, but CER-loss MBR (τ=10) over 128 hypotheses with RoBERTa pseudo-log-likelihood gives 6.02 → 5.42 (9% rel.), significant in 11 of 13 conditions. (Novosad, 2026, arXiv:2606.23306, https://arxiv.org/abs/2606.23306)
- *Recipe:*
  - Train a small character Transformer LM (about 6 layers, 512-d) on the same interpolated text.
  - Rescore the 100-best from D1 with: acoustic + α·n-gram + γ·neural LM + β·length.
  - Then run MBR with a character edit-distance loss over the rescored list (temperature tuned on dev).
- *Expected:* 1–4% rel. on top of D1.
- *Cost:* about 2–3 GPU-h to train the LM; rescoring is cheap.
- *Risk:* low; it can't do worse than D1 if the weights are tuned on both dev sets.

LLM-based generative error correction for Quechua: see §g.6 (G4) and §g.8. Hypothesis-level MBR across *systems*: see §g.3 (A4).

### g.3 Weight averaging, soups, interpolation with the pretrained model, ensembles

**A1. WiSE-FT: interpolate fine-tuned and pretrained/zero-shot weights.** [general; the main OOD lever]
- *Citation:* Wortsman et al. (2022), "Robust fine-tuning of zero-shot models", CVPR, arXiv:2109.01903, https://arxiv.org/abs/2109.01903
- *Mechanism:* θ = (1−α)·θ_zero-shot + α·θ_fine-tuned. Linear paths between a pretrained model and its fine-tune are low-loss, and the interpolated model keeps much of the zero-shot model's robustness.
- *Reported:* vision (abstract): +4 to 6 pp on ImageNet distribution shifts over prior work while *also* gaining +1.6 pp in-distribution; +2 to 23 pp on six other shifts.
- *Speech evidence:*
  - **Speech-FT** (Lin, Huang, Tang & Lee, 2026, IEEE/ACM TASLP, arXiv:2502.12672, https://arxiv.org/abs/2502.12672): interpolating a fine-tuned HuBERT with the pretrained one gives phone error rate at α (fine-tuned weight) 0 / 0.1 / **0.25** / 0.5 / 1.0 of 5.17 / 4.99 / **4.76** / 5.15 / 10.34. Caveat: this is a frozen-representation SUPERB probe, not end-to-end ASR.
  - **Vander Eeckt & Van hamme (2023, ICASSP, arXiv:2210.15282, https://arxiv.org/abs/2210.15282)**: averaging the adapted model with the previous one cuts average WER over CV English dialect tasks from 15.25 (fine-tuning) to 13.72, and the new task *itself* improved, 11.4 → 10.4. Multilingual: 34.73 → 20.28.
  - **Carvalho, Teixeira, Rolland & Abad (2026, arXiv:2603.05354, https://arxiv.org/html/2603.05354)**, Whisper-large-v3 over 10 Portuguese domains, WER as in-domain / OOD:
    - Full fine-tuning: 8.54 / 17.65
    - Uniform soup: 10.15 / 17.60
    - TSV-M merging: 9.41 / **16.07**
    - TIES damaged FLEURS: 9.63 → 14.03.
  - Merging trades 0.7–1.7 WER in-domain for up to 9% rel. OOD.
- *Recipe for us:*
  - **CTC-1B:** grid α∈{0.5, 0.6, 0.7, 0.8, 0.9, 1.0} between θ_FT and the omniASR-CTC-1B-v2 start point. Interpolate the encoder only and keep our fine-tuned 30-symbol head, since the original head has a different vocabulary.
  - Our zero-shot CTC is weak (35% spontaneous WER), so expect the best α at ≥0.7.
  - **Select α on the elderly/spontaneous dev and on a codec-processed copy of dev, never on in-domain read dev alone** (that always picks α=1).
  - **LLM-7B LoRA:** the same idea costs nothing. Scale the LoRA delta by α∈{0.25, 0.5, 0.75, 1.0}; α=0 is the strong zero-shot model (14.7 spontaneous).
- *Expected:* 0–4% rel. on dev, more on shifted test conditions. It may cost 0–2% on read dev.
- *Cost:* about 6 evaluation passes, about 2 GPU-h.
- *Risk:* low, since α is a dial. Use linear interpolation only; avoid TIES/DARE (Carvalho 2026).

**A2. Multi-run soups, plus checkpoint averaging or EMA in every run.** [general]
- *Reported:*
  - **Ducorroy & Riad (2025, Interspeech, arXiv:2505.20477, https://arxiv.org/abs/2505.20477)**, Whisper on dysarthric speech (the closest analogue to atypical elderly speech):
    - Dev: fine-tune 15.0 → single-run checkpoint average 13.9 → multi-run soup 13.2.
    - Test: 14.5 → 11.2 (multi-run) → 10.8 (selective).
    - 12% rel. overall; 16.2% rel. on long-form.
  - **Selective merging for child ASR (Balaji Shankar, Wang, Eren & Alwan, 2025, arXiv:2501.08468, https://arxiv.org/abs/2501.08468):** fine-tune 9.34 → linear merge 8.86 → selective merge + SpecAugment 8.69.
  - **NTU, MLC-SLM 2025 (arXiv:2506.13339, https://arxiv.org/abs/2506.13339):** averaging the last 15 LLM-ASR checkpoints 13.63 → 11.70.
  - **Itgan, NADI 2026 (arXiv:2610.09934, https://arxiv.org/abs/2610.09934):** LoRA adapter soup 15.09 → 14.55.
  - **Counter-evidence (Ajroldi, Orvieto & Geiping, 2025, arXiv:2502.06761, https://arxiv.org/abs/2502.06761):** on well-tuned LibriSpeech runs, LAWA/EMA give almost the same final WER (Conformer 0.08849 → 0.08841); the benefit is mostly reaching it in 82% of the steps.
- *Recipe:*
  1. Every run keeps an EMA (decay 0.999) or averages 5–10 checkpoints from the last ~10% of training.
  2. Then train 2–3 extra CTC-1B fine-tunes from the same initialisation with different seeds, learning rates and augmentation mixes.
  3. Build a greedy soup selected on mean(read-unseen, spontaneous) dev.
- *Expected:* 0–3% rel. from averaging, plus 3–6% rel. from the multi-run soup.
- *Cost:* averaging is free. Soups cost about 9 GPU-h per extra run. Each extra run can double as an experiment arm (CR-CTC, a different augmentation mix), so the soup comes almost free.
- *Risk:* soups only work when the members stay in one basin. Same initialisation and moderate learning rate keep them there; check that the soup beats the best member.

**A3. Frame-level CTC posterior fusion (same vocabulary and 20 ms frame rate).** [general]
- Average log-posteriors across CTC members, then run LM beam search.
- Older work warns that separately trained CTC models put their spikes at different frames (arXiv:1703.10356, https://arxiv.org/abs/1703.10356). That risk is smaller when members share an initialisation, which is also what makes a weight soup (A2) possible. If the soup works, prefer it: it is one model and N× cheaper to run.
- *Expected:* 2–5% rel. over the best member.
- *Cost:* N× inference.
- *Risk:* spike misalignment. Check on dev.

**A4. Cross-architecture hypothesis combination (CTC+LM and LLM-7B fine-tuned and zero-shot) with confidence-weighted ROVER or MBR, plus CTC guards.** [general]
- *Reported:*
  - Kaldi + XLS-R ROVER weighted by Rényi-entropy confidence with temperature scaling, Irish: about 14% rel. on the main test set and about 20% rel. on noisy, imbalanced data (Parikh, ten Bosch & van den Heuvel, 2024, LREC-COLING, https://aclanthology.org/2024.lrec-main.547/, abstract).
  - Time-aware ROVER (MOVER): NOTSOFAR-1 10.81 → 9.89; CHiME-8 best single 19.62–20.31 → 18.54 (Kamo, Ochiai, Delcroix & Nakatani, 2025, arXiv:2508.05055, https://arxiv.org/abs/2508.05055).
  - **NADI 2026 winner (Wifaq): 4 w2v-BERT-2.0 CTC + 5-gram KenLM + ROVER/MBR**, holdout best single 46.12 → 4-way 44.00 → 11-way 42.59. The organizers concluded "system combinations helped almost everywhere" (arXiv:2609.27086, https://arxiv.org/abs/2609.27086).
  - CTC and LLM errors are complementary: accepting the CTC draft when its entropy is low, and verifying with the LLM otherwise, gives 5.75 → 5.58 on 9 Open-ASR corpora (Saon et al., 2026, arXiv:2603.11243, https://arxiv.org/abs/2603.11243).
  - Confidence-based selection of one model per utterance preserved base-domain accuracy when a dialect-tuned model was added: SLR83 base 5.12 / fine-tuned 7.85 / ensemble 5.26 (Gitman, Lavrukhin, Laptev & Ginsburg, 2023, arXiv:2306.15824, https://arxiv.org/abs/2306.15824).
- *Recipe:*
  - Systems:
    1. CTC soup + D1/D2 decoding.
    2. LoRA LLM-7B (§g.6).
    3. Zero-shot LLM-7B.
    4. Optionally w2v-BERT-2.0 CTC.
  - Word confidence: CTC posterior mean for CTC; token log-probability for the LLM, temperature-calibrated on dev.
  - Combine with our existing ROVER/MBR code, using per-system weights tuned on both dev sets.
  - Guard: if an LLM hypothesis exceeds 1.5× the CTC hypothesis length or contains a repeated n-gram (n≥5), drop it from the vote.
- *Expected:* 8–15% rel. over the best single system.
- *Cost:* about 5–8 GPU-h (LLM N-best decoding of dev and test).
- *Risk:* weights over-tuned to dev; keep them coarse (0.5/1/2).

### g.4 Fine-tuning tricks for the omniASR/wav2vec2 CTC model

**F1. Anchor to the pretrained weights (less forgetting = better OOD).** [general]
- *Reported:*
  - **XLSR-53 CTC fine-tuning, Zaiem, Parcollet & Essid (2024, arXiv:2407.00756, https://arxiv.org/html/2407.00756).** WER as in-domain (GigaSpeech / NST Danish) | OOD (Common Voice):
    - Full fine-tune: 28.85 / 10.99 | 30.41 / 20.07
    - LoRA: 26.68 / 10.37 | 24.7 / 17.54
    - EWC: 27.21 / 10.35 | 24.44 / 17.40
    - Replay of the self-supervised loss: 27.54 / 9.29 | 23.56 / 16.43
  - Up to 15.7% / 22.5% rel. OOD. The OOD gain is 2–3× the in-domain gain.
  - **Robust wav2vec2 (Hsu et al., 2021, arXiv:2104.01027, https://arxiv.org/abs/2104.01027):** a model fine-tuned on one domain transfers poorly. Switchboard WER is 13.89–16.90 when fine-tuned on LibriSpeech vs 10.39–10.80 when fine-tuned in-domain.
- *Recipe:*
  - Keep the feature encoder frozen throughout, and train only the head for the first 1–2k steps (wav2vec 2.0 froze the transformer for 10k updates; arXiv:2006.11477, https://arxiv.org/abs/2006.11477).
  - Add an L2-SP penalty λ‖θ−θ₀‖² on the transformer, with λ tuned in {1e-4, 1e-3}.
  - Or simply rely on A1 (WiSE-FT), which gives most of the same effect after training at no training cost.
- *Expected:* 2–5% rel. on dev, more under shift.
- *Cost:* can share a run with F2.
- *Risk:* too strong an anchor under-fits the orthography. Monitor read dev.

**F2. CR-CTC: consistency regularisation between two augmented views.** [general]
- *Citation:* Yao, Kang, Yang, Kuang, … , Povey (2025), "CR-CTC: Consistency regularization on CTC for improved speech recognition", ICLR, arXiv:2410.05101, https://arxiv.org/abs/2410.05101
- *Mechanism:* each utterance gets two differently SpecAugmented views, and a symmetric KL between their CTC posteriors is added (α≈0.2), with about 2.5× wider time masking. This combines self-distillation, a masked-prediction-like objective and suppression of peaky CTC outputs.
- *Reported (Zipformer trained from scratch):*
  - LibriSpeech-M test-clean/other: 2.52/6.02 → 2.1/4.61.
  - AISHELL CER: 4.47/4.8 → 3.72/4.02.
  - GigaSpeech (10k h): 11.16/11.16 → 10.31/10.41.
  - The gain persisting at 10k h suggests it is not just an in-domain overfitting fix.
- *Recipe:* duplicate each batch with independent SpecAugment and speed/tempo views; loss = CTC(v1) + CTC(v2) + 0.2·KL_sym; halve the per-view batch size.
- *Expected:* 3–8% rel. Unverified on SSL fine-tuning, so expect less than from scratch.
- *Cost:* about 1.7× per step, about 15 GPU-h.
- *Risk:* medium. Confirm on 2 seeds: Batra et al. (2026, ICNLSP, arXiv:2608.10670, https://arxiv.org/abs/2608.10670) found single-run loss tweaks (focal CTC, matra-weighted loss) "do not replicate" under seed testing with w2v-BERT/MMS.

**F3. Schedule and masking.** [general]
- *Reported (recipes):*
  - wav2vec 2.0: tri-stage learning rate 10/40/50%; time mask p 0.065–0.075 with span 10; channel mask 0.004–0.008; LayerDrop 0.1 (arXiv:2006.11477).
  - XLS-R: learning rate searched in [2e-5, 3e-4] for 0.3B/1B; mask probability 30–75% tuned on dev; stochastic depth 10%; 13k–50k updates (arXiv:2111.09296, https://arxiv.org/abs/2111.09296).
  - omniASR: per-language fine-tuning from the CTC checkpoint at learning rate 1e-5 for 5k steps; from SSL, 30k steps (arXiv:2511.09690).
- *Recipe:* we stopped near step 4.5k. Extend to 12–20k steps (with the 30 h of pseudo-labels the data roughly doubles) with the tri-stage schedule, an effective time-mask coverage of about 40–60%, channel mask about 0.005 and LayerDrop 0.1. Run a short learning-rate sweep {1e-5, 2e-5, 3e-5}.
- *Expected:* 3–6% rel.
- *Cost:* folded into the §g.5 retrains.
- *Risk:* low.

**F4. Intermediate / self-conditioned CTC: at most one auxiliary loss.** [general; low priority]
- *Reported, from scratch:*
  - InterCTC (Lee & Watanabe, 2021, ICASSP, arXiv:2102.03216, https://arxiv.org/abs/2102.03216): WSJ eval92 13.9 → 12.4.
  - Self-conditioned CTC (Nozaki & Komatsu, 2021, Interspeech, arXiv:2104.02724, https://arxiv.org/abs/2104.02724): TEDLIUM2 12.2 → 10.1 → 9.4.
  - Hierarchical BPE targets (Higuchi et al., 2022, ICASSP, arXiv:2110.04109, https://arxiv.org/abs/2110.04109): LibriSpeech-100 test-other 25.5 → 20.6.
- *On SSL encoders, gains are small or negative:*
  - XLS-R frontend on FLEURS, CER 13.9 → 13.7 (Chen et al., 2023, arXiv:2302.12829, https://arxiv.org/abs/2302.12829).
  - CTC losses at every layer of wav2vec2 *hurt* the final layer: 3.4/8.6 → 4.3/12.2 (Wright et al., 2023, arXiv:2309.09546, https://arxiv.org/abs/2309.09546).
- *Recipe:* if tried, one character-level InterCTC at layer 24/48 with weight 0.1–0.3, inside the F2 run.
- *Expected:* 0–3% rel. *Cost:* about 0. *Risk:* can hurt; keep only if it wins on 2 seeds.

**F5. Re-initialise the top 1–2 transformer layers.** [general; low priority]
- *Reported:* wav2vec2-Base, re-initialising the top layers before fine-tuning (Pasad, Chou & Livescu, 2021, ASRU, arXiv:2107.04734, https://arxiv.org/abs/2107.04734). LibriSpeech test-other: 10 min 56.7 → 51.8; 10 h 20.6 → 19.4. The gain shrinks as labelled data grows.
- *Expected at 36 h:* 0–3%.
- *Cost:* a fresh run (9 GPU-h).
- *Risk:* the top layers of a *CTC-fine-tuned* checkpoint (our starting point) are already ASR-specialised, so this may lose the omniASR qxp prior. **Not recommended.**

**Skip:**
- Focal CTC (no gain under seed testing; Batra 2026).
- A BPE output head. There is no verified SSL evidence for it in agglutinative languages, and it does not target our single-character errors; BPE belongs in the LM (D1).
- Layer-wise learning-rate decay. I found no ASR WER evidence; it is only reported for WavLM speaker and emotion tasks.

### g.5 Self-training on the ~30 h untranscribed spontaneous audio, and test-time adaptation

**S1. Distillation from the 7B teacher, filtered by agreement between two systems; re-initialise each round; elderly-first curriculum.** [general, uses training audio only]
- *Reported, label-free filtering:*
  - **uDistil-Whisper (Waheed, Kadaoui, Raj & Abdul-Mageed, 2025, NAACL, arXiv:2407.01257, https://arxiv.org/abs/2407.01257).** Filtering on "proxy WER" between two different teachers works as well as filtering with real labels. Benchmark average WER: teacher 41.9 / student without filter 40.1 / filtered with labels 35.3 / label-free SONAR filter 35.5. On the hardest set the proxy-filtered student beat the teacher (64.6 vs 69.8).
  - **Pratinidhi (Bhogale et al., 2024, Interspeech, arXiv:2408.14026, https://arxiv.org/abs/2408.14026).** Two transcribers must agree exactly, and two evaluators must pass (entropy confidence ≥0.7, SONAR similarity ≥0.8). Only 6.4% of 28.6k h is kept. Hindi IndicYT 33.3 → 24.7, **while the OOD Vistaar benchmark stayed at 11.9 → 11.9 (no robustness loss)**.
  - **Distil-Whisper (Gandhi, von Platen & Rush, 2023, arXiv:2311.00430, https://arxiv.org/abs/2311.00430), Table 10.** With a reference-based WER filter: no filter 13.4 → λ=10% (45% dropped) 11.4 average WER; OOD 9.1 → 7.4. This shows that dropping about half the data is the sweet spot.
  - **Three-way agreement (CER < 5%) selection (Rangappa et al., 2025, arXiv:2506.03681, https://arxiv.org/abs/2506.03681):** 100 h selected out of 7,500 h gives 12.2 WER vs 12.3 using all the data.
- *Reported, low-resource rounds:*
  - **Punjabi, XLSR-53 (Singh, Hou & Wang, 2023, arXiv:2308.05269, https://arxiv.org/abs/2308.05269).** Filter on a length-normalised 5-gram score, relaxed every round (0.5, 0, −0.5 … −2.5), and re-initialise from the pretrained model every round (better than continuing). After 5 rounds: Common Voice 14.04 → 11.42; Shrutilipi 16.87 → 14.67; 13–19% rel.
  - **Warning on over-training the seed model (Berrebbi et al., 2023, ICLR, arXiv:2210.08711, https://arxiv.org/abs/2210.08711, Table 1, 10 h labelled):** more supervised steps before pseudo-labelling started made results *worse*: 10k 14.3 / 20k 17.1 / 40k 22.9 dev-clean.
  - **Kaggle Bengali.AI 2023 (Common Voice-style data with noisy labels).** 1st place: 3 rounds of iterative pseudo-label filtering (keep WER < 15%) reached public LB about 0.380; merging short clips, pseudo-labelled YouTube audio and a punctuation-model ensemble then took it to 0.312 public / 0.372 private; "fixing the annotation noise is the most crucial part" (https://www.kaggle.com/competitions/bengaliai-speech/discussion/447961). 2nd place dropped the ~10% of training clips with the highest WER and retrained (https://www.kaggle.com/competitions/bengaliai-speech/discussion/447976).
  - **LLM fusion of several systems' outputs as the pseudo-label:** pseudo-label WER 15.63 single → 14.36 ROVER → 9.30 speech-LLM fusion (Prakash et al., 2025, Interspeech, arXiv:2506.11089, https://arxiv.org/abs/2506.11089).
- *Recipe for us:*
  1. **Teacher:** omniASR-LLM-7B (zero-shot now, LoRA-tuned once available) combined via A4 with our CTC+LM. Never use our own CTC alone, which would bring confirmation bias.
  2. **Filter:**
     - Keep a clip if WER(7B, CTC+LM) ≤ 15–20%.
     - Require characters per second between 0.5× and 1.5× the median.
     - Drop the bottom 10% by CTC forced-alignment score of the 7B text.
     - Duration ≤ 30 s.
     - Expect to keep 50–70%. Tune the threshold on gold spontaneous dev; never train on dev.
  3. **Round 1:** re-initialise from omniASR-CTC-1B-v2. Mix labelled to pseudo-labelled about 1:1 by hours, with gold spontaneous upsampled 2–3×. Include E1/E2/E4 augmentation on the student and F3 schedule, 12k steps.
  4. **Round 2,** only if round 1 gives ≥5% rel.: re-label with the combined system, loosen the threshold to about 25%, add the remaining clips in confidence order, keeping elderly speakers first (E3). Stop after 2 rounds.
  5. **Regression check each round:** read-unseen dev must not get worse by more than 0.3 abs, and per-speaker WER must not regress for elderly speakers.
- *Expected:* spontaneous 10–20% rel. (21.5 → about 17–19 for the CTC student); read 0–3%.
- *Cost:* about 2 × (10 GPU-h train + 3 GPU-h decode), roughly 26 GPU-h.
- *Risk:* the 7B silver labels follow omniASR's orthography. Normalise them with the Puno respeller before filtering.

**S2. Continued SSL pretraining on the training audio (incl. the 30 h untranscribed spontaneous).** [general]
- Already in §(a) step 6. New number: adding in-domain unlabelled audio to pretraining closes 66–73% of the gap to in-domain labels (Hsu et al., 2021, arXiv:2104.01027).
- Priority is lower than S1 because omniASR already saw qxp, and S1 uses the same audio more directly.

**S3. Test-time adaptation (TTA) / transductive use of test audio.** [benchmark-specific; see §g.8]
- *Reported:*
  - **SGEM** (Kim, Park, Shim & Yang, 2023, Interspeech, arXiv:2306.01981, https://arxiv.org/abs/2306.01981), wav2vec2 CTC, WER as source → SUTA → SGEM: CHiME-3 31.2 → 25.0 → 24.7; Common Voice 36.9 → 31.4 → 31.1; TED 13.2 → 12.0 → 12.0.
  - **DSUTA** (Lin, Huang & Lee, 2024, EMNLP, arXiv:2406.11064, https://arxiv.org/abs/2406.11064): multi-domain stream, source 61.0 → SUTA 53.3 → DSUTA 35.8.
  - **NTT CHiME-7**: re-training per evaluation session on combined-system pseudo-labels gave eval 19.4 → 19.2 (~1%) (Kamo et al., 2023, CHiME workshop, https://www.isca-archive.org/chime_2023/kamo23_chime.pdf).
- *Shared-task norms:*
  - CHiME-8 allows "self-supervised adaptation on evaluation data … independently for each session" but prohibits statistics accumulated over the whole eval set (https://www.chimechallenge.org/challenges/chime8/task1/rules).
  - MLC-SLM 2025 prohibits "using the evaluation subsets for model fine-tuning or training", and also banned ROVER-style fusion (arXiv:2509.13785, https://arxiv.org/abs/2509.13785).
  - Our data-use agreement ("to develop, evaluate, and submit systems") is more permissive. But a transductive system does not demonstrate generalization.
- *Verdict:* do not include it in the primary system. If tried at all, run it as a clearly labelled contrastive run: per-utterance SGEM on LayerNorm/feature-extractor parameters only, under 1 GPU-h, expected 0–5% rel.

### g.6 LLM-ASR (omniASR-LLM-7B) fine-tuning and error correction

**G1. LoRA on the 7B decoder, encoder frozen, always conditioned on `qxp_Latn`.** [general if anchored]
- *Evidence on forgetting (why LoRA rather than full fine-tuning):*
  - **LoRA-Whisper (Song et al., 2024, arXiv:2406.06619, https://arxiv.org/abs/2406.06619).** Rank ablation, WER: r8 9.00 / r16 8.54 / **r32 7.94** / r48–64 about 8.00. On new languages LoRA 22.97 vs full fine-tuning 28.33. Original languages stay at 9.52 with LoRA but fall to **32.29** under full fine-tuning.
  - **DAMA (Xiao et al., 2026, arXiv:2602.01008, https://arxiv.org/abs/2602.01008).** English WER after adapting Whisper to Kinyarwanda: full fine-tuning 105.50 vs LoRA 12.93 (base 11.09).
  - **Qian et al. (2024, arXiv:2407.06800, https://arxiv.org/abs/2407.06800).** Adding Irish: full fine-tuning 43.7 vs LoRA 54.8 on Irish, but Welsh collapses 27.5 → 96.8 under full fine-tuning.
  - **Xu et al. (2024, arXiv:2408.10680, https://arxiv.org/abs/2408.10680).** A merged LoRA still forgets: Aishell 8.82 → 24.31 (LoRA) vs 30.87 (full). Freezing the encoder "did not have a significant impact" on forgetting. Hence G1 is paired with A1 (LoRA scaling).
  - **Biderman et al. (2024, TMLR, arXiv:2405.09673, https://arxiv.org/abs/2405.09673).** Text-only: LoRA on all modules beats attention-only, and the best learning rates are 2e-4 to 5e-4.
- *Omnilingual-specific:*
  - Zero-shot 7B qxp CER is 1.32 on Meta's split, better than single-language fine-tuned CTC (1.47–1.81) (arXiv:2511.09690, §5.7.5). So acoustics are fine and the remaining gap is orthography and domain.
  - **Table 22:** the model was trained with the language code present 50% of the time. If it is fine-tuned with the code on 100% of samples and the code is then omitted, CV22 CER rises from 6.9 to 45.1. So always pass `qxp_Latn`.
- *Out-of-domain hallucination:* SLAM-ASR trained on LibriSpeech gives unbounded WER on CallHome because insertions spiral; LoRA on the LLM improved cross-domain WER from 60.4 to 22.5 (Kumar et al., 2025, ICASSP SALMA workshop, arXiv:2411.03866, https://arxiv.org/abs/2411.03866). In low-resource Danish, LoRA increased insertions by 10.8–18.4% (Kolluri, Kampouridis & Shekhar, 2026, arXiv:2603.27981, https://arxiv.org/abs/2603.27981).
- *Recipe:*
  - LoRA r=16 (α=32, dropout 0.05) on q, k, v, o and MLP of the LLaMA decoder; r=8 as a robustness control. Encoder and projector frozen.
  - Learning rate 1e-4 to 2e-4, tri-stage 0.1/0.4/0.5, 3–6k steps (3–5 epochs over 36 h plus filtered pseudo-labels from S1). Evaluate every 500 steps and average the last 5–10 adapter checkpoints.
  - Mix 10–20% of the batch with a replay set (other CV Quechua or Spanish clips, if allowed) to limit forgetting.
- *Expected:* read speech 25–40% rel. (the zero-shot gap is mostly Puno orthography); spontaneous 5–15% rel. (14.7 → about 12.5–14).
- *Cost:* about 8–12 GPU-h per run, 2–3 runs, so 25–35 GPU-h.
- *Risk:* medium. fairseq2 LoRA support for this model is community-made (arXiv:2607.12468), and the decoder can hallucinate (G2).

**G2. Decoding guards for the LLM decoder.** [general]
- *Reported:*
  - NTU (MLC-SLM 2025, arXiv:2506.13339): beam 4 with `no_repeat_ngram=5`. Repetitions affected under 0.05% of utterances but cost over 0.8 WER.
  - SHNU (arXiv:2507.03343, https://arxiv.org/abs/2507.03343): stripping repeated tokens at inference took 21.49 → 18.75.
  - RWTH: joint decoding of CTC with the prefix LLM, 6.13 → 6.05 (Schmitt, Zeyer et al., 2026, arXiv:2603.15045, https://arxiv.org/abs/2603.15045).
- *Recipe:*
  - Beam 5, length penalty tuned in 0.8–1.2, `no_repeat_ngram=5`, maximum output length 1.5× the CTC hypothesis length.
  - If the output loops or violates the length ratio, fall back to CTC+LM.
- *Expected:* 1–3% rel., much more on rare pathological clips.
- *Cost:* about 0.
- *Risk:* none.

**G3. Feed the CTC hypothesis into the LLM prompt.** [general; low priority]
- *Reported:* MLC-SLM 2025 winner TEA-ASLP, 14.52 → 14.26 (arXiv:2507.18051, https://arxiv.org/abs/2507.18051).
- Requires changing the omniASR-LLM input template and training for it. **Not worth the engineering in 8 days.**

**G4. Generative error correction (GER) and a ByT5 corrector.** [poor generalization; see §g.8]
- *Reported:*
  - **Hindi (Kumar, Krishna, Ramakrishnan & Jyothi, 2026, EACL short, https://aclanthology.org/2026.eacl-short.45.pdf).** Correctors trained on 63.5k 1-best/reference pairs from a strong system (18.02 WER): **ByT5 18.22 (worse)**, mT5 17.50. GPT-4o-mini in-context 18.02 → 25.14 zero-shot (20.89 with 5 shots). Marathi and Telugu got worse with ByT5 (25.55 → 26.32; 23.28 → 24.51).
  - **Multilingual (Li, Chen, Kwok, Chu & Chng, 2024, Interspeech, https://www.isca-archive.org/interspeech_2024/li24h_interspeech.pdf).** A Qwen-7B corrector trained on MMS hypotheses hurts when moved to another ASR system: Thai Whisper 7.49 → 11.47, Mongolian 45.79 → 52.77.
  - **N-best T5 (Ma, Gales et al., 2023, arXiv:2303.00456, https://arxiv.org/abs/2303.00456).** A 1-best corrector hurt (2.88 → 3.02); only constrained 10-best helped.
  - **Frisian (bib 52).** A fine-tuned open LLM leaves 97% of sentences unchanged.
- *Verdict:* skip for the primary system. External Quechua text is better used in D1/D2 LMs, or as a text-only decoder adaptation. Fang et al. (2025, arXiv:2506.05671, https://arxiv.org/abs/2506.05671) report target-domain WER 27.59 → 16.22 with LibriSpeech held at 3.73 → 4.36, using LoRA r64 at learning rate 5e-6 mixed with paired audio. That idea is optional, after G1.

### g.7 What winning shared-task systems (2023–2026) have in common

| Task | Winning recipe | Component deltas (reported) |
|---|---|---|
| Mozilla CV Spontaneous Speech ASR 2025 (qxp was an unseen Task-4 language) | **No public leaderboard or findings paper as of 2026-10-09** (https://community.mozilladatacollective.com/shared-task-mozilla-common-voice-spontaneous-speech-asr/). UW system (Bhandari, Kumar & Mulhern 2026): MMS/XLS-R/Whisper fine-tuning + KenLM. | XLS-R greedy → 4-gram: −19% to −27% rel. on several languages; LM helped 19/21 (https://aclanthology.org/2026.computel-1.5/) |
| Kaggle Bengali.AI Speech 2023 (CV-style noisy labels) | 1st: Whisper-medium, iterative pseudo-label cleaning, external pseudo-labelled data, punctuation ensemble. 2nd: IndicWav2vec CTC, drop the top ~10% WER training clips, 6-gram KenLM. 3rd: validated-first then WER < 0.75 data (https://www.kaggle.com/competitions/bengaliai-speech/discussion/447957) | 1st: about 0.380 after pseudo-label cleaning → 0.312 public LB after clip merging, YouTube pseudo-labels and punctuation ensemble (link in S1). 3rd: external 5-gram about −0.01; Demucs denoising −0.003 |
| MLC-SLM 2025 (Interspeech, LLM-ASR) | TEA-ASLP: dual encoder, CTC on connector, CTC hypothesis in prompt, MoE-LoRA, top-5 checkpoint averaging | 19.87 → 15.95 (dual encoder) → 14.52 (+CTC) → 14.26 (+CTC prompt) → 10.62 (+data/LLM) (arXiv:2507.18051) |
| NADI 2026 (Arabic dialect ASR) | Wifaq: 4× w2v-BERT-2.0 CTC + 5-gram KenLM + ROVER/MBR | 46.12 → 44.00 → 42.59 (A4) |
| MADASR, ASRU 2023 | Transsion: IndicWhisper fine-tuning + 20-gram BPE KenLM fusion | Bengali 16.81 → 14.85 (D1) |
| IWSLT 2025 que-spa | GMU: SeamlessM4T-v2 fine-tuning on Siminchik | que dev **CER 15.54 / WER 37.80**, the only Quechua ASR WER in IWSLT system papers (Meng & Anastasopoulos, arXiv:2505.21781, https://arxiv.org/abs/2505.21781) |

The common elements, in order of reported impact:
1. Label cleaning and filtered pseudo-labels.
2. An n-gram LM with external text.
3. Cross-architecture system combination.
4. Checkpoint or adapter averaging.
5. Speed and length augmentation.

No winner relied on test-time adaptation.

### g.8 Not recommended for generalization (benchmark-fitting or transductive)

| Technique | Why it is not generalizable | Evidence |
|---|---|---|
| Closed-set sentence snapping, or a heavily weighted scripted-sentence LM (§C2 last bullet; §(a) step 2(i)) | The official test reads **new** sentences (RESULTS.md). This memorises the CV prompt pool and transfers to no new text | Quechua: a bigger mismatched LM 65.73 → 66.81 (Liu et al. 2024); OOD n-gram losses −8% to −17% (Whisper-LM) |
| Tuning LM, ROVER or interpolation weights on a single in-domain dev set | Optimises for dev, not robustness. Instead pick weights that are good on read-unseen *and* spontaneous *and* codec-processed dev | Carvalho 2026: in-domain-optimal models lose OOD |
| "Imitate omniASR-7B because spontaneous refs may be silver" (§0 point 3 and the §(a) step 3 hedge) | Fits the annotation artefact, not the speech. Distil the 7B only through S1's agreement filter, which is justified by its accuracy (14.7 on *gold*) | — |
| Test-time adaptation (SUTA, SGEM, DSUTA), pseudo-labelling or continued pretraining on test audio | Transductive. Gains come from the test set itself and are forbidden in some challenges (MLC-SLM 2025). Use at most as a labelled contrastive run | S3: 0–20% rel. under heavy shift; about 1% on top of a strong combination (NTT CHiME-7) |
| Per-test-speaker adaptation (f-LHUC, speaker-deficiency TTA) | Needs test-speaker audio; small gains on top of strong models | Geng 2024; Hu 2024 (§g.1) |
| LLM generative error correction (open or API LLMs), ByT5/mT5 correctors | No capable Quechua LLM. Correctors overfit to one system's errors and break on others; risk of rewriting to Cusco/Ayacucho spelling | G4: ByT5 worse in Hindi, Marathi and Telugu; Thai 7.49 → 11.47 when moved to another system |
| Full fine-tuning of LLM-7B (vs LoRA) | Catastrophic forgetting of multilingual robustness | Whisper/Kinyarwanda: English 11.09 → 105.50 (full) vs 12.93 (LoRA) |
| TIES/DARE merging | Can damage untouched domains | FLEURS 9.63 → 14.03 (Carvalho 2026) |
| InterCTC at every layer, focal CTC, re-initialising top layers of the CTC checkpoint | Gains not robust across seeds or negative on SSL encoders | Wright 2023; Batra 2026; F5 |
| Accepting single-seed dev gains below ~3% rel. | At this data scale these are often noise | Batra 2026 |

### g.9 Ranked "do next" list for OUR setup

Gains are my estimates of relative WER reduction against the current best *relevant* system: CTC-1B greedy 17.5 read / 21.5 spontaneous, and LLM-7B zero-shot 14.7 spontaneous. Gains do not add up linearly. The GPU budget left is about 190 GPU-h; the list uses about 110–140.

| # | Technique (section) | Expected gain | GPU-h | Generalization value | Main risk |
|---|---|---|---|---|---|
| 1 | **Broad open-vocabulary LM decoding.** Interpolated character 8–10-gram (lexicon-free) + subword 6–8-gram + word 4-gram; train text plus Puno-respelled external text only; α/β tuned per domain on *both* dev sets (D1) | read 8–15%, spontaneous 3–8% | ~2 (mostly CPU) | High if text is broad; never uses dev/test sentences | Over-weighting in-domain text; smaller gain than XLS-R literature |
| 2 | **LoRA-tuned omniASR-LLM-7B.** r=16, all decoder linears, encoder frozen, `qxp_Latn` always, adapter averaging, LoRA-scale α sweep (WiSE-FT), repetition/length guards with CTC fallback (G1, G2, A1) | read 25–40% vs 7B zero-shot (orthography); spontaneous 5–15% (14.7 → ~12.5–14) | 25–35 | High: LoRA plus α-scaling keeps the zero-shot robustness | Hallucination on elderly/long clips; community LoRA tooling |
| 3 | **Agreement-filtered self-training on the 30 h spontaneous audio.** 7B (+CTC+LM) teacher, keep about 50–70%, re-initialise each round, 2 rounds, elderly-first curriculum, read-dev regression check (S1, E3) | CTC student spontaneous 10–20%, read 0–3% | ~26 | High: OOD unchanged with strict filtering (Pratinidhi) | 7B orthography in labels (respell first); confirmation bias if filter is loose |
| 4 | **Elderly/device augmentation folded into #3's retrain.** Speaker-rate-matched tempo 0.8–0.9 for non-elderly spontaneous speech, elderly oversampling, codec (Opus/AMR/MP3) p≈0.3, random silence trim, crop to ~9 s (E1, E2, E4) | elderly spontaneous 5–10% | +0–5 | Very high: targets the known test shift without test audio | Over-slowing artefacts; not measurable on clean dev |
| 5 | **Cross-architecture combination.** Confidence-weighted ROVER/MBR over CTC soup+LM, LoRA-7B and zero-shot 7B, with CTC length/repetition guards (A4) | 8–15% over best single system | 5–8 | High (errors are complementary across architectures) | Over-tuned system weights; keep them coarse |
| 6 | **Weight-space robustness for CTC.** EMA or last-k averaging in every run; 2–3 run greedy soup; WiSE-FT α sweep vs the omniASR-CTC start point, chosen on shifted dev (A1, A2) | 3–8% (more under shift) | 0 + 18–27 for extra runs (shared with #7) | Very high: the most direct OOD evidence (WiSE-FT, Ducorroy 2025, Carvalho 2026) | Members leaving the basin; check soup > best member |
| 7 | **CR-CTC + anchored fine-tuning schedule.** Frozen feature encoder, head-only warm-up, L2-SP, learning-rate sweep, 12–20k steps; 2 seeds; doubles as soup members (F1–F3) | 3–8% | ~15–20 | High (consistency and less forgetting are known to help OOD) | Unverified on SSL fine-tuning; needs 2-seed confirmation |
| 8 | **Second-pass rescoring.** 100-best from #1 rescored with a small character Transformer LM, then CER-loss MBR (D2) | 1–4% on top of #1 | ~3 | Medium–high (same broad text as #1) | Small gain; weight tuning noise |

**Suggested order:**
1. #1 on CPU now, alongside #2 on the GPU (D0–2).
2. Then #3+#4+#7 as a single retrain with 2 seeds, which also gives soup members for #6 (D2–5).
3. Then #6 and #8 (D5–6).
4. Then #5 as the final system (D6–7).

Keep everything in §g.8 out of the primary submission.

### g.10 RL / sequence-level training and byte-level post-correction

Compiled 2026-10-09 (~6 days left). Question: would sequence-level training of the LoRA-7B decoder (MWER/MBR, PPO, GRPO, GSPO, RLOO, DPO) or a ByT5 post-corrector improve the current system (joint_v3/v4: dev-unseen 9.6–9.9, itest-unseen 16.0–16.5, itest-spont 6.4–6.6, OOD 14.9–15.1)? Bibliography 144–168 are new; others are cited by existing number. Algorithms: PPO (Schulman et al. 2017), GRPO (Shao et al. 2024), RLOO (Ahmadian et al. 2024), DPO (Rafailov et al. 2023), GSPO (Zheng et al. 2025); ByT5 = Xue et al. 2022.

**R1. Classic MWER / MBR sequence training.** [general; small gains]
- *Reported:*
  - **Prabhavalkar et al. (2018, ICASSP, arXiv:1712.01818, https://arxiv.org/abs/1712.01818)** introduced MWER for attention models. The loss is the expected word errors over the N-best list, renormalised and mean-subtracted, plus a small CE term. N-best beams beat sampling. Gain: **up to 8.2% rel.** over CE on Google voice search (thousands of hours).
  - **Meng et al. (2021, Interspeech, arXiv:2106.02302, https://arxiv.org/abs/2106.02302).** MWER on a transformer transducer (30k h) gives **8.8% rel.** on average over 6 test sets (5.8% with shallow fusion).
  - **Shivakumar, Kolehmainen et al. (2024, arXiv:2409.16654, https://arxiv.org/abs/2409.16654).** MWER-trained speech-text LLM used as an **N-best rescorer**: up to **20% rel.** vs Whisper-large and 15% vs a text-only LLM rescorer. MWER "always improves" the rescorer.
  - **Low-resource evidence is thin.** The closest result is sequence-discriminative LF-MMI fine-tuning of a self-supervised model (Vyas, Madikeri & Bourlard, 2021, ICASSP, arXiv:2012.14252, https://arxiv.org/abs/2012.14252). It gives **4.3% rel.** on Babel Swahili (38 h) and **4.4%** on Tagalog (84 h), against 10–35% on LibriSpeech-100h. I found no MWER/MBR result for an LLM-ASR decoder on a low-resource language with fewer than 100 h.
  - The FunAudio-ASR report (2025, arXiv:2509.12508) notes that GRPO with a WER reward "becomes similar to MWER". The two are the same family, and MWER is the cheaper, deterministic member: beam N-best, no reference model, no sampling.
- *Typical gain:* 3–9% rel. on large, well-matched data. Gains shrink as the model gets stronger and the data gets smaller.

**R2. RL for speech LLMs and LLM-ASR (2024–2026).** [mostly hallucination/robustness gains]

| Paper | Model / data | Algorithm, reward, hyper-parameters | Reported effect |
|---|---|---|---|
| Shivakumar, Gu, Gandhe & Bulyko (2025, ASRU, arXiv:2509.01939) | LLaMA-3-style 2B/8B LLM-ASR, 88k h SFT; encoder frozen, projector + LLM updated (full) | GRPO; reward −WER (best overall), exact match, −edit count; G = 6/10 ("insignificant" difference); β = 0.04 (β = 0 slightly better); lr 1e-6; batch 64; ≤ 5k steps | In-domain **up to 18.4% rel.** (abstract); typical per-set 4–9% rel., MLS 8B slightly worse. Gains are mainly from **hallucination**: AMI-IHM 56.0 → 20.0, insertions 38.4% → 2.8%. Domain adaptation on AMI: SFT 19.2 vs GRPO 15.6, with less source-domain loss than SFT |
| Gao et al. (2025, arXiv:2509.18569) | FunAudio-ASR (0.7B encoder + Qwen2.5-7B) | GRPO; reward 1−WER + hallucination penalty (−1) + keyword P/R; G = 12; β = 0.1; lr 1e-5; 80k utterances; < 1 day | **5.3% rel.** (10.25 → 9.71 short, 6.35 → 6.03 long). The most useful data were **hard samples where systems disagree**. Long audio still showed insertions |
| FunAudio-ASR report (Tongyi Fun Team, 2025, arXiv:2509.12508) | Production LLM-ASR | GRPO-style RL stage, 100k utterances (5 × 20k subsets incl. hallucination cases), 8 × A100 < 1 day | **4.1% rel.** offline, 9.2% streaming |
| Qwen3-ASR (2026, arXiv:2601.21337; bib 14) | Production LLM-ASR | **GSPO** (Zheng et al. 2025, arXiv:2507.18071), about 50k utterances | Described as essential for "noise robustness, transcription stability"; **no ablation numbers** |
| Xiong et al. (2026, SLT, arXiv:2609.14455) | 7B speech LLM | GRPO + frozen 0.3B char wav2vec2-CTC "acoustic judge" reward | Text-only rewards let the model "guess from a strong language prior". The acoustic reward cuts insertions **28% / 22%** (AMI-IHM/SDM); WER 35.89 → 34.71 |
| Kumar, Labrak, …, Stolcke (2026, arXiv:2607.08409) | LLM-ASR, synthetic (TTS) banking speech only | GRPO, WER reward | 36.71 → 22.09 (40% rel. vs SFT); SFT→GRPO 45%. The gain is attributed to fewer insertions and better alignment, not new representations |
| Ye & Vickers (2026, Interspeech, arXiv:2607.02757) | Qwen2-Audio + LoRA, code-switched TTS speech, 10 language pairs | GRPO (RLVR); error-rate reward + script-fidelity reward | RL on **10% of the data matches LoRA-SFT on all of it**, mostly by removing translation/wrong-script errors |
| Nguyen Quang et al. (2026, Interspeech, arXiv:2605.23975) | 3 audio LLMs | **DPO**, 100k pairs / 570 h; rejected = mined failure modes | MER −89.6% in-distribution, **−20% out-of-distribution** (code-switching failures) |
| Ling & Ye (2025, arXiv:2506.11091) | ASR + LLM reward model, unlabeled data | RL with LLM-judge reward | Entity WER **21% better** than self-training |
| Nagpal et al. (2025, ICASSP, arXiv:2501.00039) | Speech LLM, disordered speech | RLHF with meaning/syntax rewards | Better than SFT for adaptation, but "does not outperform existing systems" |
| Zheng, H. et al. (2026, arXiv:2606.02615) | Auditory LLM | FSA-GRPO on 2k utterances (few-shot prompting) | 53.9% rel. on child speech **via in-context demonstrations**; not applicable to our decoder |

- *Pattern across all of them:*
  1. The large RL gains come from removing **behavioural failure modes**: insertions or hallucination, translation instead of transcription, wrong script, long-audio loops. These are found on large SFT models evaluated off-domain.
  2. In-domain gains on top of a good SFT model are **4–9% rel.**
  3. Group size barely matters (6 vs 10), and KL can be small or zero.
- *Low-resource:* **no paper applies GRPO, GSPO or DPO to an LLM-ASR for a genuinely low-resource language with about 25 h of real speech.** The nearest evidence is TTS-only code-switching (Ye & Vickers) and Tibetan text MT with semantic rewards (Su et al., 2026, ACL Findings, arXiv:2605.14366), which is not speech.
- *Negative or limiting results:*
  - Xiong et al.: WER-type rewards reinforce the LM prior. This is exactly our risk for suffix errors in agglutinative words.
  - Qwen3-ASR gives no ablation.
  - MLS-8B regressed slightly (Shivakumar 2025).
  - Nagpal et al.: RL did not beat strong existing systems.
- *Why our setup differs (my analysis):*
  1. **Reward variance.** About 2k distinct scripted sentences, each read many times and seen for several LoRA epochs. On training audio the LoRA is probably near-perfect (unmeasured; the gate in R4 checks this). If so, group samples are identical, all advantages are 0, and nothing is learned.
  2. **Pseudo-label rewards.** On the 27 h pseudo-labelled spontaneous audio, the reward is agreement with our own ROVER consensus. That is sequence-level self-distillation of the ensemble into one member. It removes the v1/v2/zero-shot diversity the joint rescoring profits from (§h: v1+v2 complementary).
  3. **Hallucination is already handled.** Our main hallucination mode was fixed by pause-based chunking plus G2 guards (§h). Our residual errors are 1-character substitutions inside long words (§0), the regime where GRPO papers report their smallest gains.

**R3. ByT5 or LLM post-correction for low-resource and agglutinative languages.** [harmful at our error level; see G4, §g.8]
- *Harms:*
  - **Hindi:** ByT5 18.02 → 18.22 even with 63.5k pairs (Kumar et al. 2026, bib 133). ByT5 helps only on character-level phenomena (transliteration, segmentation), which is why it is tempting for Quechua.
  - **Marathi and Telugu** also got worse with ByT5 (G4).
  - **Kazakh (agglutinative):** zero-shot LLM correction raises WER by +5.5 (Gemma-2-9B), +7.2 (Qwen2.5-7B) and +10.8 points (32B). The authors conclude "fine-tuning is essential … LLM correction consistently hurt" (Nyssanov, Rzayeva, Batkuldin & Kozhakhmet, 2026, *Information* 17(7):690, https://www.mdpi.com/2078-2489/17/7/690).
  - **Moving to another ASR system:** correctors hurt (Thai 7.49 → 11.47; Li et al. 2024, bib 134). GEC "struggle[s] to generalize beyond the specific types of errors encountered during training" (Ghosh et al., 2024, arXiv:2410.13198, https://arxiv.org/abs/2410.13198).
- *Gains:*
  - They appear in the **high-error regime**: Burmese seq2seq corrector 51.56 → 39.82 WER (Lin, Aung, Thu & Oo, 2025, iSAI-NLP, arXiv:2511.21088).
  - They appear with **very large synthetic error corpora**: Denoising-LM / specialised correctors reach 1.5/3.3 on LibriSpeech, and the authors stress that "matching the diversity of realistic error distributions is key" (Gu et al. 2024, bib 51). Their TTS→ASR pipeline is out of reach for Quechua in 6 days.
  - Frisian GER helps with contamination-aware checks (bib 52), but only with a capable LLM for that language.
  - Our errors are at 2–3% CER, the "low-error regime where LLMs struggle" (Gu et al.).
- *Memorisation risk:*
  - Training pairs would come from our ~2k distinct sentences.
  - The target side then holds only ~2k strings, so a byte-level seq2seq learns to **snap outputs to the prompt pool**. That is exactly the closed-set behaviour §g.8 rejects: the hidden test reads new sentences, and its median distance to the pool is 0.495 (§h).
  - Hypotheses decoded from training audio are near-perfect (the models saw it), so realistic error pairs require K-fold retraining.
  - External Puno-respelled text with synthetic character noise avoids the pool, but it duplicates what the char-10g LM already contributes to joint rescoring, and it must be scored by CTC anyway.

**R4. Recommendation for OUR setup.**
- **Do not run PPO, GRPO, GSPO, RLOO or DPO** for the submission.
  - Expected gain after joint rescoring: 0–2% rel. That is below the ~3% single-seed noise floor (Batra 2026, bib 112).
  - Cost: 1.5–2 days of fairseq2 engineering plus 5–15 GPU-h.
  - Risks: zero-advantage batches, reward-hacking toward the LM prior, and loss of ensemble diversity.
- **Do not train a ByT5 corrector** (R3).
- **Do instead (same N-best machinery, inference only).** The work-in-progress `src/qxp_asr/nbest.py` already returns all beams and their sequence log-probabilities.
  1. Add the **teacher-forced log P_LoRA(candidate | audio)** of every candidate (v1 and v2, and optionally zero-shot) as a feature in `scripts/joint_rescore.py`. Currently the LLM enters only as a 0/1 indicator (`mu * [LLM hyp]`). This is the missing attention term of Watanabe et al. 2017 hybrid CTC/attention scoring (bib 143), and the setting in which Shivakumar et al. (2024) get up to 20% rel.
  2. Add each LoRA model's **beam-8 N-best** as extra candidates.
  3. Optionally, pick the final output by MBR with a CER loss over the pooled list (Jinnai, 2025, arXiv:2510.19471, https://arxiv.org/abs/2510.19471: MBR beats beam search for Whisper in most settings).
  - Expected: 2–6% rel. on unseen/OOD (my estimate). Cost: ~2–4 GPU-h of scoring and an hour of weight tuning on both dev sets. Risk: low; tune coarse weights only (§g.8).
- **If a sequence-training run is still wanted (gated pilot, no more than 1 day, after the final submission files exist):**
  1. *Gate (≤ 1 GPU-h):* run beam-8 on 500 training clips (scripted, gold spontaneous, pseudo-labelled). Continue only if at least 30% of clips have a non-zero WER spread across their 8 hypotheses, and the mean oracle-vs-1-best gap is at least 2 WER points. Otherwise MWER has no signal.
  2. *Algorithm:* N-best **MWER** (Prabhavalkar 2018) rather than GRPO. Deterministic beams stay diverse when sampling collapses, and no reference policy is needed. Loss = E_Nbest[ΔR] + 0.01·CE on the reference (the CE term anchors the model, replacing KL). The GRPO-equivalent is G = 8, β = 0.02–0.04 to the SFT LoRA, temperature 1.0.
  3. *Reward / risk:* use −(0.5·WER + 0.5·CER), because character errors dominate our WER. Set the reward to −1 if the output is longer than 1.5× the CTC hypothesis or contains a 5-gram repeat (the Gao et al. 2025 style hallucination penalty).
  4. *Data:* gold spontaneous ×3 plus only the **highest-agreement** pseudo-labels (all-system agreement), plus 20% scripted to prevent drift. Never use dev or itest.
  5. *Optimisation:* continue LoRA v2 (encoder frozen) at lr 1e-5, batch 16 clips × 8 hypotheses, 300–800 steps. Evaluate every 100 steps on dev-unseen, dev-spont and OOD, and select on their mean.
  6. *Cost:* each step runs the 7B encoder once (no grad), beam-8 decoding and decoder forward/backward, about 3–6 s per step on the L40S, so about 1 GPU-h per 800 steps. The engineering takes most of the day.
  7. *Expected:* single-model 0–4% rel. (spontaneous and OOD insertions, if any). After joint rescoring 0–2%.
  8. *Risks:* no signal (gate); overfitting the ~600 gold clips; amplifying pseudo-label errors; reduced complementarity with v1 and zero-shot. Keep the run as a separate rescoring candidate rather than replacing v2.

### g.11 RL for ASR: breakthrough candidates and experiment plan

Compiled 2026-10-10 (~6 days and 2 × L40S left). This section extends g.10 with a wider 2023–Oct 2026 survey: preference optimisation, reward design, label-free and test-time RL, robustness, distillation-style alternatives, and CTC/rescorer training. It then turns the evidence into a ranked plan for OUR system. The full recipes and the run table are in `reports/rl_experiment_plan.md`.

**Conventions.**
- Bibliography 172–232 are new; papers cited earlier keep their number.
- Every arXiv ID was opened on 2026-10-10 and its title matched.
- "Reported" numbers come from the paper. "n/r" means the number could not be read.
- **[T]** = transductive (trains on test inputs). **[I]** = inductive (separate unlabelled training data).
- **[LR]** = low-resource-language evidence. **[HACK]** = reward hacking or collapse.
- Expected gains are **my estimates**.

**Baselines (`reports/RESULTS.md`).**
- hybrid_dev (dev-mode models): dev-unseen 9.11 / 1.62, dev-spont 13.47 / 2.39, OOD 13.82 / 2.31.
- hybrid_final: dev-spont 13.38 / 2.35, OOD 14.58 / 2.41.
- Single-model LoRA v2: 13.13 / 14.93 / 15.11.
- LoRA v2 + text-CPT: 14.20 / 14.27 (clips ≤ 39 s) / 14.27.

#### g.11.1 Offline preference optimisation (DPO family) for ASR and speech LLMs

| Paper | Setup | Reported effect | Lesson for us |
|---|---|---|---|
| Wang, Chen & Xu 2025, Interspeech, *SI-SDA*, arXiv:2507.20169 (bib 188) [I] | Qwen2-Audio, LoRA 16, lr 1e-5, 2 epochs; beam 10 → top-5 group. Label-free reward: attention "prompt-reliance" score. Mean-centred policy gradient | WER: CHiME-4 6.6 → **5.7**, TED-3 7.2 → 5.7, LS-babble 25.6 → 20.7; supervised SFT 5.0 / 5.2 / 14.7. Gains plateau after 2–5 h of unlabelled audio | **DPO on the same score: 6.7 on CHiME-4 (worse than zero-shot 6.6).** Best-hypothesis SFT: 6.6. The group-relative PG over the N-best is what works |
| Guo, Li, Zeng et al. 2026, arXiv:2603.24034 (bib 189) | Whisper-v3 encoder + Vicuna-7B LoRA. DPO in a *separate* LoRA (r 8, β 0.1, lr 1e-5), merged at scale γ = 0.25. Pairs: reference vs. own decoding, kept only if WER > 20% | TED-LIUM 3: 7.89 → 5.47 (SFT) → **5.17** (DPO) | **[HACK]** γ > 0.375 over-optimises (incoherent output). Hard negatives only; scale down the DPO adapter |
| Nguyen, Truong, Nguyen et al. 2026, Interspeech, arXiv:2606.06985 (bib 190) | Whisper-small LoRA 16. Weighted CE + 0.1 × InfoNCE over K = 5 "near-miss" negatives, filtered by acoustic margin, phonetic similarity ≥ 0.6, text distance ≤ 0.4 | CS-FLEURS 16.67 → 14.06; ViMedCSS 24.72 → 21.87 | The N-best gives only about 1.4 usable negatives per utterance. Unfiltered negatives were mixed. MWER was worse on the target tokens |
| Wu, Lu, Peng et al. 2025, AAAI, *BPO*, arXiv:2412.19005 (bib 191) | OWSM v3.1, DPO β 0.1, lr 2e-6. Rejected = output on masked/noisy audio, or LLM-rewritten homophone errors | How2 10.5 → 9.3; Ego4D 59.9 → 59.2 | Audio-side negatives (masked content words) helped on every set: a way to reward *acoustic grounding* |
| Cosgrove, Grace & Wang 2026, arXiv:2610.04004 (bib 192) | Qwen2-Audio-7B, LoRA 8, β 0.2. DPO + *conditional* term (same answer preferred given intact vs. perturbed audio) + anchor | Audio-QA accuracy 56.0 → 70.0 (DCASE 2025) | Dropping the anchor cost the most (71.4 → 64.0). Not ASR |
| Hussain, Neekhara, Yang et al. 2025, *Koel-TTS*, arXiv:2502.05236 (bib 193) | TTS DPO/RPO, β 0.01, lr 2e-7, 6 samples per prompt ranked by CER | CER 2.56 → 0.58 (with CFG) | **[HACK]** Ground truth as the chosen sample degenerated. Chosen and rejected must both be model samples |
| Asaria, Salomone & Gandhi 2026, arXiv:2606.18323 (bib 182) | TTS; best-of-N selected by ASR, distilled by LoRA | Single-pass distillation recovers **52–58%** of best-of-N's failure reduction (failure rate 0.199 → 0.096 SFT / 0.083 DPO, "a tie") | Offline DPO/IPO never beat plain SFT on the selected outputs (DPO 0.292, IPO 0.319 vs. SFT 0.264 in the sweep) |

**Generic failure modes, all directly relevant because our N-best pairs differ by one character:**
- **Smaug / DPO-Positive (Pal et al. 2024, arXiv:2402.13228, bib 194).** DPO lowers the likelihood of the *chosen* answer when the two completions are close in edit distance.
- **Likelihood displacement (Razin et al. 2025, ICLR, arXiv:2410.08847, bib 195).** Similar pairs cause displacement: a refusal rate fell 74.4% → 33.4%. Filtering out such pairs fixes it.
- **Offline vs. semi-online vs. online (Lanchantin et al. 2025, arXiv:2506.21495, bib 196).** Llama-3.1-8B Math500: seed 47.4, offline DPO 53.7, semi-online DPO 58.9 (generator refreshed every 100 steps), online DPO 58.7, GRPO 58.1.
- **DPO for NMT with MBR (Yang, Chen, Lin & Byrne 2024, NAACL, arXiv:2311.08380, bib 178).** Best-vs-worst pairs work and consecutive-rank pairs hurt. β 0.7; β 0.1–0.3 gave repetitive output.

**Verdict.** Offline N-best DPO is the weakest option for us. If preference training is used at all, it should be:
- semi-online (regenerate the pairs);
- anchored (DPOP / NLL);
- built on large-gap pairs;
- kept in a separate down-scaled adapter;
- or applied to *audio-conditional* pairs (same text, intact vs. masked audio) instead of text pairs (→ RL-6).

#### g.11.2 Reward design beyond WER

| Paper | Reward | Reported effect / hacking |
|---|---|---|
| Xie, Pang, Zhang et al. 2026, *Mega-ASR*, arXiv:2605.19833 (bib 197) | DAPO on a Qwen3-ASR SFT model. Multiplicative repetition gate × (1 − WER). Gate at WER 0.3: below it, a token-level reward that counts substitutions with character similarity ≥ 0.5 as "soft" errors (α = 0.4); above it, LCS + length match. Final 0.4 static + 0.6 dynamic | VOiCES / NOIZEUS: SFT 7.59 / 8.12 → plain-WER DAPO 7.62 / 7.98 → **7.35 / 7.64**. **[HACK]** With a plain WER reward RL gave no gain, and under heavy degradation the model hallucinated or dropped content |
| Bai, Chen, Chen et al. 2024, *Seed-ASR*, arXiv:2407.04675 (bib 198) | MWER + λ·CE on N-best; plain vs. keyword-weighted WER | WER 2.02 → 1.98 (plain) / 1.94 (weighted). **[HACK]** Plain WER cut context recall 80.63 → 75.34 |
| Hwang, Ryu & Sim 2023, Interspeech, *EDRL*, arXiv:2306.01789 (bib 199) | Token-level rewards from the edit-distance alignment (RNN-T), r_p 0.1, γ 0.95, RL weight 0.5 + RNN-T loss | LibriSpeech test-other 2.7 → 2.6; sequence MWER 2.9 (worse). **[HACK]** A constant −1 per error caused end-of-utterance hallucination; RL without the likelihood loss gave **100% WER (blank collapse)** |
| Ichihara, Jinnai, Morimura et al. 2025, TACL, *MO-GRPO*, arXiv:2509.22047 (bib 200) | Normalise each reward by its own variance before summing | **[HACK]** With several rewards, GRPO optimises the highest-variance one and ignores the rest |
| Fang, Xie & Liu 2026, *ASR-TRA*, arXiv:2603.05231 (bib 201) [T] | Whisper-tiny test-time RL, 4 samples at T 0.4–0.6; CLAP audio–text + LLM reward; episodic reset | LS test-other + 10 dB noise 32.71 → 28.64 (SGEM 30.22); L2-Arctic 32.11 → 28.21. **[HACK]** On confidently wrong clips, entropy-TTA (SUTA) reached 122.37 WER vs. 83.61 baseline |
| Wu, Han, Wang et al. 2026, Interspeech MLC-SLM workshop, arXiv:2607.08208 (bib 202) [LR-ish] | GRPO on Qwen3-ASR-1.7B; WER, or **CER for ja/ko/th**, plus hallucination, repetition and length penalties | tcpMER 23.70 (−6.83 vs. baseline). SFT gave most of the gain; RL added robustness (per-language RL share n/r) |
| Xiong et al. 2026 (bib 156, g.10) | Frozen CTC acoustic judge | Insertions −28% / −22%. We already have this reward (`verifier_reward` in `train_rl.py`) |

**Lessons for an agglutinative language.**
1. The reward must give *partial credit inside words*. A WER-only reward is flat across hypotheses that differ by one suffix character, so use −(WER + CER) or a Mega-ASR-style soft token reward.
2. Keep a likelihood/CE anchor. Every reported collapse (EDRL blank collapse, Novosad below) happened without one or with a vanishing reward gap.
3. Combine rewards with per-term variance normalisation (MO-GRPO). Gate repetitions multiplicatively instead of with decode-time repetition penalties, which hurt languages with productive reduplication.
4. I found no published morpheme-level reward for agglutinative ASR. A suffix-weighted reward would be new, but untested.

#### g.11.3 RL and self-improvement on unlabelled audio (label-free, consensus, test-time)

| Paper | T/I | Method | Reported effect / failure |
|---|---|---|---|
| Zuo, Zhang, Sheng et al. 2025, *TTRL*, arXiv:2504.16084 (bib 212) | T | Majority vote over 64 samples (train on 32), lr 5e-7 | Qwen2.5-Math-7B AIME24 12.9 → 40.2. "Lucky hit": only 37% of the voted labels were correct, yet 92% of rewards were correct. Fails with a wrong temperature or episode count |
| Zhang, Guo, Yang et al. 2026, Interspeech, *AQA-TTRL*, arXiv:2510.05478 (bib 225) | T | TTRL for audio QA; confidence weighting, multiple attempts to avoid zero-variance groups | Qwen2.5-Omni-3B +11.04%; 7B +4.42% |
| Shafayat, Tajwar, Salakhutdinov et al. 2025, *SRT*, arXiv:2505.21444 (bib 213) | I | Self-consistency reward, 16–32 samples | Reaches 0.20 vs. 0.33 for gold RL. **[HACK] Sudden total collapse** to a constant answer while self-reward rises. KL / lr / GRPO-vs-RLOO sweeps did not prevent it |
| Zhang, Zhang, Guan et al. 2025, *No Free Lunch*, arXiv:2506.17219 (bib 214) | I | Entropy and self-certainty rewards | Gains in about the first 20 steps, then below the start. **Instruction-tuned models degrade**: MATH500 63.6 → 49.6 by step 80 |
| Shao, Li, Xin et al. 2025, *Spurious Rewards*, arXiv:2506.10947 (bib 215) | I | Random / incorrect rewards | Qwen2.5-Math-7B +21.4 points with *random* rewards (vs. +29.1 correct) via GRPO clipping bias; Llama/OLMo get none → **always run a shuffled-reward control** |
| Yu, Su, Tao et al. 2025, *RESTRAIN*, arXiv:2510.02172 (bib 216) | I/T | Soft vote weights; if the top answer gets < κ = 3 of 16 votes, zero the rewards and add a uniform penalty; down-weight low-consensus prompts | AIME25 8.3 (TTRL) → 17.9 (gold GRPO 20.8). TTRL collapses after about 50 steps; RESTRAIN stays stable |
| Zhang, Zhu, Ge et al. 2026, ICLR, *Co-rewarding*, arXiv:2508.00410 (bib 217) | I | Pseudo-label from a slowly updated EMA teacher (0.99 → 0.9999), G 8, lr 3e-6 | Self-vote baselines collapse on *training* data. GSM8K Qwen3-8B 94.01 vs. 87.19 for gold-reward GRPO |
| Liu 2026, *LLMs as a Jury*, arXiv:2607.10139 (bib 218) | I | Consensus of *independently trained* models as the reward / rejection filter | Beats self-consistency; limited by errors all models share (single author, not peer reviewed) |
| Zhao, Kang, Feng et al. 2026, ICLR, *INTUITOR*, arXiv:2505.19590 (bib 219) | I | Self-certainty reward, G 7, lr 3e-5, β 0.005 | Qwen2.5-3B MATH500 54.4 → 61.2 (gold GRPO 63.6), better OOD. **[HACK]** Scoring the reward with a frozen copy of the model was hacked (output padding) around step 100 |
| Agarwal, Zhang, Yuan et al. 2025, *EM-RL*, arXiv:2505.15134 (bib 220); Prabhudesai, Chen, Ippoliti et al. 2025, *RENT*, arXiv:2505.22660 (bib 221) | I / mostly T | Entropy minimisation as the reward | EM-RL MATH500 70.8 vs. 71.8 gold. Fails with weak bases, biased confidence, or distribution shift. RENT acknowledges confident-wrong reinforcement |
| Roy et al. 2025 (bib 222); Yu, He, Liang et al. 2026, ACL Findings, *DDRL* (bib 223); Lin, Kuai, Xue et al. 2026, "extinction window" (bib 224) | T | Analyses of TTRL failure | Label-free RL hurts weak bases. Medium-consistency items are the main source of reward noise. Once a wrong answer holds a majority, the correct one goes extinct (vote share about 58% → 0) |
| Deng, Xie, Hu et al. 2026, Interspeech, arXiv:2606.16546 (bib 90) | I | Not RL: confidence-ordered, per-speaker curriculum pseudo-labelling for **elderly** ASR (Whisper-medium) | DementiaBank Pitt 23.62 → 21.50 (supervised 20.68); confidence order beat random order |

**Synthesis.**
- (a) Self-consensus rewards collapse on unlabelled *training* data too, not only on test data. The reference must come from **independent, frozen** systems (our hybrid rescorer is exactly that), be gated by agreement, and be stopped early on gold dev.
- (b) Entropy and self-certainty rewards are unsafe on an already fine-tuned model.
- (c) No paper applies RL or preference optimisation to elderly-speech ASR, or to a genuinely low-resource language with real speech (also g.10). A positive or negative result here would be new.

#### g.11.4 Robustness, generalisation, forgetting and data selection

- **SFT Memorizes, RL Generalizes (Chu, Zhai, Yang et al. 2025, ICML, arXiv:2501.17161, bib 226).** On Llama-3.2-Vision-11B, OOD V-IRL-Language went 80.8 → 91.8 with RL and 80.8 → 1.3 with SFT. SFT is still needed first to fix the output format.
- **The ASR analogues (g.10) agree.**
  - GRPO beats SFT on domain shift: AMI-SDM 87.88 → 31.98 vs. 44.34 (bib 148).
  - SFT→GRPO trained on TTS speech and tested on real speech: 36.71 → 20.21 (bib 157).
- **Forgetting.** On-policy data, not the algorithm, explains why RL forgets less.
  - *RL's Razor* (Shenfeld, Pari & Agrawal 2025, arXiv:2509.04259, bib 227): forgetting is predicted by forward KL on the new task, and SFT on RL-generated outputs matches RL.
  - *Retaining by Doing* (Chen, Razin, Narasimhan & Chen 2025, arXiv:2510.18874, bib 228): Llama-3.1-8B forgetting 27.8% (SFT) vs. 3.4% (RL). Regenerating SFT targets with the current model each epoch recovers most of the benefit.
  - *RL finetunes small subnetworks* (Mukherjee, Yuan, Hakkani-Tür & Peng 2025, NeurIPS, arXiv:2505.11711, bib 229): RL updates only 5–30% of parameters, consistent with a rank-16 LoRA being sufficient.
- **Data selection for RL.**
  - *DAPO* (Yu, Zhang, Zhu et al. 2025, arXiv:2503.14476, bib 230): dynamic sampling (drop all-correct / all-wrong groups) is the largest single step, AIME +8 (to 50 overall).
  - *Online difficulty filtering* (Bae, Hong, Lee et al. 2026, EACL, arXiv:2504.03380, bib 231): keep pass rates 0.3–0.7; MATH500 57.2 → 64.6 in half the steps.
  - FunAudio's RL used "hard samples where systems disagree" (bib 149).
  - **For us, "difficulty" = disagreement among our systems on a clip.** That is already computed by the hybrid pool.

#### g.11.5 Distillation-style alternatives that capture RL gains cheaply

- **Expert iteration / reward-ranked fine-tuning.**
  - ExIt (Anthony, Tian & Barber 2017, NeurIPS, arXiv:1705.08439, bib 172): a slow search-based "expert" generates targets for a fast "apprentice", and the improved apprentice in turn strengthens the expert.
  - Text variants: ReST (Gulcehre et al. 2023, arXiv:2308.08998, bib 173), ReST-EM (Singh et al. 2024, TMLR, arXiv:2312.06585, bib 174), STaR (Zelikman et al. 2022, NeurIPS, arXiv:2203.14465, bib 175), RAFT (Dong et al. 2023, TMLR, arXiv:2304.06767, bib 176).
  - **Our hybrid rescorer is the expert; LoRA v1/v2/tc and the CTC model are the apprentices.**
- **MBR and QE fine-tuning (Finkelstein, Naskar, Mirzazadeh et al. 2024, ICLR, arXiv:2309.10966, bib 177).** en→de COMET: base 57.79; fine-tuning on *beam* self-labels 57.08 (worse); MBR-FT 58.02; QE-FT 59.63; reference-FT 60.95. Self-MBR on top of reference-FT 61.49; a stronger-teacher MBR 63.62. **The teacher must be stronger than plain beam search.**
- **Chasing COMET (Guttmann, Pokrywka, Charkiewicz & Nowakowski 2024, EAMT, arXiv:2405.11937, bib 232) [LR, en–Hausa].** Iterated MBR self-training gains BLEU only in round 1 (neural metrics peak around round 3) → 1–2 rounds.
- **STAR (Hu, Chen, Yang et al. 2024, NeurIPS, arXiv:2405.14161, bib 179).** Whisper-large-v3 adaptation from < 1 h of unlabelled audio with confidence/attention token reweighting, dropping the worst 20%: −13.5% rel. on average over 14 domains. CHiME-4: vanilla self-training 6.9 (worse than 6.8) vs. STAR 6.0.
- **O-1 (Baskar, Rosenberg, Ramabhadran & Audhkhasi 2023, Interspeech, arXiv:2308.07486, bib 180).** Boosting the oracle hypothesis closes 80% of the gap to oracle WER (EMBR: 43%).
- **Pseudo2Real (Lin, Liang, Su et al. 2026, ACL Findings, arXiv:2510.08047, bib 181).** A task vector (gold − pseudo) corrects systematic pseudo-label bias, λ 0.2–0.3. Large gains for Whisper-tiny; near zero for a large teacher.
- **On-policy distillation (OPD).**
  - GKD (Agarwal, Vieillard, Zhou et al. 2024, ICLR, arXiv:2306.13649, bib 183).
  - Thinking Machines blog (Lu 2025, bib 184): per-token reverse KL on student samples; AIME'24 70% with 1,800 GPU-h vs. 17,920 for RL.
  - **ASR:** Lin, Wang, Cai & Zeng 2026 (arXiv:2605.28139, bib 185): a 0.6B student, LibriSpeech test-other 7.17 → 5.50 (4.56 with a teacher-data stage); no off-policy KD ablation, single seed.
  - **Speech LLM:** X-OPD (Cao, Fu, Yu et al. 2026, Interspeech, arXiv:2603.24596, bib 186): average gap to the text model 11.29% → 3.43% with OPD, while SFT (22.76%) and offline KD (19.62%) *degraded* the strong base.
  - **Condition:** OPD helps only if teacher and student are compatible *and* the teacher has a real capability gap (Li, Zuo, He et al. 2026, arXiv:2604.13016, bib 187).

#### g.11.6 Sequence training of CTC models and learned rescorers/selectors

- **CTC + RL is a trap at our scale.**
  - **Novosad 2026, arXiv:2606.23306 (bib 203).** [HACK] All four MWER configurations on CR-CTC collapse (+6.18 to +8.90 points WER). The *training* oracle gap was 0.007 points, so there was no usable signal. Plain CTC drifted 7.07 → 7.31. RAFT on the oracle hypothesis diverged (4.86 → 65.5).
  - Novosad also finds that CTC-only N-best scoring is no better than greedy decoding (11 strategies, p > 0.05). An external LM via MBR gave −9% rel.
  - Older large-data results are positive: policy-gradient CTC 4–13% rel. (Zhou, Xiong & Socher 2018, arXiv:1712.07101, bib 204); Align-With-Purpose's auxiliary ranking loss keeps the CTC loss intact, −4.5% rel. (Segev et al. 2024, ICLR, arXiv:2307.01715, bib 205).
  - **For our CTC-1B, which has seen every training sentence, do not train with MWER.** Improve it through better *labels* instead (RL-1b).
- **Learned rescorers: the gains come from new features and discriminative scorers, not from a fancier combiner.**
  - **Wu, Chen & Gandhe 2022, Interspeech** (https://www.isca-archive.org/interspeech_2022/wu22_interspeech.pdf, bib 206):
    - LambdaMART over *score-only* features: LibriSpeech 1.92 / 4.11 → 1.90 / 4.07 (about 1% WERR).
    - Adding a listwise BERT confidence feature: 1.78 / 3.70.
    - Alexa (280 h of N-best training data): tuned-weight baseline 0.20%, LambdaMART 3.37%, with listwise confidence features 5.16%, oracle 20.24%.
  - **MWER-trained LLM scorers beat likelihood scorers by 5.9–8.5% rel.** (Shivakumar, Kolehmainen, Gu & Gandhe 2023, ASRU, arXiv:2310.06248, bib 208). GPT-2 likelihood 4.77 / 11.19 → MWER 4.52 / 10.89; BERT + MWER 4.36 / 10.93. Also RescoreBERT (Xu et al. 2022, ICASSP, arXiv:2202.01094, bib 207).
  - **LoRB (Yu, Yang, Kolehmainen et al. 2023, ASRU, arXiv:2309.15223, bib 209).** LoRA + MWER rescoring 4.50 / 10.81 vs. full fine-tuning 4.37 / 10.80. **[HACK]** MWER-LoRA degrades representations and OOD; their correlation regulariser partly fixes it.
  - **Kang, Van Gysel & Siu 2024, Interspeech, arXiv:2406.08207 (bib 210).** A rescorer that sees the whole N-best is flat alone (5.78 → 5.72) and helps only when interpolated with the ASR scores (5.46).
  - **[LR] Li & Niehues 2025, Interspeech, arXiv:2505.20445 (bib 211).** A generic Llama-3-8B rescoring **Kichwa** (Quechuan) MMS 10-best: 17.31 → 17.00. A general LLM prior adds almost nothing for Quechua, so our in-domain char-LM and LoRA scorers stay the core.

#### g.11.7 Where a breakthrough could come from in OUR system

Five observations drive the plan:

1. **Our real "policy" is the search-and-rescore system, not any single model.** hybrid_dev beats the best single model by 3–31% rel. (dev-unseen 13.13 → 9.11, dev-spont 14.27 → 13.47 on clips ≤ 39 s, OOD 14.27 → 13.82). Every member is trained only with token CE on its own data. **Expert iteration** feeds the expert's decisions back into the apprentices, which then improve the expert. It is the best-evidenced route (MBR/QE-FT, ReST, ExIt, Asaria 52–58% recovery). It needs no reward model, and it uses the 57 h of *non-test* unlabelled, elderly-rich audio.
2. **The zero-advantage problem is real, and it is the root cause of most failures above.** On training audio our models are near-perfect, so groups have no reward spread; that is Novosad's collapse condition. Usable signal exists only:
   - on unlabelled audio (with independent-system references);
   - on augmented "hard" versions of gold audio (tempo, codec, noise, which also simulate elderly speech);
   - in candidate pools that include *other* systems' errors (zero-shot, CTC).
3. **Our errors are one-character suffix errors.** Rewards and losses must give credit at the character or token level: −(WER + CER), soft token rewards, or token-level OPD. Not sequence-level WER.
4. **The selector has unused headroom only if new information enters it.** Score-only learned combiners give about 1% (Wu 2022). The gains come from:
   - listwise consensus features (MBR risk, number of systems proposing a candidate);
   - discriminatively (MWER) trained scorers.
5. **The decoder's sentence prior is a known liability** (2k training sentences, a hidden test of new sentences). Losses that reward *acoustic grounding* target the dev-unseen gap directly; RL with text-only rewards reinforces the prior (Xiong 2026). An audio-conditional preference loss is the one preference method with a clear mechanism for our failure mode.

#### g.11.8 Ranked experiment plan

**Rank** = expected value × probability, my estimates. "Δ hybrid" is the expected relative WER change of the final hybrid on the clean sets (dev-spont, OOD), and on dev-unseen for dev-mode runs.

**Gate G0 (CPU, ~1 h; run first; a new small script, e.g. `scripts/oracle_gap.py`).** On the existing dev candidate pools (`pool_dev`, `pool_dev_tc`), compute:
- pooled oracle WER/CER vs. hybrid, per set: the headroom for RL-2 and RL-3;
- the share of clips where hybrid 1-best ≠ LoRA v2 1-best: the signal for RL-1, RL-4 and RL-7.

On 500 training clips, compute the reward spread of beam-8 groups on clean vs. augmented audio: the signal for RL-3 and RL-5.

**Memorisation check for the unlabelled pool.** Every fine-tuned member has already been trained on the pool's audio with labels: v1 on silver, v2/tc/CTC v3 on pseudo_v1. Compare:
- WER(LoRA v2 1-best, pseudo_v1) on pool clips;
- WER(LoRA v2 1-best, hybrid 1-best) on dev_spont.

If the pool agreement is much higher, the hybrid mostly re-emits the old labels there. RL-1 and RL-7 then need the **cross-fit variant**: two LoRA folds split by pool speakers, each pool half labelled by a hybrid of zero-shot + CTC + the *other* fold's LoRA (+10 GPU-h). The ~2.4 h the pseudo-label filter rejected never entered training as pseudo labels, though v1 saw some of it as silver. Those clips are also the hard, high-disagreement ones (DAPO / FunAudio selection), so they are prime RL-1/RL-7 material.

If the pooled oracle is within 10% rel. of hybrid, drop RL-2 and RL-3.

| Rank | ID | Experiment | T/I | Δ single model (est.) | Δ hybrid (est.) | P(success) | GPU-h | Main risk |
|---|---|---|---|---|---|---|---|---|
| 1 | RL-1 | **Expert iteration: distil the hybrid rescorer into a new LoRA (v3) and the CTC (v4)** via hybrid-selected, margin-gated labels on the 57 h unlabelled pool (ReST/MBR-FT style) | I | dev-spont −4 to −8%, OOD −2 to −5% | −1 to −4% | 0.55 | 20 (+12 CTC v4; +10 if cross-fit) | Members memorised the pool's old labels (G0 check → cross-fit); ensemble diversity loss (add v3, do not replace); shared errors confirmed |
| 2 | RL-2 | **Listwise consensus/MBR features + MBR selection in the hybrid rescorer** (selector "RL" done as a feature/decision change; linear weights kept) | I | — | −1 to −3% | 0.5 | ~0 (CPU 2 h) | Overfitting dev with more weights; keep ≤ 6 |
| 3 | RL-3 | **MWER-trained discriminative LLM scorer** over *pooled* candidates (CTC + zs + LoRA beams) on augmented gold audio, CE-anchored, used as a new `lp_mw` feature | I | n/a (scorer) | −1 to −4% | 0.35 | ~10 | Learns the 2k-sentence prior (check dev-unseen); MWER-LoRA OOD loss (LoRB) |
| 4 | RL-5 | **Gold GSPO/MWER with hard (augmented) rollouts and a character-level reward** (−(WER + CER), repetition/length gate, MO-GRPO normalisation, DAPO dynamic sampling); extends exp11 | I | spont/OOD −1 to −4% | 0 to −2% | 0.3 | ~10 | Over-generation on a strong base (watch the length ratio); spurious-reward artefact (run a control) |
| 5 | RL-6 | **Acoustic-grounding preference loss**: same transcript, intact vs. masked audio (conditional DPO with anchor), added to the v2 SFT recipe | I | dev-unseen −2 to −5%, OOD 0 to −3% | −1 to −2% | 0.25 | ~8 | No ASR precedent for this exact loss; may hurt read speech, where the prior helps |
| 6 | RL-4 | **On-policy distillation from the multi-adapter ensemble** (log-linear mix of zs/v1/v2/tc next-token distributions with hybrid weights; per-token reverse KL on student rollouts; unlabelled + gold) | I | −2 to −5% | 0 to −2% | 0.3 | ~12 (+1 day engineering) | Teacher–student gap too small (same family) |
| 7 | RL-7 | **Consensus-reward GRPO on the unlabelled elderly pool**: reward vs. *frozen* hybrid labels from RL-1, RESTRAIN-style gate, CTC-verifier term, length guard, 50% gold; the on-policy counterpart of RL-1 | I | 0 to −3% beyond RL-1 | 0 to −1% | 0.2 | ~8 | Collapse (SRT, No Free Lunch); consensus favours deletions |
| 8 | RL-8 | **TTRL-style transductive adaptation** (RL-7 run on hidden-test audio, rewards from the hybrid consensus on test clips) | **T** | — | 0 to −5% (contrastive only) | 0.3 | ~4 | Not generalisable; **only as a labelled contrastive run**, never in the primary submission (§g.8) |

**Decision rule** (any training run):
- Accept only if the mean clean WER (dev-spont, OOD; plus dev-unseen for dev-mode runs) improves by ≥ 3% rel. (the Batra noise floor, bib 112), no set regresses by more than 0.3 abs, and the gain survives inside the hybrid.
- For every RL run that shows a gain, repeat it with shuffled rewards (Spurious Rewards control).
- Winners are retrained in final mode (all labelled data except dev) and enter the hybrid with dev-mode weights. Spontaneous weights are retuned on dev_spont, as in exp12.

**Not recommended** (evidence above):
- Offline N-best DPO / IPO / KTO on text pairs (likelihood displacement for 1-character pairs; SI-SDA, Asaria, Lanchantin).
- MWER/RL on the CTC model (Novosad; EDRL blank collapse).
- Entropy or self-certainty rewards (No Free Lunch; ASR-TRA's 122 WER on confident errors).
- LambdaMART over the existing score features (about 1%, Wu 2022).
- LLM-judge or semantic rewards (no capable Quechua LLM: Kichwa 17.31 → 17.00).
- Decode-time repetition penalties (reduplication).
- Rewards computed from the policy's own majority vote (collapse; SRT, Co-rewarding).

**Paper-ready summary.** "Sequence-level RL for LLM-ASR mainly removes behavioural failures (hallucination, wrong script) on large SFT models; on a strong low-resource system whose errors are single characters inside agglutinative words, and whose training audio it already transcribes almost perfectly, rewards carry little signal. We therefore treat the hybrid CTC + multi-LLM rescorer as an 'expert' and distil its decisions on untranscribed, elderly-rich audio back into its members (expert iteration), and we test RL only where a reward signal exists: on independent-system consensus for unlabelled audio, on acoustically perturbed gold audio, and for the selector itself."

---

## (g′) Novelty check for two research directions (2026-10-10, agent web search; not yet re-verified by hand)

**A. Prompt-sentence leakage in Common Voice ASR evaluation.**
- **Official splits.** CorporaCreator keeps one recording per sentence, so its splits are sentence- and speaker-disjoint. Re-splitting `validated.tsv` (train = validated minus dev/test) leaks other speakers' recordings of the test sentences into training.
- **Known only informally:** CV v1 overlap called "bogus" in Kaldi issue #2141 (2017), https://github.com/kaldi-asr/kaldi/issues/2141; a 2026 GitHub issue (setsosie/ssl-vs-btm-asr #34) names the mechanism but gives no numbers.
- **Related but different:** pretraining-text contamination of LLMs (Tseng, Parcollet, van Dalen, Zhang, Bhattacharya 2025, arXiv:2505.22251 [233]); LM-text/test overlap of 77–80 % for n-gram LMs (de Zuazo et al. 2025, Whisper-LM, arXiv:2503.23542 [234]); Swiss German benchmark contamination (Akeret 2026, arXiv:2606.07608 [235]).
- **Verdicts:**
  - Quantifying the seen/unseen-sentence WER gap across many CV languages: **novel**.
  - Whether CTC + n-gram LM and speech-LLM decoders differ in how much they exploit memorised sentences: **novel**.
  - A survey of non-official splits in published low-resource CV work: **none found**.
- **Note:** the organizers' benchmark paper ([2], 70/25/5 split; no speaker or sentence constraint stated) would be an affected example. Frame this neutrally.

**B. CTC alignment likelihood as a reward or verifier for speech-LLM RL.**
- **Closest:** Xiong et al. (SLT 2026), "Grounded in Sound", arXiv:2609.14455 [236]. A frozen wav2vec2-CTC length-normalised loss is added to the WER reward in GRPO for a 7B speech LLM (English, LibriSpeech): AMI insertions −22 to −28 %, SDM WER 35.89 → 34.71. It **requires references**.
- **Label-free ASR RL uses other rewards:** ASR-TRA (Fang, Xie, Liu, AAAI 2026, arXiv:2603.05231 [237], audio–text semantic reward, test-time); LLM-scored rewards (Ling & Ye 2025, arXiv:2506.11091 [238]).
- **Hallucination studies** do not use CTC scores as detectors (arXiv:2501.11378 finds Conformer-CTC hallucinates far less than Whisper [239]).
- **Verdict: partially covered.** Open gap: a reference-free CTC verifier on unlabelled audio for a low-resource language (and elderly speech).

**Elderly speech in Indigenous/low-resource ASR:** essentially no work found (Cook Islands Māori, LREC 2022, includes elderly speakers without an age breakdown); none for any Quechua variety.

### g.12 Transferring LLM RL post-training techniques to ASR

Compiled 2026-10-10 (~5 days, 2 × L40S left). This section asks which techniques from 2024–2026 LLM post-training reports and RL papers could transfer to OUR system, and whether each has already been applied to ASR or speech LLMs. The reports covered are Zhipu GLM-4.5 and slime, DeepSeek-R1 and V3.2, Qwen3, Kimi k1.5 and K2, and MiniMax-M1. The algorithm papers cover DAPO, Dr. GRPO, VAPO, IcePop, CISPO, entropy control, pass@k objectives, mixed-policy RL and self-distillation.

**Scope and conventions.**
- Already covered in g.10–g.11, and only cross-referenced here: GRPO/GSPO/MWER basics, DAPO dynamic sampling, online difficulty filtering, on-policy distillation (GKD, Thinking Machines, ASR-OPD, X-OPD), RL's Razor / Retaining by Doing, expert iteration (ReST, RAFT, MBR-FT), label-free/TTRL rewards, and the GLM-5 masked trust region already implemented in `train_rl.py`.
- Bibliography 240–284 are new. Every arXiv ID below was resolved through the arXiv API on 2026-10-10 and its title matched. Numbers were read from the abstract or the HTML full text; "n/r" means not read.
- "ASR prior art" means a published ASR or speech-LLM paper that uses the technique. "None found" means three or more targeted web searches returned nothing.
- All effects on OUR system are **estimates**. GPU-h means L40S hours.
- About 2 GPU-h per 400-step `train_rl.py` arm (exp20 setting: 4 clips × group 4, 5 evaluations) is also an estimate, from the g.10 figure of 3–6 s per step.

**What OUR trainer currently does (relevant to the transfers).** `scripts/train_rl.py`:
- builds groups from **beam search**, so they are deterministic and not samples from π;
- standardises advantages per group, (r − mean)/std;
- uses a **length-normalised sequence log-ratio** for GSPO, which makes the gradient on hypothesis i scale with A_i/|o_i|;
- or uses N-best-posterior MWER;
- adds a CE anchor on gold.

Our residual errors are 1–2 characters inside long agglutinative words, so the hypotheses in a group share long prefixes.

#### g.12.1 Pipelines and stage design (multi-stage reports)

**T1. Multi-stage cold-start → RL → rejection-sampling SFT → RL (DeepSeek-R1 [244]; Qwen3 [246]; GLM-4.5 [240]; DeepSeek-V3.2 [245]).**
- *What it is.* The common pipeline has four stages.
  1. A small SFT cold start.
  2. Reasoning RL. R1 adds a language-consistency reward, at a "slight degradation" in accuracy.
  3. Rejection-sampling SFT. R1 uses about 800k samples (reasoning + non-reasoning, 2 epochs), regenerated from the RL checkpoint and filtered.
  4. A second RL stage. R1 runs 1,700 steps and adds model-based preference rewards only in the last 400 steps, to limit reward hacking.
- *Variants.*
  - **GLM-4.5** (355B/32B active): trains domain *expert* models first, then unifies them by self-distillation into one model.
  - **V3.2:** specialist distillation, then one mixed RL stage.
  - **Qwen3:** cold start, then reasoning RL on only **3,995 query–verifier pairs**, then thinking-mode fusion, then general RL.
- *ASR prior art.* Production LLM-ASR uses only SFT → RL: FunAudio [149], Qwen3-ASR (GSPO, [14]), NIM4-ASR (GRPO, [280]), Mega-ASR (DAPO, [197]), streaming ASR with a joint accuracy + latency reward (GRPO, WER −30.8% rel. at 80 ms lookahead, [284]). **No ASR paper regenerates SFT data from an RL-trained checkpoint** (RS-SFT after RL).
- *For OUR system.*
  - Our exp17 (RL-1) is already the "RS-SFT" stage, but its generator is the CE-trained ensemble.
  - The R1 transfer is to put the **RL-trained member** from exp20 (if it wins) into the candidate generators and scorers of `17_expert_iteration.sh` step 1, relabel the pool, and train a fresh LoRA.
  - Flags: none new; add `--adapter rl=<exp20 best>` to the `llm_candidates.py` calls.
  - Cost: about 20 GPU-h (same as exp17).
  - Expected (est.): +0 to −2% rel. on top of RL-1 for the hybrid.
  - Risks: it only makes sense if exp20 shows a gain above noise. The RL member may narrow N-best diversity (T9).
- *Qwen3's 3,995 pairs* support our premise that RL needs few but *informative* prompts. They argue for selecting clips (T7), not for adding more of them.

**T2. Expert models → unified model (GLM-4.5 Stage 1/2 [240]).**
- Our adapters (zs, v1, v2, tc, v3) are de facto experts, and the hybrid is their "unification" at inference.
- Distilling them into one model is RL-4 (on-policy distillation, g.11.5) and is not repeated here.
- *Note:* Qwen3's own table [246] shows on-policy distillation beating RL on Qwen3-8B at 1/10 of the compute: AIME'24 **74.4 vs 67.6**, AIME'25 **65.5 vs 55.5**, **1,800 vs 17,920 GPU-h**. That supports RL-4 only if a real teacher–student gap exists (g.11.5).

#### g.12.2 Policy-gradient estimators and loss aggregation

**T3. Token-sum aggregation without std normalisation (Dr. GRPO [250]; DAPO token-level loss [230]; GLM-4.5 "token-weighted mean loss" [240]; LUSPO [282]).**
- *What it is.*
  - **Dr. GRPO** removes GRPO's 1/|o_i| length normalisation and its per-group std normalisation. The first biases updates toward longer *incorrect* outputs; the second over-weights groups that are too easy or too hard. Result: 43.3% AIME'24 with a 7B base in about 27 h on 8 × A100.
  - **DAPO's** progressive ablation on Qwen2.5-32B, AIME'24: naive GRPO 30 → overlong filtering 36 → clip-higher 38 → soft overlong punishment 41 → token-level loss 42 → dynamic sampling 50.
  - **GLM-4.5** found a token-weighted mean "highly beneficial compared to conventional sequence-mean loss" for code RL (no number given).
  - **LUSPO** shows that **GSPO's** sequence-level ratio carries the same length bias and corrects it (numbers n/r).
- *ASR prior art.* **None found.** No ASR or speech-LLM RL paper ablates loss aggregation or advantage normalisation. NIM4-ASR [280] and Shivakumar et al. [148] use sequence-level GRPO; Mega-ASR [197] uses DAPO without reporting this component.
- *Why it matters more for ASR than for maths (my analysis).*
  - **The cancellation.** In a beam group, all hypotheses share the prefix up to the first divergent token. Take a token-sum loss with mean-centred advantages: Σ_i A_i Σ_t log π(o_{i,t} | o_{i,<t}). The gradient on a prefix token shared by a set S of hypotheses is (Σ_{i∈S} A_i)·∇log π. That is **exactly zero for the prefix all hypotheses share**, and equals the summed advantage of the branch at each fork. This is "free" token-level credit assignment on the prefix tree.
  - **What breaks it.** With per-sequence 1/|o_i| normalisation (GRPO, and our length-normalised GSPO ratio), the shared-prefix gradient becomes Σ_i A_i/|o_i| ≠ 0. Gradient then leaks into the dozens of tokens that every hypothesis got right, with a sign set by length differences: Dr. GRPO's length bias, here concentrated on deletions and insertions.
  - **MWER already has the cancellation.** Its weights post_i·(err_i − Ē) sum to zero. This may explain why classic MWER behaves well on N-best lists.
  - **Std normalisation is a separate problem.** It turns any reward gap, however small, into ±1 advantages. On the unlabelled verifier reward (55% pair accuracy on dev-spont, reward bench) that amplifies noise.
- *Experiment (exp20 arm 4).*
  - Add `--objective grpo_tok` to `train_rl.py`: token-sum log-prob loss, advantage r − mean(r) with no std, a constant normaliser (clips × group × a fixed max length), and the same CE anchor.
  - This needs `token_logprobs()` in `qxp_asr/nbest.py`, returning per-token values instead of the sum that `sequence_logprobs` returns.
  - Optional `--adv-norm {std,none}` for the existing GSPO arm, to isolate the two Dr. GRPO changes.
  - Cost: about 2 h of code and about 2 GPU-h per arm.
  - Expected (est.): single model, unseen-other-half WER 0 to −3% rel. versus the GSPO arm. It should match MWER if the cancellation argument is right.
  - Risk: low. The arm only differs in loss algebra. A null result is informative for the paper ("aggregation does not matter for short ASR outputs").

**T4. Fork-token restriction (Wang et al., "Beyond the 80/20 rule" [254]).**
- *What it is.* Restrict the policy gradient to the top 20% highest-entropy tokens ("forking tokens"). This matches full updates on Qwen3-8B and beats them on Qwen3-32B (**+11.04 AIME'25, +7.71 AIME'24**). Training on the bottom 80% only hurts.
- *ASR prior art.* **None in RL.**
  - The closest is token-level edit-distance rewards for RNN-T (EDRL [199], g.11.2).
  - Inference-time "Hybrid Search" [271] refines only the tokens with high semantic dependence in LoRA-adapted LLM-ASR, and reports large gains over global rescoring (numbers n/r).
- *For OUR system.*
  - In a beam group the forks are observable without any entropy threshold: they are the positions where the prefix trie branches. These are exactly our suffix-character decisions.
  - Experiment: `--token-mask fork:k`, keeping the gradient only on trie-branch tokens and the k = 2 tokens after each; it requires T3's `token_logprobs`. Run as exp20 arm 5.
  - Cost: about 2 GPU-h.
  - Expected (est.): 0 to −2% rel. beyond T3. The main value is less drift on tokens that were already correct; watch OOD.
  - Risk: with token-sum and mean-centred advantages, most non-fork tokens already cancel (T3), so the effect may be nil.

**T5. Importance-weight handling: CISPO, IcePop, DeepSeek-V3.2 off-policy masking, FP16 (MiniMax-M1 [249]; Ring-1T [252]; [245]; [279]).**
- *What each one is.*
  - **CISPO** clips the importance-sampling *weight* with a stop-gradient, sg(clip(r, ·, 1 + ε_high))·A·log π, instead of dropping tokens outside the trust region. Only ε_high is tuned. It matches DAPO in **50% of the steps** on Qwen2.5-32B.
  - **MiniMax** also found a training/inference probability mismatch and fixed it by computing the LM head in FP32: probability correlation **~0.9x → 0.99x**.
  - **IcePop** masks tokens whose *training-engine / inference-engine* probability ratio leaves **[0.5, 5]**, which is 1–2‰ of tokens. It gives AIME-25 **+14% over a 63% baseline** and is 6% rel. better than TIS on Ring-mini-2.0.
  - **V3.2** masks *negative-advantage* sequences that are too off-policy (KL above δ), because "highly off-policy negative samples can be detrimental". It also uses an unbiased KL estimator.
  - **FP16 [279]** argues that the mismatch comes from BF16 rounding and that FP16 removes it.
- *ASR prior art.* **None found** for any of the four.
- *For OUR system: mostly not applicable.*
  - Our "old" log-probs are recomputed with the same teacher-forced forward pass (`sequence_logprobs`), so there is no train/inference-engine mismatch to correct.
  - With `inner = 2`, ratios stay near 1.
  - The one transfer that applies is V3.2's: *very low-probability negative beams* are off-policy in the sampling sense, so pushing them further down wastes gradient.
  - A one-line option `--mask-neg-below <logp/token>` can ride on the T3 arm. Expected ≈ 0 (est.).
- *Not recommended as separate runs.*

**T6. Value-based PPO (VAPO [251]).**
- *What it is.* 60.4 AIME'24 on Qwen-32B, more than 10 points above DAPO under identical settings. It targets credit assignment over long CoT.
- *ASR prior art.* None found.
- *Verdict.* Not applicable to 5–40-token transcripts. A critic adds a model and instability for no credit-assignment gain that T3/T4 do not already give.

#### g.12.3 Data, curricula and exploration

**T7. Offline difficulty curriculum and prioritised sampling (GLM-4.5 [240]; GLM-4.1V RLCS [242]; Kimi k1.5 [247]; E2H [281]).**
- *What it is.*
  - **GLM-4.5** runs reasoning RL in two stages: moderate problems first, then "extremely difficult" ones with **pass@8 = 0 but pass@512 ≫ 0**. AIME'24 keeps improving where the baseline plateaus.
  - **Kimi k1.5** samples problems **∝ 1 − success rate** (prioritised sampling) on top of an easy-to-hard curriculum.
  - **E2H** finds that easy tasks matter early but must be *faded out* to avoid overfitting.
  - Already covered: DAPO dynamic sampling and online filtering (g.11.4).
- *ASR prior art.*
  - FunAudio's RL used "hard samples where systems disagree" [149].
  - Deng et al. [90] use a confidence-ordered curriculum, but for pseudo-labelling, not RL.
  - **No ASR RL paper filters by "greedy wrong, but solvable inside the N-best"** (the ASR analogue of pass@1 = 0, pass@k > 0).
- *For OUR system.*
  - The exp20 RL clips (training speakers reading the half-0 held-out sentences) mix three kinds of clip: ones the model already gets right (no signal), ones it can fix (the right answer is in the beam), and ones it cannot (no better beam).
  - Experiment: precompute greedy and beam-8 for every RL clip with LoRA v2. A new `scripts/rl_difficulty.py` (about 1 GPU-h) writes `artifacts/rl/difficulty_h0.parquet` with columns `greedy_err`, `oracle_err`, `solvable = oracle_err < greedy_err`.
  - Add `--clip-weights <parquet>` to `train_rl.py`: sample ∝ greedy_err for solvable clips, plus a 20% uniform floor (Kimi prioritised sampling).
  - Add `--stage2-after N`: after N steps, sample only solvable clips (GLM-4.5 stage 2).
  - Cost: 1 GPU-h + about 2 GPU-h per arm.
  - Expected (est.): `groups_kept` rises from its logged level towards 1, and convergence is faster. WER: 0 to −2% rel. versus the same arm without it.
  - Risk: the clips the model already gets right also carry the CE anchor's "keep doing this" signal. Keep the floor.

**T8. Entropy-collapse control (Clip-Cov / KL-Cov [253]; DAPO clip-higher [230]; GLM-TTS [243]; temperature schedules in GLM-4.5 [240], Kimi K2 [248], NIM4-ASR [280]).**
- *What each one is.*
  - **Entropy mechanism** [253]: performance trades against entropy as R = −a·e^H + b. Entropy falls because of the covariance between log-prob and advantage. Clip-Cov (r = 2×10⁻⁴) and KL-Cov (k = 2×10⁻³ for 7B, 2×10⁻⁴ for 32B) give **+2.0% (7B) and +6.4% (32B)** on average over GRPO, and **AIME'24 +15.0 / AIME'25 +14.6** at 32B.
  - **Clip-higher** uses ε_low 0.2, ε_high 0.28.
  - **GLM-4.5** raises the sampling temperature when reward plateaus.
  - **K2** and **NIM4-ASR** *decay* it instead: NIM4-ASR uses cosine 1.0 → 0.7.
- *ASR / speech prior art.*
  - **GLM-TTS [243]** (Zhipu, speech generation) uses clip-higher with ε_high 0.3 / ε_low 0.2, relaxed later in training, plus dynamic resampling up to 3× for homogeneous batches. CER 1.03 → **0.89** (test-zh), WER 2.23 → **1.91** (test-en).
  - **NIM4-ASR [280]**: GRPO with reward exp(−2·CER), a length-based hallucination term (0.5–2× the reference) and a context term. CER improves by −0.07 to −0.31 abs.
  - **No ASR paper measures or controls policy entropy.**
- *For OUR system.*
  - Beam groups cannot "collapse" in the sampling sense: the beams are always distinct.
  - The ASR-relevant form of collapse is **loss of N-best diversity**, which is what the hybrid lives on (32–46% oracle headroom, G0). See T9; no separate entropy run.
  - Clip-higher only matters for `--update clip`.

**T9. Diversity-preserving objectives for an ensemble member: pass@k / max@k / best-of-N-aware training / negative-sample reinforcement ([255]–[259]).**
- *What each one is.*
  - **Yue et al. [255]:** RLVR models beat their base at pass@1 but the *base* wins at large k. RL narrows the set of reachable answers; distillation can widen it.
  - **NSR [256]:** training only on *negative* samples improves pass@k across k ≤ 256 and often matches PPO/GRPO. Positive-only reinforcement improves pass@1 but hurts high k.
  - **Pass@k training [257]**, Enigmata, Qwen2.5-7B-Instruct, pass@1 / pass@k:
    - pass@1 training: 12.9 / 21.3;
    - pass@8 training: **17.9 / 29.8**;
    - pass@8 then pass@1: **30.8 / 40.6**.
  - **BoN-aware fine-tuning [258]:** Gemma-2B MATH Bo32 26.8 → **30.8**, pass@32 60.0 → **67.0**.
  - **max@k [259]:** the continuous-reward generalisation, with unbiased on-policy and off-policy gradients (numbers n/r).
- *ASR prior art.* **None found.**
  - ASR RL papers optimise 1-best WER.
  - O-1 [180] boosts the oracle *into* the 1-best rather than keeping the list diverse.
  - No paper trains an ASR model as a member of a downstream N-best selector.
- *Why it matters for OUR system.*
  - **Our deployed "policy" is the hybrid selector over the pooled N-best**, not any member's 1-best (g.11.7).
  - A GSPO/MWER member that sharpens its 1-best can *lower* the pooled oracle and the hybrid's WER even while its own greedy WER improves.
  - This risk is unmeasured for every RL arm so far.
- *Experiment.*
  1. **Mandatory metric for all exp20 arms (no training cost).** Run `llm_candidates.py --nbest rl` with the arm's adapter on `pool_dev_in`. Report the member's beam-8 oracle WER and the hybrid WER when the RL adapter *replaces* v2 (`joint_rescore2.py`, weights retuned as in exp15). Cost: about 3 GPU-h per arm evaluated.
  2. **New arms.** `--adv nsr` (W-REINFORCE: A_i = min(0, r_i − mean) + λ·max(0, r_i − mean), λ = 0.1) and `--adv maxk:2`, which gives the rank-based max@k advantage of [259] for k = 2 within G = 4. About 2 GPU-h each.
  - Expected (est.): member greedy WER is equal or slightly worse; pooled oracle 0 to −5% rel.; hybrid 0 to −2% rel.
  - Risk: a higher oracle does not guarantee a better hybrid. Selection accuracy is the bottleneck (reward bench: 70–73% pairwise accuracy even for the supervised hybrid).
- *Novelty.* Training a member *for the selector* is, as far as I found, new for ASR.

**T10. Mixed-policy RL: inject the reference into the group (LUFFY [260]; CHORD [261]) versus SFT-then-RL [262].**
- *What each one is.*
  - **LUFFY** replaces one rollout with the off-policy reference and shapes its importance weight: **+6.4** on average over six maths benchmarks, **+6.2** out of distribution. It trains weak models where on-policy RL fails.
  - **CHORD** schedules a global SFT/RL weight plus token-wise weights (numbers n/r).
  - **Counter-evidence [262]:** many mixed-policy papers compared against a *buggy* SFT baseline (a DeepSpeed CPU-offload optimiser that drops micro-batches, and an OpenRLHF loss-aggregation bug). Corrected SFT → RL beats every mixed-policy method tested, by **+3.8** (Qwen2.5-Math-7B) and **+22.2** (Llama-3.1-8B). Even 50 RL steps suffice.
- *Speech prior art.* Elmakies et al. [263] (Granite-Speech 2B):
  - mixed-policy GRPO **helped speech translation, hurt spoken QA**;
  - ASR gave "only minor improvements" in preliminary runs, which the authors attribute to ASR having a single correct output.
- *For OUR system.*
  - Our CE anchor on the gold reference is already the CHORD/PTX form (and Kimi K2's PTX loss [248]).
  - Unsolvable clips (no beam better than greedy, T7) get signal only through it.
  - **Do not add LUFFY-style injection.** The ASR evidence is minor, and [262] says to keep stages separate. Our exp20 design (SFT control arm on identical clips) already guards against a weak-SFT baseline.

#### g.12.4 Rewards, verifiers, self-play and test-time compute

**T11. Generative / process reward models and learned verifiers (GenRM [266]; DeepSeek-GRM [267]; Kimi K2 self-critique rubric reward [248]; process verifiers in Snell et al. [269]).**
- *What each one is.*
  - **GenRM** trains the verifier by next-token prediction ("is this correct? yes/no", optionally with CoT): best-of-N on GSM8K **73% → 93.4%**, algorithmic tasks **5% → 45.3%**.
  - **DeepSeek-GRM** scales a generative RM at inference with parallel sampling and a meta-RM.
  - **K2** keeps its self-critic honest by continually updating it on verifiable-reward rollouts.
- *ASR prior art.*
  - Discriminative confidence-estimation modules on seq2seq ASR [268] (overconfidence fixed; no RL).
  - Generative speech reward models exist only for naturalness (GSRM [283], used as an online-RLHF verifier for speech generation).
  - LLM judges for ASR evaluation exist, but need a capable LLM in the language (Kichwa: 17.31 → 17.00, g.11.6).
  - **No speech-conditioned generative verifier trained to accept or reject ASR hypotheses was found.**
- *For OUR system.*
  - The reward bench says the bottleneck is verifier accuracy: no reference-free reward exceeds about 70% pairwise accuracy on spontaneous/OOD speech.
  - The GenRM transfer is a **speech-conditioned yes/no verifier**:
    - the LoRA-7B decoder sees audio plus candidate, then predicts a "✓/✗" token;
    - training data: positives and negatives from augmented gold N-best lists of *held-out-sentence* clips (sentence cross-fit, as in exp20), so it cannot memorise the 2k prompts;
    - its log P(✓) enters `joint_rescore2.py` as a new feature and the reward bench as a new reward.
  - This is a variant of RL-3 (g.11.8) with a classification instead of an MWER head.
  - Script: `scripts/train_verifier.py` (new).
  - Cost: about 1 day of code and about 8–10 GPU-h.
  - Expected (est.): pairwise accuracy +3–6 points on dev-spont; hybrid 0 to −2% rel.
  - Risk: it learns the 2k-sentence prior; check on itest-unseen and OOD. It may be too late for the deadline.

**T12. Test-time scaling with verifiers and adaptive compute (Snell et al. [269]; ASR: LatentASR [270], LARM [272], Hybrid Search [271]).**
- *What each one is.*
  - **Snell et al.:** the best use of test-time compute depends on prompt difficulty. "Compute-optimal" allocation is **more than 4× more efficient than best-of-N**, and beats a 14× larger model where the base has non-trivial success.
  - **ASR (2026):**
    - **LatentASR** adds about 4M parameters to a frozen Qwen3-ASR-0.6B, trained on 500 utterances. Full FT, LoRA and prompt tuning all *regressed* in that regime. FLEURS −2.54% rel., ASCEND **−16.0% rel. CER**, and a value head skips about half of the utterances.
    - **LARM** loops a CTC encoder for more test-time depth.
    - **Hybrid Search** corrects targeted tokens with the base LLM.
- *ASR prior art for verifier-guided best-of-N.* Our joint/hybrid rescoring is already this. **No paper allocates extra N-best candidates adaptively by difficulty.**
- *For OUR system.*
  - Inference-only experiment: for clips whose hybrid margin (top-1 minus top-2 score) is below the 30th percentile, add candidates:
    - LoRA beam-16;
    - tempo 0.9/1.1 re-decodes, an elderly-like perturbation;
    - CTC N-best at two LM weights.
  - Then re-run the hybrid on the enlarged pool. Flags: `llm_candidates.py --only-clips <list> --tempo 0.9,1.1`, `joint_rescore2.py --extra-cands`.
  - Cost: about 4 GPU-h, plus CPU for tuning.
  - Expected (est.): oracle −5–10% rel. on those clips; hybrid 0 to −1.5% rel.
  - Risk: Snell and our reward bench agree that more candidates under a ~70%-accurate selector add distractors. Accept only through the paired bootstrap.

**T13. Self-play (SPIN [264]; speech chain [265]).**
- *What each one is.* **SPIN** iterates DPO between human data (chosen) and the model's own previous generations (rejected). It beats DPO with extra GPT-4 preferences on the Open LLM Leaderboard (numbers n/r).
- *ASR prior art.*
  - **None for SPIN.** The "Spin" cited in speech SSL papers is a speaker-invariant clustering method, unrelated.
  - The ASR form of self-play is the **ASR↔TTS speech chain** [265], which our exp18 (MMS-TTS back-translation) already instantiates.
- *Verdict: do not run SPIN.* It is exactly "ground truth as chosen, own output as rejected" on 1-character pairs. That is the configuration that degenerated in Koel-TTS [193] and that likelihood displacement predicts fails [195] (g.11.1).

**T14. Rejection-sampling fine-tuning on gold audio (RFT [273]; Retaining by Doing [228]).**
- *What it is.* **RFT** fine-tunes on the model's own *correct* samples, pooled across models: LLaMA-7B GSM8K **49.3 vs 35.9** SFT, with larger gains for weaker models.
- *ASR prior art.*
  - Pseudo-label N-best selection and O-1 [180] on unlabelled audio.
  - **On gold audio, "SFT on own correct beams"** reduces to standard CE when the 1-best is already right, and to oracle-boosting otherwise.
- *For OUR system.* T14 is subsumed by the exp20 SFT arm plus RL-1. **No new experiment.**

#### g.12.5 Forgetting, adapters and self-distillation

**T15. Small, separate RL adapters (LoRA Without Regret [278]; RL's Razor [227]; Guo et al. [189]).**
- *What it is.*
  - **Schulman / Thinking Machines [278]:** for RL, **rank-1 LoRA matches full fine-tuning** ("RL requires very low capacity", about 1 bit per episode). The optimal LoRA LR is about **10×** the full-FT LR. Attention-only LoRA underperforms MLP LoRA. LoRA tolerates large batches less well.
  - **RL's Razor:** forgetting tracks the forward KL from the base.
  - **Guo et al.:** a separate DPO LoRA merged at a scale **γ = 0.25** worked, and γ > 0.375 over-optimised.
- *ASR prior art.* Guo et al. [189] (separate scaled DPO adapter) is the closest. **No ASR paper uses a low-rank RL-only adapter with post-hoc scaling.**
- *For OUR system.*
  - Currently RL continues training the rank-16 SFT LoRA in place, so there is no way to dial the RL update back.
  - Experiment: `train_rl.py --merge-init --rl-rank 4`. Merge v2 into the weights, then add a fresh rank-4 adapter on the same modules (attention + FFN, as `DECODER_LORA` already does).
  - At evaluation, `--rl-scale γ ∈ {0.25, 0.5, 1}` picks the dose on dev-unseen-other-half and dev-spont. This is WiSE-FT for RL, consistent with our only OOD-positive trick so far (WiSE, §h).
  - Also log forward KL(π_v2 ‖ π_RL) on OOD clips as an early-stopping signal (RL's Razor).
  - Cost: about 1 h of code; the same GPU-h as the arm it replaces.
  - Expected (est.): the same peak gain as full-rank RL, with less OOD regression; makes any exp20 gain safer to deploy.
  - Risk: low.

**T16. On-policy self-distillation with privileged *acoustic* context (SDFT [274]; OPSD privileged-context design [275]; SDPG [276]; teacher–student ASR [277]).**
- *What it is.*
  - **SDFT** makes the *same model* its own teacher by giving the teacher privileged context (a demonstration). It trains with reverse KL on the student's own rollouts, using an EMA teacher.
    - Science QA: SFT 66.2 vs **SDFT 70.2** new-task accuracy, with prior capabilities 53.4 vs **64.5**. Tool use: 63.2 vs **70.6**.
    - It *underperforms* SFT at 3B and gains 4–7 points at 7–14B.
  - **OPSD** [275]: answer-only privileged context is within 0.2 points of full solutions. The right context is "the level of abstraction the student can still act on".
  - **SDPG** combines OPSD with group-relative verifier advantages.
- *ASR prior art.*
  - Offline frame-level teacher–student with *parallel* clean/target audio [277] gives **up to 44% WER reduction** (clean → noisy, adult → child) without transcripts.
  - X-OPD [186] (g.11.5) is on-policy, with a text teacher, but not for ASR.
  - **No on-policy, sequence-level self-distillation where the privileged context is the clean signal was found.**
- *For OUR system.*
  - The omniASR-LLM decoder in our code takes only audio and target (`nbest.make_batch`), so the privileged view must be **acoustic**.
  - The teacher is a frozen or EMA LoRA v2 on the clean clip. The student is the same model on an elderly-like perturbation of that clip: tempo 0.85–0.9, jitter/tremor, band-limiting via `qxp_asr/augment.py`.
  - Train with per-token reverse KL on the student's beam hypotheses. This uses gold spontaneous + held-out-sentence clips (no labels needed), plus the CE anchor.
  - This targets the elderly gap (no labelled elderly speech) without trusting any pseudo-label.
  - New `--objective sdft --perturb elderly` (about 1 day of code; reuses `cached_audio` twice). About 6–8 GPU-h.
  - Expected (est.): dev-spont/OOD 0 to −3% rel.; elderly effect unmeasurable without labels (RESULTS "Evaluation gap").
  - Risk: the perturbation may not resemble real 60+ speech. The reference model shares the student's errors.

#### g.12.6 Checked and not applicable

| Technique (source) | Why not for us |
|---|---|
| Overlong reward shaping / filtering (DAPO [230]); length penalty λ = 0.5 − (len − min)/(max − min) (Kimi k1.5 [247]); budget control (K2 [248]); repetition early-stop at 3,000 tokens with p > 0.99 (MiniMax [249]) | Transcripts are 5–40 tokens. Length is fixed by the audio; our length/repetition guard (g.10 R4) covers the failure mode |
| Partial rollouts (k1.5); keep-routing (V3.2); FP8 rollouts and asynchronous mode (slime [241]) | MoE, long-context or throughput infrastructure. Our 7B dense model with a cached encoder is not rollout-bound |
| Dynamic sampling temperature (GLM-4.5) | Beam groups have no temperature. If sampling is ever used, decay it (K2, NIM4-ASR 1.0 → 0.7) |
| Language-consistency reward (R1) | Wrong-script / translation errors are absent in our outputs (§h) |
| Long2short (k1.5: shortest-of-8 rejection sampling, merging; k1.5-short 60.8 AIME'24 at 3,272 tokens) | No reasoning trace to shorten |
| Clip-Cov / KL-Cov as separate runs | See T8: the ASR-relevant "collapse" is N-best diversity, handled by T9's metric and objectives |

#### g.12.7 Ranked top 5 for OUR system (estimates)

Rank = expected value × probability × fit to the remaining ~5 days. All runs follow the g.11.8 acceptance protocol: ≥ 3% rel. mean clean gain, paired bootstrap, and a shuffled-reward control for any RL gain.

| Rank | ID | Experiment | Where | GPU-h | Δ single (est.) | Δ hybrid (est.) | P(success) | Main risk |
|---|---|---|---|---|---|---|---|---|
| 1 | T3 + T4 | **Dr. GRPO token-sum, no-std objective (+ fork-token mask)** as exp20 arms 4–5. Prefix-sharing beams then give exact token-level credit at the suffix forks | `train_rl.py --objective grpo_tok [--token-mask fork:2]`, `nbest.token_logprobs` | 4 | 0 to −3% | 0 to −1% | 0.35 | Effect equals MWER (still a useful result) |
| 2 | T9 | **Member-for-selector evaluation and objectives**: beam-8 oracle + hybrid-with-RL-member metric for every exp20 arm; NSR and max@k arms | `llm_candidates.py`, `joint_rescore2.py`; `--adv nsr / maxk:2` | 3 per evaluated arm + 4 | ≈ 0 | 0 to −2% | 0.25 | Higher oracle is not converted by a ~70%-accurate selector |
| 3 | T7 | **GLM-4.5 / Kimi difficulty curriculum** on held-out-sentence clips (greedy wrong, solvable in beam-8; prioritised ∝ error, 20% floor, stage 2) | new `scripts/rl_difficulty.py`; `--clip-weights`, `--stage2-after` | 1 + 2 | 0 to −2% (vs the same arm) | ≈ 0 to −1% | 0.4 (for signal), 0.2 (for WER) | Fewer "keep correct" signals; keep the floor |
| 4 | T15 | **Fresh rank-4 RL adapter on the merged v2, scaled at inference (γ)** + forward-KL logging | `--merge-init --rl-rank 4 --rl-scale` | ≈ 0 extra | Same peak, less OOD loss | Safer deployment | 0.5 (as a safeguard) | None material |
| 5 | T1 | **R1-style RS-SFT round**: an exp20 winner joins the exp17 labeller, then a fresh LoRA. Only if exp20 passes the acceptance rule | `17_expert_iteration.sh` with `--adapter rl=` | 20 | −1 to −3% (dev-spont) | 0 to −2% beyond RL-1 | 0.2 | Conditional on exp20; narrows diversity (check with T9) |

**Next tier (do only if time remains):**
- T16, privileged-acoustic self-distillation for elderly robustness: novel, 1 day of code.
- T11, speech-conditioned generative verifier: attacks the 70% verifier ceiling, but 1 day of code plus about 10 GPU-h.
- T12, adaptive extra candidates for low-margin clips: inference-only, small expected gain.

**Not recommended:** SPIN (T13), LUFFY-style reference injection (T10), VAPO (T6), and separate CISPO / IcePop / FP16 runs (T5). Each either has no mechanism for our failure mode or targets infrastructure problems we do not have.

**Paper-ready summary.** "Most stabilisers in recent LLM RL reports target long, sampled, MoE rollouts and do not apply to 20-token ASR beams. Three ideas do transfer:
- *unbiased token-sum aggregation*, which on prefix-sharing N-best groups gives exact credit to the divergent suffix tokens;
- *difficulty curricula*, which keep only clips the model can fix within its N-best;
- *diversity-preserving objectives*, because the deployed policy is a selector over several models' N-best lists, not any member's 1-best.

None of these has been reported for ASR."

## (g″) Novelty check: exploration by input perturbation, generate-under-perturbation / verify-on-clean (2026-10-12, agent web search; arXiv/S2 APIs rate-limited, abstracts read)

**Verdicts on our findings:**

| Finding | Verdict | Prior art and gap |
|---|---|---|
| (1) Beam search yields few distinct hypotheses | partially covered | Subword beams fill with re-segmentations of the same words (Drexler & Glass, ICASSP 2019; Lakomkin et al., Interspeech 2020, arXiv:2008.04034, 97 % → 55 % unique). Sampling vs N-best for sequence training: Shannon 2017 (arXiv:1706.02776); Prabhavalkar et al. 2018 (arXiv:1712.01818; N-best MWER beat sampling). **Gap:** not measured for LLM-based ASR or linked to GRPO/GSPO rollouts. |
| (2) Perturbed-view decodes as exploration | partially covered | Vision RL: NoisyRollout (Liu et al., NeurIPS 2025, arXiv:2504.13055; rollouts on distorted images, update on clean); NoisyGRPO (arXiv:2510.21122). ASR pseudo-labelling: Lei et al. 2026 (arXiv:2602.13263; up to 28 hypotheses from pitch/stretch + weight noise, reference-free selection). MC-dropout hypotheses (Vyas et al., ICASSP 2019; DUST, arXiv:2011.13439). Audio QA: EchoDistill (arXiv:2605.23954). **Gap:** no pass@k / oracle comparison of beam vs sampling vs perturbed views for ASR. |
| (3) Generate under perturbation, verify on clean audio with a multi-model hybrid | **novel as a combination** | Nearest: ProGRes (arXiv:2409.00217; LLM-generated hypotheses added to the N-best and rescored); ASR-TRA (arXiv:2603.05231; CLAP verification of samples); multi-encoder ROVER (arXiv:2609.23525). No paper found that verifies perturbed decodes with clean-audio CTC + multi-adapter likelihoods. Our observation that view voting helps in-domain but hurts OOD while clean-audio verification helps both is also new. |
| (4) RL on labelled data ≤ SFT; MWER gradients vanish for perturbation-found candidates | partially covered and contested | NoisyRollout anneals the noise because the mismatch hurts (54.9 → 43.9 without annealing); our vanishing gradient is the ASR counterpart. GRPO-for-ASR reports gains (Shivakumar et al., ASRU 2025, arXiv:2509.01939, up to 18.4 % rel.; Gao et al., arXiv:2509.18569). These must be reconciled; likely factors are hallucination-heavy baselines and data scale vs our strong, low-resource SFT base. |

**Questions:**
- (a) TTA for ASR is thin: GTTA (Jelea et al., arXiv:2507.01347; Whisper mel perturbation + vote, −1.7 % rel., distilled back); MBR over Whisper samples (Jinnai, arXiv:2510.19471). Perturbed decodes have not been used as candidates for an external clean-audio verifier.
- (c) Test-time compute scaling for ASR with best-of-N + verifier (WER vs N curves): **not found**. Our view-scaling curve is a plausible contribution.
- (d) Search-then-distill with perturbations: partially (GTTA distillation; MBR/QE fine-tuning for MT, arXiv:2309.10966; Lei et al.). It differs from noisy student, where the teacher sees clean input and the student sees noisy input; here the search runs on perturbed input and the verification on clean input.
- (e) Likelihood averaged over perturbed views as a verifier: **not found**. The nearest work uses agreement-under-perturbation confidence (Vyas; DUST; Lei et al.).

## (h) Applied in this work — technique → implementation → measured effect

Living index for the system-description paper. Every row cites its inspiration (full entries in the
bibliography or the section given) and points to the code and the measured result
(`reports/RESULTS.md`). Dev = speaker-disjoint; "unseen" = held-out sentences (the hidden test reads
new sentences, `scripts/pool_overlap.py`); OOD = organizer radio-drama clips (never tuned on).

| Technique | Inspiration | Where | Measured effect (WER, greedy unless stated) |
|---|---|---|---|
| omniASR-CTC-1B-v2 fine-tuning, head pruned to 30-symbol Puno alphabet | Omnilingual ASR (arXiv:2511.09690) §5.7.5 | `scripts/train_ctc.py`, `qxp_asr/omni.py` | unseen 30.4→16.2, spont 35.4→20.7 |
| Unseen-sentence + speaker-disjoint dev protocol | test-pool analysis (ours); CV split practice | `qxp_asr/splits.py`, `scripts/pool_overlap.py` | test median dist. to pool 0.495 = unseen (0.489) vs seen (0.028) |
| Silence-trim / speed / noise augmentation | Ko et al. 2015; EDA (test clips trimmed) | `qxp_asr/augment.py` | always on (no ablation yet) |
| Weighted multi-source sampling (gold spont ×3) | data-mixing practice (fairseq/omniASR mixtures) | `qxp_asr/data.py` | always on |
| Model-internal SpecAugment-style masking | Park et al. 2019; wav2vec 2.0 FT | omniASR train mode | always on |
| WiSE-FT weight interpolation with zero-shot | Wortsman et al. 2022 (arXiv:2109.01903) | `scripts/merge_weights.py` | a=0.7: OOD 20.6→19.8, spont 20.7→20.3; a=0.85: unseen 16.2→16.0 |
| Word 4-gram KenLM beam search | Heafield 2011; pyctcdecode | `scripts/decode.py` | unseen 16.2→15.5, spont 20.7→19.8; OOD slightly worse |
| Char 6-gram lexicon-free decoding | Likhomanenko et al. 2019; flashlight | `qxp_asr/decode.py` | worse than greedy (order capped at 6) — negative result |
| Char 10-gram N-best rescoring | §g.2 (char/subword LMs for agglutinative languages) | `scripts/rescore.py` | itest-unseen 22.8→20.6, spont 20.7→19.2 |
| External Quechua text respelled to Puno orthography | QuBERT corpus, Wikipedia, FineWeb-2, FLORES (§d) | `scripts/build_text_corpus.py`, `qxp_asr/respell.py` | spont-dev OOV 24.4 %→15.7 % |
| LoRA on omniASR-LLM-7B decoder (frozen encoder) | Hu et al. 2021; §g.6 | `scripts/train_llm.py`, `qxp_asr/lora.py` | unseen 21.3→12.6, OOD **23.3→14.9**, itest-spont 7.1→6.6 (≤39 s); apparent dev-spont regression was a chunking artefact |
| Pause-based chunking of long inputs | ours (fixed chunks → 4× insertions on > 39 s clips) | `qxp_asr/segment.py`, `qxp_asr/omni.py` | removes tail hallucination |
| Word-level ROVER of LLM-7B-LoRA + CTC+LM + zero-shot 7B | Fiscus 1997 | `scripts/combine_systems.py`, `qxp_asr/combine.py` | dev-spont 14.7→14.5, OOD 14.9→14.4, itest-unseen 20.2→20.0 |
| Orthographic-error decomposition | ours | `scripts/error_analysis.py` | zero-shot 7B: 6.8 of 24 scripted WER points are spelling-convention substitutions; ~1 point on spont |
| Pause-based segmentation of long untranscribed audio | slimIPL data prep (Likhomanenko et al. 2021) | `scripts/segment_long.py` | 404 clips → 1,518 segments (6.75 h) |
| Joint CTC + LLM-ASR N-best rescoring (exact CTC likelihood + char-10g LM) | Watanabe et al. 2017 (hybrid CTC/attention, IEEE JSTSP); §g.2 | `scripts/joint_rescore.py` | itest-unseen 20.0→**16.5** (beats organizer 17.8), dev-unseen 12.3→10.5, itest-spont 6.9→6.7 |
| Hybrid CTC + multi-LLM likelihood rescoring (CTC N-best + beam-8 N-best of zero-shot / LoRA v1 / LoRA v2, every candidate scored by every LLM) | Watanabe et al. 2017; §g.10 recommendation | `scripts/ctc_nbest.py`, `scripts/llm_candidates.py`, `scripts/joint_rescore2.py`, `qxp_asr/nbest.py` | dev-spont 14.20→13.47, OOD 15.11→13.82, dev-unseen 9.91→9.11 (dev-mode); final: dev-spont 13.95→13.38 |
| Final-mode retraining with automatic dev-based recipe selection | standard practice | `scripts/runs/10_final.sh`, `scripts/select_best.py` | clean sets: dev-spont 14.20→13.95, OOD 15.11→14.58 |
| Decoder text-CPT (text-only LM loss through the decoder LoRA, interleaved with ASR) | internal-LM / text-only adaptation of end-to-end ASR; continued pretraining (§c C1) | `qxp_asr/textlm.py`, `scripts/train_llm.py --text-corpus` | OOD 15.11→**14.27** WER, 2.53→2.22 CER; in-domain neutral |
| Encoder LoRA-CPT (wav2vec 2.0 SSL objective on in-language audio, adapters merged) | Hsu et al. 2021; DeHaven & Billa 2022; Nowakowski et al. 2023; Hu et al. 2021 | `scripts/cpt_w2v.py`, `scripts/train_ctc.py --init-encoder` | exp13b: not completed; superseded by supervised encoder LoRA (paper Sect. 5) |
| Sequence-level RL: GSPO / MWER on beam-search N-best groups, encoder cached per clip | Zheng et al. 2025 (GSPO, arXiv:2507.18071); Shao et al. 2024 (GRPO, arXiv:2402.03300); Prabhavalkar et al. 2018 (MWER) | `scripts/train_rl.py` | done (exp11, exp20, exp46): no gain over SFT on held-out sentences (paper Sect. 6) |
| IcePop-style double-sided trust-region masking (instead of PPO clipping), dropping noisy / signal-free groups, hybrid reward (rule-based WER/CER on gold + independent-verifier CTC likelihood on unlabelled audio) | GLM-5 Team 2026, "GLM-5: from Vibe Coding to Agentic Engineering", arXiv:2602.15763 (GLM-5.3 shares this post-training stack) | `scripts/train_rl.py` (`--update mask`, `--max-best-err`, `--unlabeled`) | done (exp11, exp20, exp46): no gain over SFT on held-out sentences (paper Sect. 6) |
| Speed: TF32, cuDNN autotune, fused AdamW, activation checkpointing only where memory requires | PyTorch performance practice | `qxp_asr/speed.py` | lossless; frozen-encoder LoRA ~25-30 % faster |
| Unsloth / vLLM feasibility (transformers PR #43265, vLLM PR #36864) | — | `reports/unsloth_vllm_spike.md`, `scripts/spike_hf_omniasr.py` | HF port parity verified (7B+LoRA); no-go for RL speed-up |
| Agreement-filtered pseudo-labels (ROVER consensus) + 50+ up-weighting | Park et al. 2020 (NST); Xu et al. 2020 (IPL); Fiscus 1997; §g.1 | `scripts/pseudo_label.py`, `qxp_asr/combine.py` | 27.5 h kept of 29.9 h; LoRA v2 spont dev 16.2→14.9, itest-spont 7.1→6.6 (scripted −0.5) |
| Multi-candidate joint rescoring (CTC N-best + LoRA v1 + v2 + zero-shot) | Watanabe et al. 2017 | `scripts/joint_rescore.py` | joint_v3: dev-unseen 9.9, itest-unseen **16.0**, itest-spont 6.4, OOD 15.1/2.46 |
| Perturbed-view candidates (decode perturbed copies, verify on the clean audio) | NoisyRollout (Liu et al. 2025); GTTA (Jelea et al. 2025); Lei et al. 2026 | `scripts/llm_candidates.py --views`, `scripts/joint_rescore2.py` | better candidate among 8 for 16.7% of clips vs 6.0% for beam search; WER gain n.s. (paper Sects. 4, 6) |
| Encoder LoRA, enc+dec LoRA (where to adapt) | EAVA (Shi et al. 2026); Hu et al. 2022 | `scripts/train_llm.py --encoder lora` | 7B itest unseen 20.2 to 16.4 / 16.0; no gain for the 300M and 1B encoders (paper Sect. 5) |
| Transfer to another model family | Whisper (Radford et al. 2023) | `scripts/train_whisper.py` | encoder LoRA beats decoder LoRA (-6.4, -4.6 over two seeds) |
| Linear probes with shuffled-label controls; linear CKA | Hewitt & Liang 2019; Kornblith et al. 2019; Pasad et al. 2021 | `scripts/encoder_probe.py` | stop contrasts at 94-97% in all pretrained encoders; adaptation moves only the 7B early |
| Causal block ablation, spectral truncation, exact seed averaging of LoRA updates | Aghajanyan et al. 2021; Hu et al. 2022 | `scripts/infer_llm.py --keep-encoder-layers`, `scripts/lora_surgery.py` | bottom quarter needed, top quarter not; 4 of 16 directions keep the gain; averaging no better |
| Pigeonhole (speaker x sentence) bootstrap, Holm, exact sign tests | Bisani & Ney 2004; Owen 2007; Holm 1979 | `src/qxp_asr/stats.py`, `scripts/paper_stats.py`, `scripts/compare_runs.py` | every interval in the paper |
| TTS likelihood as a verifier | OmniVoice (k2-fsa) | `scripts/tts_score.py` | 55% pairwise accuracy out of domain: negative |
| Synthetic verifier curves (blend to oracle, noisy oracle) | ours | `scripts/verifier_curve.py` | at our accuracy the noisy oracle matches our WER on 3 of 4 sets |

## Bibliography (consolidated)

Each entry gives authors, year, title, venue and a link.

1. Omnilingual ASR team (Keren, G., et al.) 2025. *Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages.* arXiv:2511.09690. https://arxiv.org/abs/2511.09690
2. Huaman, E., Gamarra Lafuente, A., Cordova, J., Korhonen, A. 2026. *Building Community-Centred NLP Resources for Puno Quechua.* AmericasNLP 2026. https://aclanthology.org/2026.americasnlp-6.13/ (arXiv:2605.28253)
3. Huaman, E., Huaman, W., Huaman, J. L., Quispe, N. 2025. *Quechua Speech Datasets in Common Voice: The Case of Puno Quechua.* SIMBig 2025, Springer CCIS. https://link.springer.com/chapter/10.1007/978-3-032-20322-9_13 (arXiv:2510.13871)
4. Pratap, V., et al. 2023. *Scaling Speech Technology to 1,000+ Languages.* JMLR 2024. arXiv:2305.13516
5. Babu, A., et al. 2021. *XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale.* Interspeech 2022. arXiv:2111.09296
6. Baevski, A., et al. 2020. *wav2vec 2.0.* NeurIPS. arXiv:2006.11477
7. Chung, Y.-A., et al. 2021. *W2v-BERT.* ASRU. arXiv:2108.06209
8. Seamless Communication. 2023. *SeamlessM4T.* arXiv:2308.11596; *Seamless.* arXiv:2312.05187
9. Chen, W., et al. 2024. *Towards Robust Speech Representation Learning for Thousands of Languages (XEUS).* EMNLP. arXiv:2407.00837
10. Zanon Boito, M., et al. 2024. *mHuBERT-147.* Interspeech. arXiv:2406.06371
11. Radford, A., et al. 2022. *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper).* ICML 2023. arXiv:2212.04356
12. Peng, Y., et al. 2025. *OWSM v4.* Interspeech. arXiv:2506.00338
13. Sekoyan, M., et al. 2025. *Canary-1B-v2 & Parakeet-TDT-0.6B-v3.* arXiv:2509.14128; Koluguri, N. R., et al. 2025. *Granary.* arXiv:2505.13404
14. Shi, X., et al. 2026. *Qwen3-ASR Technical Report.* arXiv:2601.21337
15. Liu, A. H., et al. 2025. *Voxtral.* arXiv:2507.13264
16. Saon, G., et al. 2025. *Granite-speech.* arXiv:2505.08699
17. Hsu, W.-N., et al. 2021. *Robust wav2vec 2.0.* Interspeech. arXiv:2104.01027
18. DeHaven, M., Billa, J. 2022. *Improving Low-Resource Speech Recognition with Pretrained Speech Models: Continued Pretraining vs. Semi-Supervised Training.* arXiv:2207.00659
19. Nowakowski, K., et al. 2023. *Adapting Multilingual Speech Representation Model for a New, Underresourced Language through Multilingual Fine-tuning and Continued Pretraining.* IP&M. arXiv:2301.07295
20. Liang, S., Levow, G.-A. 2025. *Breaking the Transcription Bottleneck: Fine-tuning ASR Models for Extremely Low-Resource Fieldwork Languages.* FieldMatters. arXiv:2506.17459
21. Nahabwe, A., et al. 2025. *Benchmarking Automatic Speech Recognition Models for African Languages.* arXiv:2512.10968
22. Heafield, K. 2011. *KenLM: Faster and Smaller Language Model Queries.* WMT. https://aclanthology.org/W11-2123/
23. Kensho Technologies. *pyctcdecode.* https://github.com/kensho-technologies/pyctcdecode ; von Platen, P. 2022. *Boosting Wav2Vec2 with n-grams in Transformers.* https://huggingface.co/blog/wav2vec2-with-ngram
24. Kahn, J., et al. 2022. *Flashlight.* ICML. arXiv:2201.12465; Yang, Y.-Y., et al. 2022. *TorchAudio.* ICASSP. arXiv:2110.15018
25. Likhomanenko, T., et al. 2019. *Who Needs Words? Lexicon-Free Speech Recognition.* Interspeech. arXiv:1904.04479
26. Hirsimäki, T., et al. 2006. *Unlimited vocabulary speech recognition with morph language models applied to Finnish.* Computer Speech & Language 20(4)
27. Smit, P., Virpioja, S., Kurimo, M. 2021. *Advances in subword-based HMM-DNN speech recognition across languages.* Computer Speech & Language 66:101158
28. Smit, P., Virpioja, S., Grönroos, S.-A., Kurimo, M. 2014. *Morfessor 2.0: Toolkit for statistical morphological segmentation.* EACL demos. https://aclanthology.org/E14-2006/
29. de Zuazo, X., et al. 2025. *Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages.* arXiv:2503.23542
30. Park, D. S., et al. 2020. *Improved Noisy Student Training for ASR.* Interspeech. arXiv:2005.09629
31. Xu, Q., et al. 2020. *Iterative Pseudo-Labeling for Speech Recognition.* Interspeech. arXiv:2005.09267
32. Likhomanenko, T., et al. 2021. *slimIPL.* Interspeech. arXiv:2010.11524
33. Higuchi, Y., et al. 2021. *Momentum Pseudo-Labeling.* Interspeech. arXiv:2106.08922
34. Kahn, J., Lee, A., Hannun, A. 2020. *Self-Training for End-to-End Speech Recognition.* ICASSP. arXiv:1909.09116
35. Xu, Q., et al. 2021. *Self-training and Pre-training are Complementary for Speech Recognition.* ICASSP. arXiv:2010.11430
36. Lugosch, L., et al. 2022. *Pseudo-Labeling for Massively Multilingual Speech Recognition.* ICASSP. arXiv:2111.00161
37. *Cross-lingual Knowledge Transfer and Iterative Pseudo-labeling for Low-Resource Speech Recognition with Transducers.* 2023. arXiv:2305.13652
38. *A Novel Self-training Approach for Low-resource Speech Recognition.* 2023. arXiv:2308.05269
39. Park, D. S., et al. 2019. *SpecAugment.* Interspeech. arXiv:1904.08779
40. Ko, T., Peddinti, V., Povey, D., Khudanpur, S. 2015. *Audio augmentation for speech recognition.* Interspeech
41. Snyder, D., Chen, G., Povey, D. 2015. *MUSAN.* arXiv:1510.08484
42. Rossenbach, N., et al. 2020. *Generating Synthetic Audio Data for Attention-Based Speech Recognition Systems.* ICASSP. arXiv:1912.09257
43. Huang, Z., et al. 2023. *Text Generation with Speech Synthesis for ASR Data Augmentation.* arXiv:2305.16333
44. Ibaraki, K., et al. 2025. *Frustratingly Easy Data Augmentation for Low-Resource ASR.* arXiv:2509.15373
45. Fiscus, J. 1997. *A post-processing system to yield reduced word error rates: ROVER.* IEEE ASRU
46. Goel, V., Byrne, W. 2000. *Minimum Bayes-risk automatic speech recognition.* Computer Speech & Language 14(2)
47. Wortsman, M., et al. 2022. *Model soups.* ICML. arXiv:2203.05482; Izmailov, P., et al. 2018. *SWA.* UAI. arXiv:1803.05407
48. Chen, C., et al. 2023. *HyPoradise.* NeurIPS. arXiv:2309.15701
49. Radhakrishnan, S., et al. 2023. *Whispering LLaMA.* EMNLP. arXiv:2310.06434
50. Yang, C.-H. H., et al. 2023. *Generative Speech Recognition Error Correction with LLMs and Task-Activating Prompting.* ASRU. arXiv:2309.15649
51. Gu, Z., et al. 2024. *Denoising LM.* arXiv:2405.15216
52. Yun, H., Amooie, R., de Vries, W., van Noord, R., Wieling, M. 2026. *Can Large Language Models Reliably Correct Errors in Low-Resource ASR? A Contamination-Aware Case Study on West Frisian.* arXiv:2605.19711
53. Lin, G.-T., Li, S.-W., Lee, H.-y. 2022. *Listen, Adapt, Better WER (SUTA).* Interspeech. arXiv:2203.14222
54. Hu, E. J., et al. 2021. *LoRA.* ICLR 2022. arXiv:2106.09685
55. Rafkin, E., DeGenaro, D., Yang, X. 2026. *Task Arithmetic with Support Languages for Low-Resource ASR.* arXiv:2601.07038
56. Shi, J., et al. 2023. *Findings of the 2023 ML-SUPERB Challenge.* ASRU. arXiv:2310.05513
57. Chen, W., et al. 2025. *The ML-SUPERB 2.0 Challenge.* arXiv:2509.07139; Alumäe, T., et al. 2025. *TalTech Systems for the Interspeech 2025 ML-SUPERB 2.0 Challenge.* arXiv:2506.01458; Wang et al. 2025. arXiv:2505.24200
58. Ardila, R., et al. 2020. *Common Voice: A Massively-Multilingual Speech Corpus.* LREC. arXiv:1912.06670
59. Cardenas, R., Zevallos, R., Baquerizo, R., Camacho, L. 2018. *Siminchik: A Speech Corpus for Preservation of Southern Quechua.* LREC CCURL workshop
60. Zevallos, R., Bel, N., Cámbara, G., Farrús, M., Luque, J. 2022. *Data Augmentation for Low-Resource Quechua ASR Improvement.* Interspeech. DOI 10.21437/Interspeech.2022-770
61. Zevallos, R., Camacho, L., Melgarejo, N. 2022. *Huqariq.* LREC. arXiv:2207.05498
62. Zevallos, R., et al. 2022. *Introducing QuBERT.* DeepLo. https://aclanthology.org/2022.deeplo-1.1/
63. Ebrahimi, A., et al. 2023. *Findings of the Second AmericasNLP Competition on Speech-to-Text Translation.* PMLR 220. https://proceedings.mlr.press/v220/ebrahimi23a.html
64. Romero, M., Gómez, S., Torre, I. G. 2024. *ASR Advancements for Indigenous Languages of the Americas.* Applied Sciences 14(15):6497. arXiv:2404.08368
65. Ortega, J. E., Zevallos, R., et al. 2026. *Team QUESPA System Submission for the IWSLT 2026 Dialectal and Low-resource ST Task.* IWSLT 2026. https://aclanthology.org/2026.iwslt-1.5.pdf
66. Paccotacya-Yanque, R., et al. 2022. *A speech corpus of Quechua Collao for automatic dimensional emotion recognition.* Scientific Data 9:778
67. Mager, M., et al. 2021. *Findings of the AmericasNLP 2021 Shared Task on Open MT for Indigenous Languages of the Americas.* AmericasNLP. https://aclanthology.org/2021.americasnlp-1.23/
68. Agić, Ž., Vulić, I. 2019. *JW300.* ACL. https://aclanthology.org/P19-1310/
69. Penedo, G., et al. 2025. *FineWeb2.* arXiv:2506.20920; Kargaran, A. H., et al. 2024. *GlotCC.* NeurIPS D&B. arXiv:2410.23825; Kudugunta, S., et al. 2023. *MADLAD-400.* arXiv:2309.04662; Akerman, V., et al. 2023. *eBible Corpus.* arXiv:2304.09919
70. Rios, A. 2015. *A Basic Language Technology Toolkit for Quechua.* PhD thesis, Univ. Zurich. https://github.com/a-rios/squoia
71. Wortsman, M., Ilharco, G., Kim, J. W., Li, M., Kornblith, S., Roelofs, R., Gontijo-Lopes, R., Hajishirzi, H., Farhadi, A., Namkoong, H., Schmidt, L. 2022. *Robust fine-tuning of zero-shot models (WiSE-FT).* CVPR. https://arxiv.org/abs/2109.01903
72. Carvalho, C., Teixeira, F., Rolland, T., Abad, A. 2026. *Exploring the potential and limitations of Model Merging for Multi-Domain Adaptation in ASR.* arXiv:2603.05354. https://arxiv.org/abs/2603.05354
73. Vander Eeckt, S., Van hamme, H. 2023. *Weight Averaging: A Simple Yet Effective Method to Overcome Catastrophic Forgetting in Automatic Speech Recognition.* ICASSP. https://arxiv.org/abs/2210.15282
74. Lin, T.-Q., Huang, W.-P., Tang, H., Lee, H.-y. 2026. *Speech-FT: Merging Pre-trained And Fine-Tuned Speech Representation Models For Cross-Task Generalization.* IEEE/ACM TASLP 34. https://arxiv.org/abs/2502.12672
75. Ducorroy, A., Riad, R. 2025. *Robust fine-tuning of speech recognition models via model merging: application to disordered speech.* Interspeech. https://arxiv.org/abs/2505.20477
76. Balaji Shankar, N., Wang, Z., Eren, E., Alwan, A. 2025. *Selective Attention Merging for low resource tasks: A case study of child ASR.* arXiv:2501.08468. https://arxiv.org/abs/2501.08468
77. Ajroldi, N., Orvieto, A., Geiping, J. 2025. *When, Where and Why to Average Weights?* arXiv:2502.06761. https://arxiv.org/abs/2502.06761
78. Gitman, I., Lavrukhin, V., Laptev, A., Ginsburg, B. 2023. *Confidence-based Ensembles of End-to-End Speech Recognition Models.* Interspeech. https://arxiv.org/abs/2306.15824
79. Parikh, A., ten Bosch, L., van den Heuvel, H. 2024. *Ensembles of Hybrid and End-to-End Speech Recognition.* LREC-COLING. https://aclanthology.org/2024.lrec-main.547/
80. Kamo, N., Ochiai, T., Delcroix, M., Nakatani, T. 2025. *MOVER: Combining Multiple Meeting Recognition Systems.* arXiv:2508.05055. https://arxiv.org/abs/2508.05055
81. Saon, G., Thomas, S., Fukuda, T., Nagano, T., et al. 2026. *Self-Speculative Decoding for LLM-based ASR with CTC Encoder Drafts.* arXiv:2603.11243. https://arxiv.org/abs/2603.11243
82. Hu, S., Xie, X., Jin, Z., Geng, M., Wang, Y., Cui, M., Deng, J., Liu, X., Meng, H. 2023. *Exploring Self-supervised Pre-trained ASR Models for Dysarthric and Elderly Speech Recognition.* ICASSP. https://arxiv.org/abs/2302.14564
83. Hu, S., Xie, X., Geng, M., Jin, Z., Deng, J., Li, G., Wang, Y., Cui, M., Wang, T., Meng, H., Liu, X. 2024. *Self-supervised ASR Models and Features for Dysarthric and Elderly Speech Recognition.* IEEE/ACM TASLP. https://arxiv.org/abs/2407.13782
84. Wang, T., Deng, J., Geng, M., Ye, Z., Hu, S., Wang, Y., Cui, M., Jin, Z., Liu, X., Meng, H. 2022. *Conformer Based Elderly Speech Recognition System for Alzheimer's Disease Detection.* Interspeech. https://arxiv.org/abs/2206.13232
85. Geng, M., Xie, X., Liu, S., Yu, J., Hu, S., Liu, X., Meng, H. 2020. *Investigation of Data Augmentation Techniques for Disordered Speech Recognition.* Interspeech. https://www.isca-archive.org/interspeech_2020/geng20_interspeech.html
86. Liu, S., Geng, M., Hu, S., Xie, X., et al. 2022. *Recent Progress in the CUHK Dysarthric Speech Recognition System.* arXiv:2201.05845. https://arxiv.org/abs/2201.05845
87. Geng, M., Xie, X., Deng, J., Jin, Z., Li, G., Wang, T., Hu, S., Li, Z., Meng, H., Liu, X. 2024. *Homogeneous Speaker Features for On-the-Fly Dysarthric and Elderly Speaker Adaptation.* arXiv:2407.06310. https://arxiv.org/abs/2407.06310
88. Hu, S., Xie, X., Geng, M., et al. 2024. *Structured Speaker-Deficiency Adaptation of Foundation Models for Dysarthric and Elderly Speech Recognition.* arXiv:2412.18832. https://arxiv.org/abs/2412.18832
89. Deng, C., Xie, X., Hu, S., Geng, M., et al. 2025. *MOPSA: Mixture of Prompt-Experts Based Speaker Adaptation for Elderly Speech Recognition.* Interspeech. https://arxiv.org/abs/2505.24224
90. Deng, C., Xie, X., Hu, S., Deng, J., et al. 2026. *Confidence Score Guided Incremental and Speaker Adaptive Pseudo-Labeling for Semi-Supervised Elderly Speech Recognition.* arXiv:2606.16546. https://arxiv.org/abs/2606.16546
91. Chen, Y., Wang, H., Wang, S., et al. 2025. *SeniorTalk: A Chinese Conversation Dataset with Rich Annotations for Super-Aged Seniors.* arXiv:2503.16578. https://arxiv.org/abs/2503.16578
92. Shekoufandeh, G., Boersma, P., van den Bosch, A. 2025. *Improving the Inclusivity of Dutch Speech Recognition by Fine-tuning Whisper on the JASMIN-CGN Corpus.* arXiv:2502.17284. https://arxiv.org/abs/2502.17284
93. Lee, M., Hong, S., Lee, C., Choi, S., et al. 2026. *Elderly-Contextual Data Augmentation via Speech Synthesis for Elderly ASR.* arXiv:2604.24770. https://arxiv.org/abs/2604.24770
94. Vu, T.-L., Zeng, Z., Xu, H., Chng, E.-S. 2019. *Audio Codec Simulation based Data Augmentation for Telephony Speech Recognition.* APSIPA ASC. http://www.apsipa.org/proceedings/2019/pdfs/216.pdf
95. Moisio, A., Porjazovski, D., Rouhe, A., Getman, Y., Virkkunen, A., Grósz, T., Lindén, K., Kurimo, M. 2022. *Lahjoita puhetta: a large-scale corpus of spoken Finnish with some benchmarks.* Language Resources and Evaluation. https://arxiv.org/abs/2203.12906
96. Rouhe, A., Virkkunen, A., Leinonen, J., Kurimo, M. 2022. *Low Resource Comparison of Attention-based and Hybrid ASR Exploiting wav2vec 2.0.* Interspeech. https://www.isca-archive.org/interspeech_2022/rouhe22_interspeech.pdf
97. Olev, A., Alumäe, T. 2022. *Estonian Speech Recognition and Transcription Editing Service.* Baltic J. Modern Computing 10(3). https://doi.org/10.22364/bjmc.2022.10.3.14
98. Jain, A., Rouhe, A., Grönroos, S.-A., Kurimo, M. 2020. *Finnish ASR with Deep Transformer Models.* Interspeech. https://www.isca-archive.org/interspeech_2020/jain20b_interspeech.pdf
99. Gupta, V., Boulianne, G. 2020. *Automatic Transcription Challenges for Inuktitut, a Low-Resource Polysynthetic Language.* LREC. https://aclanthology.org/2020.lrec-1.307.pdf
100. Manohar, K., Jayan, A. R., Rajan, R. 2023. *Syllable Subword Tokens for Open Vocabulary Speech Recognition in Malayalam.* arXiv:2301.06736. https://arxiv.org/abs/2301.06736
101. Javed, T., Doddapaneni, S., Raman, A., Bhogale, K. S., Ramesh, G., Kunchukuttan, A., Kumar, P., Khapra, M. M. 2022. *Towards Building ASR Systems for the Next Billion Users (IndicWav2Vec).* AAAI. https://arxiv.org/abs/2111.03945
102. Liu, Z., Venkateswaran, N., Le Ferrand, É., Prud'hommeaux, E. 2024. *How Important is a Language Model for Low-resource ASR?* Findings of ACL. https://aclanthology.org/2024.findings-acl.13.pdf
103. Novosad, I. 2026. *The Anatomy of the CTC Oracle Gap: Acoustic Exhaustion and Linguistic Recovery.* arXiv:2606.23306. https://arxiv.org/abs/2606.23306
104. Pasad, A., Chou, J.-C., Livescu, K. 2021. *Layer-wise analysis of a self-supervised speech representation model.* ASRU. https://arxiv.org/abs/2107.04734
105. Zaiem, S., Parcollet, T., Essid, S. 2024. *Less Forgetting for Better Generalization: Exploring Continual-learning Fine-tuning Methods for Speech Self-supervised Representations.* arXiv:2407.00756. https://arxiv.org/abs/2407.00756
106. Lee, J., Watanabe, S. 2021. *Intermediate Loss Regularization for CTC-based Speech Recognition.* ICASSP. https://arxiv.org/abs/2102.03216
107. Nozaki, J., Komatsu, T. 2021. *Relaxing the Conditional Independence Assumption of CTC-based ASR by Conditioning on Intermediate Predictions (Self-conditioned CTC).* Interspeech. https://arxiv.org/abs/2104.02724
108. Higuchi, Y., Karube, K., Ogawa, T., Kobayashi, T. 2022. *Hierarchical Conditional End-to-End ASR with CTC and Multi-Granular Subword Units.* ICASSP. https://arxiv.org/abs/2110.04109
109. Chen, W., Yan, B., Shi, J., et al. 2023. *Improving Massively Multilingual ASR With Auxiliary CTC Objectives.* ICASSP. https://arxiv.org/abs/2302.12829
110. Wright, G. A., Cappellazzo, U., Zaiem, S., Raj, D., et al. 2023. *Training dynamic models using early exits for automatic speech recognition on resource-constrained devices.* arXiv:2309.09546. https://arxiv.org/abs/2309.09546
111. Yao, Z., Kang, W., Yang, X., Kuang, F., Guo, L., Zhu, H., Jin, Z., Li, Z., Lin, L., Povey, D. 2025. *CR-CTC: Consistency regularization on CTC for improved speech recognition.* ICLR. https://arxiv.org/abs/2410.05101
112. Batra, K. S., Singh, P., Sood, A., Singh, J., et al. 2026. *Seeds Before Objectives: Rethinking Evaluation for Low-Resource Garhwali ASR.* ICNLSP. https://arxiv.org/abs/2608.10670
113. Gandhi, S., von Platen, P., Rush, A. M. 2023. *Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling.* arXiv:2311.00430. https://arxiv.org/abs/2311.00430
114. Waheed, A., Kadaoui, K., Raj, B., Abdul-Mageed, M. 2025. *uDistil-Whisper: Label-Free Data Filtering for Knowledge Distillation in Low-Data Regimes.* NAACL. https://arxiv.org/abs/2407.01257
115. Bhogale, K. S., Mehendale, D., Parasa, N., G, S. K. R., et al. 2024. *Empowering Low-Resource Language ASR via Large-Scale Pseudo Labeling (Pratinidhi).* Interspeech. https://arxiv.org/abs/2408.14026
116. Rangappa, P., Carofilis, A., Prakash, J., Kumar, S., et al. 2025. *Efficient Data Selection for Domain Adaptation of ASR Using Pseudo-Labels and Multi-Stage Filtering.* arXiv:2506.03681. https://arxiv.org/abs/2506.03681
117. Prakash, J., Kumar, B., Hacioglu, K., Sharma, B., et al. 2025. *Better Pseudo-labeling with Multi-ASR Fusion and Error Correction by SpeechLLM.* Interspeech. https://arxiv.org/abs/2506.11089
118. Berrebbi, D., Collobert, R., Bengio, S., Jaitly, N., Likhomanenko, T. 2023. *Continuous Pseudo-Labeling from the Start.* ICLR. https://arxiv.org/abs/2210.08711
119. Kim, C., Park, J., Shim, H., Yang, E. 2023. *SGEM: Test-Time Adaptation for Automatic Speech Recognition via Sequential-Level Generalized Entropy Minimization.* Interspeech. https://arxiv.org/abs/2306.01981
120. Lin, G.-T., Huang, W.-P., Lee, H.-y. 2024. *Continual Test-time Adaptation for End-to-end Speech Recognition on Noisy Speech (DSUTA).* EMNLP. https://arxiv.org/abs/2406.11064
121. Kamo, N., Tawara, N., Matsuura, K., Ashihara, T., Moriya, T., et al. 2023. *NTT Multi-Speaker ASR System for the DASR Task of CHiME-7 Challenge.* CHiME-2023 workshop. https://www.isca-archive.org/chime_2023/kamo23_chime.pdf
122. Kumar, S., Thorbecke, I., Burdisso, S., Villatoro-Tello, E., et al. 2025. *Performance evaluation of SLAM-ASR: The Good, the Bad, the Ugly, and the Way Forward.* ICASSP 2025 SALMA workshop. https://arxiv.org/abs/2411.03866
123. Kolluri, G. P. K. B., Kampouridis, M., Shekhar, R. 2026. *On the Role of Encoder Depth: Pruning Whisper and LoRA Fine-Tuning in SLAM-ASR.* arXiv:2603.27981. https://arxiv.org/abs/2603.27981
124. Song, Z., Zhuo, J., Yang, Y., Ma, Z., Zhang, S., Chen, X. 2024. *LoRA-Whisper: Parameter-Efficient and Extensible Multilingual ASR.* Interspeech. https://arxiv.org/abs/2406.06619
125. Qian, M., Tang, S., Ma, R., Knill, K. M., et al. 2024. *Learn and Don't Forget: Adding a New Language to ASR Foundation Models.* Interspeech. https://arxiv.org/abs/2407.06800
126. Xu, T., Huang, K., Guo, P., Zhou, Y., et al. 2024. *Towards Rehearsal-Free Multilingual ASR: A LoRA-based Case Study on Whisper.* arXiv:2408.10680. https://arxiv.org/abs/2408.10680
127. Biderman, D., et al. 2024. *LoRA Learns Less and Forgets Less.* TMLR. https://arxiv.org/abs/2405.09673
128. Xiao, Y., Holden, E.-J., Dang, T. 2026. *Adapting Where It Matters: Depth-Aware Adaptation for Efficient Multilingual Speech Recognition in Low-Resource Languages (DAMA).* arXiv:2602.01008. https://arxiv.org/abs/2602.01008
129. Peng, Y., Wang, B., Chao, Y.-W., Ma, Z., et al. 2025. *NTU Speechlab LLM-Based Multilingual ASR System for Interspeech MLC-SLM Challenge 2025.* arXiv:2506.13339. https://arxiv.org/abs/2506.13339
130. Mei, Y., Zheng, Y., Xu, D., Long, Y., et al. 2025. *SHNU Multilingual Conversational Speech Recognition System for INTERSPEECH 2025 MLC-SLM Challenge.* arXiv:2507.03343. https://arxiv.org/abs/2507.03343
131. Xue, H., Huang, K., Zhou, Z., Huang, S., et al. 2025. *The TEA-ASLP System for Multilingual Conversational Speech Recognition and Speech Diarization in MLC-SLM 2025 Challenge.* arXiv:2507.18051. https://arxiv.org/abs/2507.18051; MLC-SLM 2025 challenge summary: arXiv:2509.13785. https://arxiv.org/abs/2509.13785
132. Schmitt, R., Zeyer, A., Zeineldeen, M., Schlüter, R., et al. 2026. *LLMs and Speech: Integration vs. Combination.* arXiv:2603.15045. https://arxiv.org/abs/2603.15045
133. Kumar, R., Krishna, A., Ramakrishnan, G., Jyothi, P. 2026. *Post-ASR Correction in Hindi: Comparing Language Models and Large Language Models in Low-Resource Scenarios.* EACL (short). https://aclanthology.org/2026.eacl-short.45/
134. Li, S., Chen, C., Kwok, C. Y., Chu, C., Chng, E. S., et al. 2024. *Investigating ASR Error Correction with Large Language Model and Multilingual 1-best Hypotheses.* Interspeech. https://www.isca-archive.org/interspeech_2024/li24h_interspeech.html
135. Ma, R., Gales, M. J. F., Knill, K. M., Qian, M. 2023. *N-best T5: Robust ASR Error Correction using Multiple Input Hypotheses and Constrained Decoding Space.* Interspeech. https://arxiv.org/abs/2303.00456
136. Fang, Y., Peng, J., Li, X., Xi, Y., et al. 2025. *Low-Resource Domain Adaptation for Speech LLMs via Text-Only Fine-Tuning.* arXiv:2506.05671. https://arxiv.org/abs/2506.05671
137. Bhandari, V., Kumar, T., Mulhern, K. 2026. *Voices from the Margins: Modeling Linguistic Diversity in Spontaneous Speech for Low-Resource Languages.* ComputEL-9. https://aclanthology.org/2026.computel-1.5/
138. Meng, C., Anastasopoulos, A. 2025. *GMU Systems for the IWSLT 2025 Low-Resource Speech Translation Shared Task.* IWSLT. https://arxiv.org/abs/2505.21781
139. Kaggle Bengali.AI Speech Recognition 2023, top-3 solution write-ups: 1st https://www.kaggle.com/competitions/bengaliai-speech/discussion/447961 ; 2nd https://www.kaggle.com/competitions/bengaliai-speech/discussion/447976 ; 3rd https://www.kaggle.com/competitions/bengaliai-speech/discussion/447957
140. Li, X., Zhang, G., Zhu, A., Li, W., et al. 2023. *Transsion TSUP's Speech Recognition System for ASRU 2023 MADASR Challenge.* arXiv:2307.11778. https://arxiv.org/abs/2307.11778
141. Sullivan, P., Talafha, B., Ashraf, A., Bougares, F., et al. 2026. *NADI 2026: The Second Multidialectal Arabic Speech Processing Shared Task.* arXiv:2609.27086. https://arxiv.org/abs/2609.27086; Almajai, I. 2026. *Itgan at NADI 2026 shared task: Parameter-Efficient Whisper Adaptation for Robust, Mixed-Dialect and Code-Switched Arabic ASR.* arXiv:2610.09934. https://arxiv.org/abs/2610.09934
142. CHiME-8 DASR task rules. https://www.chimechallenge.org/challenges/chime8/task1/rules
143. Watanabe, S., Hori, T., Kim, S., Hershey, J. R., Hayashi, T. 2017. *Hybrid CTC/Attention Architecture for End-to-End Speech Recognition.* IEEE JSTSP 11(8):1240–1253. (Inspiration for joint CTC + LLM-ASR rescoring, §h.)
144. Prabhavalkar, R., Sainath, T. N., Wu, Y., Nguyen, P., Chen, Z., Chiu, C.-C., Kannan, A. 2018. *Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models.* ICASSP. https://arxiv.org/abs/1712.01818
145. Meng, Z., Wu, Y., Kanda, N., Lu, L., Chen, X., Ye, G., Sun, E., Li, J., Gong, Y. 2021. *Minimum Word Error Rate Training with Language Model Fusion for End-to-End Speech Recognition.* Interspeech. https://arxiv.org/abs/2106.02302
146. Shivakumar, P. G., Kolehmainen, J., Gourav, A., Gu, Y., Gandhe, A., Rastrow, A., Bulyko, I. 2024. *Speech Recognition Rescoring with Large Speech-Text Foundation Models.* arXiv:2409.16654. https://arxiv.org/abs/2409.16654
147. Vyas, A., Madikeri, S., Bourlard, H. 2021. *Lattice-Free MMI Adaptation of Self-Supervised Pretrained Acoustic Models.* ICASSP. https://arxiv.org/abs/2012.14252
148. Shivakumar, P. G., Gu, Y., Gandhe, A., Bulyko, I. 2025. *Group Relative Policy Optimization for Speech Recognition.* ASRU. https://arxiv.org/abs/2509.01939
149. Gao, C., Li, Y., An, K., Gao, Z., Du, Z., Zhao, H., Li, X. 2025. *Explore the Reinforcement Learning for the LLM based ASR and TTS system.* arXiv:2509.18569. https://arxiv.org/abs/2509.18569
150. Tongyi Fun Team (Alibaba). 2025. *FunAudio-ASR Technical Report.* arXiv:2509.12508. https://arxiv.org/abs/2509.12508
151. Zheng, C., Liu, S., Li, M., et al. 2025. *Group Sequence Policy Optimization (GSPO).* arXiv:2507.18071. https://arxiv.org/abs/2507.18071
152. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. 2017. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347. https://arxiv.org/abs/1707.06347
153. Shao, Z., Wang, P., Zhu, Q., et al. 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO).* arXiv:2402.03300. https://arxiv.org/abs/2402.03300
154. Ahmadian, A., Cremer, C., Gallé, M., et al. 2024. *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs (RLOO).* ACL. https://arxiv.org/abs/2402.14740
155. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., Finn, C. 2023. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS. https://arxiv.org/abs/2305.18290
156. Xiong, T., Chen, R., Li, W., Zhang, W., Xie, Q. 2026. *Grounded in Sound: Reinforcement Learning with a Frozen Acoustic Judge to Curb ASR Insertion Hallucinations.* IEEE SLT. https://arxiv.org/abs/2609.14455
157. Kumar, S., Labrak, Y., Watawana, H., Burdisso, S., Villatoro-Tello, E., Hacioğlu, K., Motlicek, P., Stolcke, A. 2026. *When Synthetic Speech Is All You Have: Better Call GRPO.* arXiv:2607.08409. https://arxiv.org/abs/2607.08409
158. Ye, Z., Vickers, P. 2026. *Reinforcement Learning for Data-Efficient Code-Switched ASR.* Interspeech. https://arxiv.org/abs/2607.02757
159. Nguyen Quang, T., Won, C. Y. L., Pham, M. D., He, Y., Sun, S., Aw, A. T. 2026. *Direct Preference Optimization for English-Mandarin Code-Switching Speech Recognition in Audio LLMs.* Interspeech. https://arxiv.org/abs/2605.23975
160. Ling, S., Ye, G. 2025. *Customizing Speech Recognition Model with Large Language Model Feedback.* arXiv:2506.11091. https://arxiv.org/abs/2506.11091
161. Nagpal, C., Venugopalan, S., Tobin, J., Ladewig, M., Heller, K., Tomanek, K. 2025. *Speech Recognition With LLMs Adapted to Disordered Speech Using Reinforcement Learning.* ICASSP. https://arxiv.org/abs/2501.00039
162. Zheng, H., Wang, S., Fan, X., Jin, Z., Hasegawa-Johnson, M. 2026. *FSA-GRPO: Teaching Auditory LLMs to Use Few-Shot Demonstrations.* arXiv:2606.02615. https://arxiv.org/abs/2606.02615
163. Su, Z., Zhang, Z., Liu, Z., et al. 2026. *Reinforcement Learning with Semantic Rewards Enables Low-Resource Language Expansion without Alignment Tax.* Findings of ACL. https://arxiv.org/abs/2605.14366
164. Nyssanov, N., Rzayeva, L., Batkuldin, A., Kozhakhmet, Z. 2026. *Evaluating ASR Pipeline Configurations for Kazakh: Implications for Low-Resource Turkic Languages.* Information 17(7):690. https://www.mdpi.com/2078-2489/17/7/690
165. Ghosh, S., Rasooli, M. S., Levit, M., Wang, P., Xue, J., Manocha, D., Li, J. 2024. *Failing Forward: Improving Generative Error Correction for ASR with Synthetic Data and Retrieval Augmentation.* arXiv:2410.13198. https://arxiv.org/abs/2410.13198
166. Lin, Y. B., Aung, T., Thu, Y. K., Oo, T. M. 2025. *ASR Error Correction in Low-Resource Burmese with Alignment-Enhanced Transformers using Phonetic Features.* iSAI-NLP. https://arxiv.org/abs/2511.21088
167. Jinnai, Y. 2025. *Re-evaluating Minimum Bayes Risk Decoding for Automatic Speech Recognition.* arXiv:2510.19471. https://arxiv.org/abs/2510.19471
168. Xue, L., Barua, A., Constant, N., Al-Rfou, R., Kale, M., Roberts, A., Narang, S., Raffel, C. 2022. *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.* TACL. https://arxiv.org/abs/2105.13626
169. GLM-5 Team (Zeng, A., Lv, X., Hou, Z., et al.) 2026. *GLM-5: from Vibe Coding to Agentic Engineering.* arXiv:2602.15763. https://arxiv.org/abs/2602.15763 (GRPO + IcePop with double-sided masking, dropping off-policy/noisy samples, hybrid reward system, on-policy cross-stage distillation; GLM-5.3 reuses the GLM-5.2 base with further post-training, https://github.com/zai-org/GLM-5.)
170. Zheng, C., et al. 2025. *Group Sequence Policy Optimization.* arXiv:2507.18071.
171. Shao, Z., et al. 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (introduces GRPO). arXiv:2402.03300.
172. Anthony, T., Tian, Z., Barber, D. 2017. *Thinking Fast and Slow with Deep Learning and Tree Search (Expert Iteration).* NeurIPS. https://arxiv.org/abs/1705.08439
173. Gulcehre, C., Paine, T. L., Srinivasan, S., Konyushkova, K., et al. 2023. *Reinforced Self-Training (ReST) for Language Modeling.* arXiv:2308.08998. https://arxiv.org/abs/2308.08998
174. Singh, A., Co-Reyes, J. D., Agarwal, R., Anand, A., et al. 2024. *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models (ReST-EM).* TMLR. https://arxiv.org/abs/2312.06585
175. Zelikman, E., Wu, Y., Mu, J., Goodman, N. D. 2022. *STaR: Bootstrapping Reasoning With Reasoning.* NeurIPS. https://arxiv.org/abs/2203.14465
176. Dong, H., Xiong, W., Goyal, D., Zhang, Y., et al. 2023. *RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment.* TMLR. https://arxiv.org/abs/2304.06767
177. Finkelstein, M., Naskar, S., Mirzazadeh, M., Shah, A., Freitag, M. 2024. *MBR and QE Finetuning: Training-time Distillation of the Best and Most Expensive Decoding Methods.* ICLR. https://arxiv.org/abs/2309.10966
178. Yang, G., Chen, J., Lin, W., Byrne, B. 2024. *Direct Preference Optimization for Neural Machine Translation with Minimum Bayes Risk Decoding.* NAACL. https://arxiv.org/abs/2311.08380
179. Hu, Y., Chen, C., Yang, C.-H. H., Qin, C., et al. 2024. *Self-Taught Recognizer: Toward Unsupervised Adaptation for Speech Foundation Models (STAR).* NeurIPS. https://arxiv.org/abs/2405.14161
180. Baskar, M. K., Rosenberg, A., Ramabhadran, B., Audhkhasi, K. 2023. *O-1: Self-training with Oracle and 1-best Hypothesis.* Interspeech. https://arxiv.org/abs/2308.07486
181. Lin, Y.-C., Liang, Y.-H. L., Su, H., Lin, T.-Q., et al. 2026. *Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition.* Findings of ACL. https://arxiv.org/abs/2510.08047
182. Asaria, A., Salomone, T., Gandhi, D. 2026. *Reliable Neural-Codec Text-to-Speech by ASR Self-Verification and Distillation: Near-Zero Catastrophic Failures Across Models and Codecs.* arXiv:2606.18323. https://arxiv.org/abs/2606.18323
183. Agarwal, R., Vieillard, N., Zhou, Y., Stanczyk, P., et al. 2024. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD).* ICLR. https://arxiv.org/abs/2306.13649
184. Lu, K., and Thinking Machines Lab. 2025. *On-Policy Distillation.* Blog post, 27 Oct 2025. https://thinkingmachines.ai/blog/on-policy-distillation/
185. Lin, Y., Wang, Y., Cai, R., Zeng, X. 2026. *Data-Efficient On-Policy Distillation for Automatic Speech Recognition.* arXiv:2605.28139. https://arxiv.org/abs/2605.28139
186. Cao, D., Fu, D., Yu, H., Zheng, S., et al. 2026. *X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs.* Interspeech. https://arxiv.org/abs/2603.24596
187. Li, Y., Zuo, Y., He, B., Zhang, J., et al. 2026. *Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.* arXiv:2604.13016. https://arxiv.org/abs/2604.13016
188. Wang, S., Chen, X., Xu, Y. 2025. *Self-Improvement for Audio Large Language Model using Unlabeled Speech (SI-SDA).* Interspeech. https://arxiv.org/abs/2507.20169
189. Guo, X., Li, N., Zeng, Z., Wang, K., et al. 2026. *From Oracle to Noisy Context: Mitigating Contextual Exposure Bias in Speech-LLMs.* arXiv:2603.24034. https://arxiv.org/abs/2603.24034
190. Nguyen, T. X., Truong, H. M., Nguyen, G. S., Vo, N., et al. 2026. *Contrastive Training with LLM-generated Near-Misses for Robust Code-Switching Speech Recognition.* Interspeech. https://arxiv.org/abs/2606.06985
191. Wu, Y., Lu, Y., Peng, Y., Wang, X., et al. 2025. *Enhancing Audiovisual Speech Recognition through Bifocal Preference Optimization.* AAAI. https://arxiv.org/abs/2412.19005
192. Cosgrove, B., Grace, A. I., Wang, W. 2026. *Hallucination Reduction for LLM-Based Audio Understanding via Multimodal Direct Preference Optimization.* arXiv:2610.04004. https://arxiv.org/abs/2610.04004
193. Hussain, S., Neekhara, P., Yang, X., Casanova, E., et al. 2025. *Koel-TTS: Enhancing LLM based Speech Generation with Preference Alignment and Classifier Free Guidance.* EMNLP. https://arxiv.org/abs/2502.05236
194. Pal, A., Karkhanis, D., Dooley, S., Roberts, M., et al. 2024. *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive.* arXiv:2402.13228. https://arxiv.org/abs/2402.13228
195. Razin, N., Malladi, S., Bhaskar, A., Chen, D., et al. 2025. *Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization.* ICLR. https://arxiv.org/abs/2410.08847
196. Lanchantin, J., Chen, A., Lan, J., Li, X., et al. 2025. *Bridging Offline and Online Reinforcement Learning for LLMs.* arXiv:2506.21495. https://arxiv.org/abs/2506.21495
197. Xie, Z., Pang, K., Zhang, H., Ye, D., Hu, X., Yan, S., Miao, C. 2026. *Mega-ASR: Towards In-the-wild² Speech Recognition via Scaling up Real-world Acoustic Simulation.* arXiv:2605.19833. https://arxiv.org/abs/2605.19833
198. Bai, Y., Chen, J., Chen, J., Chen, W., et al. (ByteDance) 2024. *Seed-ASR: Understanding Diverse Speech and Contexts with LLM-based Speech Recognition.* arXiv:2407.04675. https://arxiv.org/abs/2407.04675
199. Hwang, D., Ryu, C., Sim, K. C. 2023. *Edit Distance based RL for RNNT decoding.* Interspeech. https://arxiv.org/abs/2306.01789
200. Ichihara, Y., Jinnai, Y., Morimura, T., Sakamoto, M., et al. 2025. *MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems.* TACL. https://arxiv.org/abs/2509.22047
201. Fang, L., Xie, T., Liu, L. 2026. *Boosting ASR Robustness via Test-Time Reinforcement Learning with Audio-Text Semantic Rewards (ASR-TRA).* arXiv:2603.05231. https://arxiv.org/abs/2603.05231
202. Wu, H., Han, R., Wang, Z., Liang, W., et al. 2026. *Diarization-Guided Qwen-ASR Adaptation for Multilingual Two-Speaker Conversational Speech.* Interspeech 2026 MLC-SLM Workshop. https://arxiv.org/abs/2607.08208
203. Novosad, I. 2026. *The Anatomy of the CTC Oracle Gap: Acoustic Exhaustion and Linguistic Recovery.* arXiv:2606.23306. https://arxiv.org/abs/2606.23306
204. Zhou, Y., Xiong, C., Socher, R. 2018. *Improving End-to-End Speech Recognition with Policy Learning.* ICASSP. https://arxiv.org/abs/1712.07101
205. Segev, E., Alroy, M., Katsir, R., Wies, N., et al. 2024. *Align With Purpose: Optimize Desired Properties in CTC Models with a General Plug-and-Play Framework.* ICLR. https://arxiv.org/abs/2307.01715
206. Wu, T.-W., Chen, I-F., Gandhe, A. 2022. *Learning to rank with BERT-based confidence models in ASR rescoring.* Interspeech. https://www.isca-archive.org/interspeech_2022/wu22_interspeech.pdf
207. Xu, L., Gu, Y., Kolehmainen, J., Khan, H., et al. 2022. *RescoreBERT: Discriminative Speech Recognition Rescoring with BERT.* ICASSP. https://arxiv.org/abs/2202.01094
208. Gurunath Shivakumar, P., Kolehmainen, J., Gu, Y., Gandhe, A., et al. 2023. *Discriminative Speech Recognition Rescoring with Pre-trained Language Models.* ASRU. https://arxiv.org/abs/2310.06248
209. Yu, Y., Yang, C.-H. H., Kolehmainen, J., Shivakumar, P. G., et al. 2023. *Low-rank Adaptation of Large Language Model Rescoring for Parameter-Efficient Speech Recognition (LoRB).* ASRU. https://arxiv.org/abs/2309.15223
210. Kang, I. E., Van Gysel, C., Siu, M.-H. 2024. *Transformer-based Model for ASR N-Best Rescoring and Rewriting.* Interspeech. https://arxiv.org/abs/2406.08207
211. Li, Z., Niehues, J. 2025. *In-context Language Learning for Endangered Languages in Speech Recognition.* Interspeech. https://arxiv.org/abs/2505.20445
212. Zuo, Y., Zhang, K., Sheng, L., Qu, S., et al. 2025. *TTRL: Test-Time Reinforcement Learning.* arXiv:2504.16084. https://arxiv.org/abs/2504.16084
213. Shafayat, S., Tajwar, F., Salakhutdinov, R., Schneider, J., et al. 2025. *Can Large Reasoning Models Self-Train?* arXiv:2505.21444. https://arxiv.org/abs/2505.21444
214. Zhang, Y., Zhang, Z., Guan, H., Cheng, Y., et al. 2025. *No Free Lunch: Rethinking Internal Feedback for LLM Reasoning.* arXiv:2506.17219. https://arxiv.org/abs/2506.17219
215. Shao, R., Li, S. S., Xin, R., Geng, S., et al. 2025. *Spurious Rewards: Rethinking Training Signals in RLVR.* arXiv:2506.10947. https://arxiv.org/abs/2506.10947
216. Yu, Z., Su, W., Tao, L., Wang, H., et al. 2025. *RESTRAIN: From Spurious Votes to Signals — Self-Driven RL with Self-Penalization.* arXiv:2510.02172. https://arxiv.org/abs/2510.02172
217. Zhang, Z., Zhu, J., Ge, X., Zhao, Z., et al. 2026. *Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models.* ICLR. https://arxiv.org/abs/2508.00410
218. Liu, N. 2026. *LLMs as a Jury: Cross-Model Consensus Can Outperform Process Reward Models for LLM Reasoning.* arXiv:2607.10139. https://arxiv.org/abs/2607.10139
219. Zhao, X., Kang, Z., Feng, A., Levine, S., et al. 2026. *Learning to Reason without External Rewards (INTUITOR).* ICLR. https://arxiv.org/abs/2505.19590
220. Agarwal, S., Zhang, Z., Yuan, L., Han, J., et al. 2025. *The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning.* arXiv:2505.15134. https://arxiv.org/abs/2505.15134
221. Prabhudesai, M., Chen, L., Ippoliti, A., Fragkiadaki, K., et al. 2025. *Maximizing Confidence Alone Improves Reasoning (RENT).* arXiv:2505.22660. https://arxiv.org/abs/2505.22660
222. Roy, S., Hajimirsadeghi, H., Zhai, M., Samei, G. 2025. *You Need Reasoning to Learn Reasoning: The Limitations of Label-Free RL in Weak Base Models.* NeurIPS 2025 MATH-AI Workshop. https://arxiv.org/abs/2511.04902
223. Yu, Y., He, L., Liang, J., Guo, K., et al. 2026. *Understanding and Mitigating Spurious Signal Amplification in Test-Time Reinforcement Learning for Math Reasoning (DDRL).* Findings of ACL. https://arxiv.org/abs/2604.21327
224. Lin, H., Kuai, Z., Xue, E., Wang, L., et al. 2026. *Detecting and Mitigating the Correct-Answer Extinction Window in Test-Time Reinforcement Learning with Majority Voting.* arXiv:2605.19444. https://arxiv.org/abs/2605.19444
225. Zhang, H., Guo, J., Yang, D., Iwasawa, Y., et al. 2026. *AQA-TTRL: Self-Adaptation in Audio Question Answering with Test-Time Reinforcement Learning.* Interspeech. https://arxiv.org/abs/2510.05478
226. Chu, T., Zhai, Y., Yang, J., Tong, S., et al. 2025. *SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training.* ICML. https://arxiv.org/abs/2501.17161
227. Shenfeld, I., Pari, J., Agrawal, P. 2025. *RL's Razor: Why Online Reinforcement Learning Forgets Less.* arXiv:2509.04259. https://arxiv.org/abs/2509.04259
228. Chen, H., Razin, N., Narasimhan, K., Chen, D. 2025. *Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting.* arXiv:2510.18874. https://arxiv.org/abs/2510.18874
229. Mukherjee, S., Yuan, L., Hakkani-Tür, D., Peng, H. 2025. *Reinforcement Learning Finetunes Small Subnetworks in Large Language Models.* NeurIPS. https://arxiv.org/abs/2505.11711
230. Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., et al. 2025. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* arXiv:2503.14476. https://arxiv.org/abs/2503.14476
231. Bae, S., Hong, J., Lee, M. Y., Kim, H., et al. 2026. *Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning.* EACL. https://arxiv.org/abs/2504.03380
232. Guttmann, K., Pokrywka, M., Charkiewicz, A., Nowakowski, A. 2024. *Chasing COMET: Leveraging Minimum Bayes Risk Decoding for Self-Improving Machine Translation.* EAMT. https://arxiv.org/abs/2405.11937
233. Tseng, Parcollet, van Dalen, Zhang, Bhattacharya. 2025. (LLM pretraining contamination with CV/LibriSpeech transcripts.) arXiv:2505.22251. https://arxiv.org/abs/2505.22251
234. de Zuazo et al. 2025. *Whisper-LM.* arXiv:2503.23542. https://arxiv.org/abs/2503.23542
235. Akeret. 2026. (Swiss German ASR benchmark contamination.) arXiv:2606.07608. https://arxiv.org/abs/2606.07608
236. Xiong et al. 2026. *Grounded in Sound.* SLT 2026. arXiv:2609.14455. https://arxiv.org/abs/2609.14455
237. Fang, Xie, Liu. 2026. *ASR-TRA* (test-time RL for ASR). AAAI 2026. arXiv:2603.05231. https://arxiv.org/abs/2603.05231
238. Ling, Ye. 2025. (LLM-scored rewards for ASR RL without labels.) arXiv:2506.11091. https://arxiv.org/abs/2506.11091
239. (Hallucination in CTC vs. attention ASR.) 2025. arXiv:2501.11378. https://arxiv.org/abs/2501.11378
240. GLM-4.5 Team (Zeng, A., Lv, X., Zheng, Q., et al.) 2025. *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models.* arXiv:2508.06471. https://arxiv.org/abs/2508.06471 (expert iteration + unified self-distillation; difficulty curriculum with pass@8 = 0 / pass@512 ≫ 0; GRPO without KL; token-weighted mean loss for code RL; dynamic sampling temperature; slime.)
241. THUDM / Zhipu AI. 2025. *slime: an LLM post-training framework for RL scaling* (software; used for GLM-4.5 and later). https://github.com/THUDM/slime
242. GLM-V Team (Hong, W., Yu, W., Gu, X., et al.) 2025. *GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning* (RL with Curriculum Sampling, RLCS). arXiv:2507.01006. https://arxiv.org/abs/2507.01006
243. Cui, J., Yang, Z., Li, N., Tian, J., Ma, X., et al. 2025. *GLM-TTS Technical Report.* arXiv:2512.14291. https://arxiv.org/abs/2512.14291
244. DeepSeek-AI (Guo, D., Yang, D., Zhang, H., Song, J., et al.) 2025. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948. https://arxiv.org/abs/2501.12948
245. DeepSeek-AI (Liu, A., Mei, A., Lin, B., Xue, B., et al.) 2025. *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models.* arXiv:2512.02556. https://arxiv.org/abs/2512.02556
246. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., et al. 2025. *Qwen3 Technical Report.* arXiv:2505.09388. https://arxiv.org/abs/2505.09388
247. Kimi Team (Du, A., Gao, B., Xing, B., Jiang, C., et al.) 2025. *Kimi k1.5: Scaling Reinforcement Learning with LLMs.* arXiv:2501.12599. https://arxiv.org/abs/2501.12599
248. Kimi Team (Bai, Y., Bao, Y., Charles, Y., Chen, C., et al.) 2025. *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534. https://arxiv.org/abs/2507.20534
249. MiniMax (Chen, A., Li, A., Gong, B., et al.) 2025. *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention* (introduces CISPO). arXiv:2506.13585. https://arxiv.org/abs/2506.13585
250. Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., et al. 2025. *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO). arXiv:2503.20783. https://arxiv.org/abs/2503.20783
251. Yue, Y., Yuan, Y., Yu, Q., Zuo, X., Zhu, R., et al. 2025. *VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks.* arXiv:2504.05118. https://arxiv.org/abs/2504.05118
252. Ling Team (Shen, A., Li, B., Hu, B., Jing, B., et al.) 2025. *Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model* (Ring-1T; IcePop). arXiv:2510.18855. https://arxiv.org/abs/2510.18855
253. Cui, G., Zhang, Y., Chen, J., Yuan, L., Wang, Z., et al. 2025. *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models* (Clip-Cov, KL-Cov). arXiv:2505.22617. https://arxiv.org/abs/2505.22617
254. Wang, S., Yu, L., Gao, C., Zheng, C., Liu, S., et al. 2025. *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning.* arXiv:2506.01939. https://arxiv.org/abs/2506.01939
255. Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., et al. 2025. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* arXiv:2504.13837. https://arxiv.org/abs/2504.13837
256. Zhu, X., Xia, M., Wei, Z., Chen, W.-L., Chen, D., et al. 2025. *The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning.* arXiv:2506.01347. https://arxiv.org/abs/2506.01347
257. Chen, Z., Qin, X., Wu, Y., Ling, Y., Ye, Q., et al. 2025. *Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models.* arXiv:2508.10751. https://arxiv.org/abs/2508.10751
258. Chow, Y., Tennenholtz, G., Gur, I., Zhuang, V., Dai, B., et al. 2024. *Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models.* arXiv:2412.15287. https://arxiv.org/abs/2412.15287
259. Bagirov, F., Arkhipov, M., Sycheva, K., Glukhov, E., Bogomolov, E. 2025. *The Best of N Worlds: Aligning Reinforcement Learning with Best-of-N Sampling via max@k Optimisation.* arXiv:2510.23393. https://arxiv.org/abs/2510.23393
260. Yan, J., Li, Y., Hu, Z., Wang, Z., Cui, G., et al. 2025. *Learning to Reason under Off-Policy Guidance (LUFFY).* arXiv:2504.14945. https://arxiv.org/abs/2504.14945
261. Zhang, W., Xie, Y., Sun, Y., Chen, Y., Wang, G., et al. 2025. *On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting (CHORD).* arXiv:2508.11408. https://arxiv.org/abs/2508.11408
262. Limozin, A., Durech, E., Hoefler, T., Schlag, I., Pyatkin, V. 2026. *SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning.* arXiv:2604.23747. https://arxiv.org/abs/2604.23747
263. Elmakies, A., Aronowitz, H., Shabtay, N., Schwartz, E., Hoory, R., et al. 2025. *Advancing Speech Understanding in Speech-Aware Language Models with GRPO.* arXiv:2509.16990. https://arxiv.org/abs/2509.16990
264. Chen, Z., Deng, Y., Yuan, H., Ji, K., Gu, Q. 2024. *Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (SPIN).* ICML. https://arxiv.org/abs/2401.01335
265. Tjandra, A., Sakti, S., Nakamura, S. 2017. *Listening while Speaking: Speech Chain by Deep Learning.* ASRU. https://arxiv.org/abs/1707.04879
266. Zhang, L., Hosseini, A., Bansal, H., Kazemi, M., Kumar, A., et al. 2024. *Generative Verifiers: Reward Modeling as Next-Token Prediction (GenRM).* arXiv:2408.15240. https://arxiv.org/abs/2408.15240
267. Liu, Z., Wang, P., Xu, R., Ma, S., Ruan, C., et al. 2025. *Inference-Time Scaling for Generalist Reward Modeling (DeepSeek-GRM).* arXiv:2504.02495. https://arxiv.org/abs/2504.02495
268. Li, Q., Qiu, D., Zhang, Y., Li, B., He, Y., et al. 2021. *Confidence Estimation for Attention-based Sequence-to-sequence Models for Speech Recognition.* ICASSP. https://arxiv.org/abs/2010.11428
269. Snell, C., Lee, J., Xu, K., Kumar, A. 2024. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* arXiv:2408.03314. https://arxiv.org/abs/2408.03314
270. Chung, H. L., Chen, Y., Lyu, D.-C., Hung, H.-T., Lee, H.-y. 2026. *Listen, Think, Transcribe: Continuous Latent Test-Time Scaling for ASR (LatentASR).* arXiv:2607.05051. https://arxiv.org/abs/2607.05051
271. Hsu, C.-J., Kim, J., Yang, C.-H. H., Watanabe, S., Lee, H.-y., et al. 2026. *Listen to the Latents: Self-Correcting Speech Recognition in Large Audio Language Models Through Hidden-State Interactions (Hybrid Search).* arXiv:2609.02940. https://arxiv.org/abs/2609.02940
272. Kaloga, Y., Kumar, S., Sheikh, S. A., Khalil, D., Motlicek, P., et al. 2026. *Test-Time Compute Scaling for ASR with Depth-Conditioned Looped Transformers (LARM).* arXiv:2606.04678. https://arxiv.org/abs/2606.04678
273. Yuan, Z., Yuan, H., Li, C., Dong, G., Lu, K., et al. 2023. *Scaling Relationship on Learning Mathematical Reasoning with Large Language Models* (rejection-sampling fine-tuning, RFT). arXiv:2308.01825. https://arxiv.org/abs/2308.01825
274. Shenfeld, I., Damani, M., Hübotter, J., Agrawal, P. 2026. *Self-Distillation Enables Continual Learning (SDFT).* arXiv:2601.19897. https://arxiv.org/abs/2601.19897
275. Tian, K., Liu, S., Jiang, T., Dong, S., Li, Y., et al. 2026. *What Should a Self-Teacher See? Privileged Context Design for On-Policy Self-Distillation.* arXiv:2609.25623. https://arxiv.org/abs/2609.25623
276. Liu, Y., Zhang, S., Zhang, Y., Gu, Q. 2026. *Self-Distilled Policy Gradient (SDPG).* arXiv:2606.04036. https://arxiv.org/abs/2606.04036
277. Li, J., Seltzer, M. L., Wang, X., Zhao, R., Gong, Y. 2017. *Large-Scale Domain Adaptation via Teacher-Student Learning.* Interspeech. https://arxiv.org/abs/1708.05466
278. Schulman, J., and Thinking Machines Lab. 2025. *LoRA Without Regret.* Blog post, 29 Sep 2025. https://thinkingmachines.ai/blog/lora/
279. Qi, P., Liu, Z., Zhou, X., Pang, T., Du, C., et al. 2025. *Defeating the Training-Inference Mismatch via FP16.* arXiv:2510.26788. https://arxiv.org/abs/2510.26788
280. Xie, Y., Song, J., Qiu, G., Wang, X., Qiao, K., et al. 2026. *NIM4-ASR: Towards Efficient, Robust, and Customizable Real-Time LLM-Based ASR.* arXiv:2604.18105. https://arxiv.org/abs/2604.18105
281. Parashar, S., Gui, S., Li, X., Ling, H., Vemuri, S., et al. 2026. *Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning (E2H).* ICLR. https://arxiv.org/abs/2506.06632
282. Liu, F., Yin, Y., Shi, P., Yang, S., Zeng, Z., et al. 2026. *Length-Unbiased Sequence Policy Optimization: Revealing and Controlling Response Length Variation in RLVR (LUSPO).* arXiv:2602.05261. https://arxiv.org/abs/2602.05261
283. Shen, M., Jayashankar, T., Hanna, O., Kanda, N., Wang, Y., et al. 2026. *GSRM: Generative Speech Reward Model for Speech RLHF.* arXiv:2602.13891. https://arxiv.org/abs/2602.13891
284. Zheng, X. 2026. *Look Less, Hear Better: Jointly Rewarded GRPO for Streaming ASR.* arXiv:2609.18333. https://arxiv.org/abs/2609.18333

## §k. Where to adapt LLM-ASR, and how to look inside (literature check, 2026-10-16)

Every entry was verified by a literature agent; the EAVA entry was re-checked directly on arXiv. Recent preprints are flagged as less established.

**Encoder vs LLM adaptation**
- **EAVA** (Shi, Wang, Balaji Shankar, Zhang, Eren, Alwan; IEEE SLT 2026; arXiv:2609.17981). Training only the LLM leaves the speech encoder under-adapted. Encoder adapters first, then LLM LoRA, wins on children's and dialectal speech. **The WER-level claim "adapt the encoder" is therefore not new.**
- Kolluri et al. (SPEAKABLE@LREC 2026; arXiv:2603.27981): LLM LoRA works through linguistic priors and gains little for lower-resource Danish.
- Verdini et al. (Interspeech 2025; arXiv:2409.17044): the choice of speech encoder matters more than the LLM or the connector. Consistent with our null result for the connector.
- MMS (Pratap et al., JMLR 2024): language capacity sits in per-layer encoder adapters.
- Prior Puno Quechua ASR to cite: Huaman et al. (AmericasNLP 2026; arXiv:2605.28253).

**Layer-wise SSL analyses**
- Pasad et al. (ASRU 2021, arXiv:2107.04734; ICASSP 2023, arXiv:2211.03929; TACL 2024): phonetic and word information peaks in the middle layers; ASR fine-tuning rewrites the top layers.
- Chung, Belinkov, Glass (ICASSP 2021): CCA/CKA comparisons of speech representations.
- Choi et al. (Interspeech 2024): SSL speech representations are more phonetic than semantic.

**Fine-tuning dynamics and LoRA geometry**
- Merchant et al. (BlackboxNLP 2020): fine-tuning mostly changes the top layers and has less effect out of domain.
- LoRA paper §7 (Hu et al., ICLR 2022): seeds and ranks share only their top singular directions; the update amplifies directions that are weak in the pretrained weights.
- Aghajanyan et al. (ACL 2021): low intrinsic dimension.
- Shuttleworth et al. (NeurIPS 2025; arXiv:2410.21228): LoRA "intruder dimensions" carry the forgetting.

**Interpretability of speech models**
- Glazer et al. (arXiv:2508.15882, AAAI 2026 per the proceedings listing): logit lens, encoder lens, probes and activation patching on Whisper / Qwen2-Audio.
- Ma et al. (ICASSP 2026; arXiv:2509.08454): CKA, SVD and logit lens on LoRA-adapted Whisper; "delayed specialization" with depth.
- Martin et al. (Interspeech 2023): aspiration is linearly encoded early, in a low-dimensional subspace. This predicts our null result for aspiration.
- Millet & Dunbar (ACL 2022) and Rodriguez et al. (CoNLL 2024): SSL models fail on fine-grained non-native contrasts.
- Pouw et al. (arXiv:2406.15265, Computational Linguistics): causal interventions in wav2vec2.
- No probing study of ejectives or of the uvular/velar contrast was found.

**Errors and oracles**
- Novosad (arXiv:2606.23306, single-author preprint): for English CTC, the N-best ranking bottleneck is linguistic. In our low-resource case, it is acoustic.
- Akavarapu et al. (Findings of ACL 2026): phoneme accuracy follows log training frequency. Control for this confound.

**Methods to adopt**
- Linear CKA on frame states of the same utterances, per 16-layer block, with a seed-vs-seed noise floor and an OOD repetition. Descriptive only: CKA can be gamed (Davari et al., ICLR 2023).
- Linear probes on force-aligned segments for ejective/plain, q/k and aspirated/plain, with held-out speakers, balanced classes, control tasks (Hewitt & Liang 2019) and frequency matching.
- Causal tests: switch the LoRA update off per block (and only-one-block on), scored by WER and per-contrast ranking accuracy; amnesic probing (Elazar et al. 2021) and activation patching.
- Spectral: effective rank, cross-seed subspace overlap, intruder dimensions.

**Our novel claim relative to this literature:** in speech-encoder + LLM ASR for an unseen language, encoder adaptation leaves the reachable hypotheses unchanged (equal oracle) and improves their ranking (+7.3 pairwise points on read speech, significant for whole-word differences). For ejectives and q/k the gain is positive but not significant under the speaker × sentence bootstrap; aspiration does not improve. The literature suggests frozen encoders already encode aspiration.

**Our LoRA geometry** (`scripts/lora_geometry.py`, `reports/lora_geometry.csv`; enc+dec seed 0 vs seed 1, by encoder quarter, bottom → top):
- relative size ‖ΔW‖/‖W‖: 0.014 / 0.019 / 0.024 / 0.023;
- effective rank (of 16): 6.5 / 6.3 / 5.3 / 5.0;
- cross-seed top-4 subspace overlap: 0.022 / 0.053 / 0.083 / 0.079, against a random baseline of 0.002.
- So the updates are small, grow and concentrate with depth, and reproduce across seeds (10–40× random) in the upper half.
