# Every public Puno Quechua (qxp) ASR system on sets that are clean for all of them

Benchmarked 2026-10-09/10. Script: `scripts/bench_all_systems.py`. Earlier organizer-only
benchmark: `reports/benchmarks/organizer_baselines.md`.

**Bottom line:** on the OOD radio drama, the one set no system has trained on, every one of our systems beats every public system. Our best (LLM-7B + LoRA v2 + text-CPT) scores **14.27 / 2.22** WER / CER and joint_final scores 14.58 / 2.49. The strongest public system is zero-shot **omniASR-LLM-1B-v2 at 20.85 / 2.94**, so ours has 32 % lower WER and 24 % lower CER. Zero-shot omniASR-LLM-7B-v2 is next at 23.26 / 4.04. The best fine-tuned public checkpoint, the organizers' `xls-r-cpt-qxp-cv26-silver`, scores 24.47 / 4.09, and the best community fine-tune (`ctaguchi/ssc-qxp-mms-model-mix-adapt-max-lowlr`, MMS-1B) scores 31.19 / 5.17. The ranking shifts on `dev_spont<=39`, because the organizers' checkpoints were trained on those clips (4.0–12.6 % WER there). Among systems with no known overlap, ours (13.63 / 2.38) is only slightly ahead of zero-shot omniASR-LLM-7B (14.25 / 2.45) and well ahead of everything else (≥ 18.75 WER). Our systems were tuned on this set, though, so that margin is an upper bound. Of the 37 public systems we evaluated, 11 are unusable on this speech (OOD WER ≥ 88 %).

## Setup

| | |
|---|---|
| Primary set (clean for everyone) | `role == "ood"`: the 109 organizer radio-drama clips (15.9 min, max 21 s). The organizers use it as their own out-of-domain set, so they did not train on it. It became public with the organizer repo in 2026, after most community checkpoints were trained. We never trained or tuned on it. |
| Secondary set (flagged) | `role == "dev_spont"` with `duration_s <= 39`: 295 clips, 87.9 min, 4 speakers. The organizers' cv26 checkpoints were trained on 307 of the 310 `dev_spont` clips, and their older qxp_v2 split contains about half of them. Community models trained on the Spontaneous Speech (SPS) corpus may have seen them too (300 of the clips are in SPS's official `dev` split). Our systems were never trained on these speakers, but `dev_spont` was used to select models and tune rescoring weights. |
| Never used | `role == "test"` (data-use agreement). `eval_set()` raises an error if a test clip ever enters a set. |
| Scoring | Identical for every row. Hypotheses go through `qxp_asr.text.normalize` (curated alphabet `[a-zñ' ]`, accents folded, bracketed tags dropped). Micro WER / CER is computed with `qxp_asr.metrics.wer_cer` against the manifest's curated reference. Scores on raw (unnormalized) hypotheses are in `scores.csv` (`wer_raw`/`cer_raw`). Normalization lowers WER by 0–3 points for usable systems; it matters most for omniASR (casing, punctuation, accents). |
| Decoding | Greedy throughout. **omniASR**: the official `ASRInferencePipeline(model_card=…, device="cpu", dtype=torch.float32)` with `lang=["qxp_Latn"]`. The LLM pipeline's default beam is `nbest=1`, i.e. greedy. **HF CTC** (wav2vec2, MMS, w2v-BERT): argmax, one utterance at a time so padding cannot affect the output. MMS-1b-all loads the `quz`/`quy` adapter with `target_lang`. **Whisper**: `generate(num_beams=1)` with each checkpoint's own generation config (language/task), in 30 s windows. **fairseq checkpoints** (organizers, Adriatogi, ayayon): converted in memory with `bench_organizer.fairseq_to_hf`, per-utterance waveform layer-norm, greedy CTC. |
| Hardware | CPU only (`CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, `torch.set_num_threads(4)`, one job per server at a time) on both GPU boxes while they trained. Each job had a 45 min cap on the OOD set. |
| Our systems | Not re-run. Rows are scored from the stored per-clip predictions (`artifacts/preds/{joint_final,joint_v3,llm7b_lora_v1,llm7b_lora_v2}.parquet` on server A, `llm7b_textcpt.parquet` on server B) with the same scorer. They reproduce `reports/RESULTS.md` exactly: OOD joint_final 14.58 / 2.49, joint_v3 15.11 / 2.46, text-CPT 14.27 / 2.22. The zero-shot omniASR-LLM-7B-v2 row is our stored pipeline run (`zs_llm7b_all`, `raw` column); after normalization it is identical to the 23.26 / 4.04 in RESULTS. |

## Results — WER / CER (%), sorted by OOD WER

| # | system | params | training-data overlap: OOD / dev_spont | OOD WER / CER | dev_spont<=39 WER / CER |
|---:|---|---:|---|---:|---:|
| 1 | **Ours**: LLM-7B + LoRA v2 + decoder text-CPT | 7.8B | none / selection | 14.27 / 2.22 | 14.27 / 2.42 |
| 2 | **Ours**: joint_final (CTC-1B + LoRA v1/v2 + zero-shot 7B, joint rescoring) | ~8.8B | none / tuned | 14.58 / 2.49 | 13.63 / 2.38 |
| 3 | **Ours**: LLM-7B + LoRA v1 | 7.8B | none / selection | 14.88 / 2.73 | 14.41 / 2.43 |
| 4 | **Ours**: joint_v3 (dev-mode models) | ~8.8B | none / tuned | 15.11 / 2.46 | 13.82 / 2.35 |
| 5 | **Ours**: LLM-7B + LoRA v2 | 7.8B | none / selection | 15.11 / 2.53 | 14.45 / 2.44 |
| 6 | omniASR-LLM-1B-v2 zero-shot | 2.28B | none / unknown | 20.85 / 2.94 | 18.75 / 3.13 |
| 7 | omniASR-LLM-7B-v2 zero-shot | 7.8B | none / unknown | 23.26 / 4.04 | 14.25 / 2.45 |
| 8 | omniASR-LLM-300M-v2 zero-shot | 1.63B | none / unknown | 23.49 / 3.74 | not run |
| – | omniASR-LLM-3B-v2 zero-shot | 4.38B | none / unknown | skipped: > 40 min on CPU | not run |
| 9 | Organizers: xls-r-cpt-qxp-cv26-silver | 315M | none / known: 307/310 clips in train | 24.47 / 4.09 | 9.70 / 1.40 (leaked) |
| 10 | ayayon/asr-puno-quechua-curated (organizer CPT recipe, qxp_v2 train_silver) | 315M | unknown (not in recipe) / likely (organizer qxp_v2 silver train) | 24.70 / 3.97 | 13.47 / 1.95 (leaked) |
| 11 | Organizers: xls-r-qxp-cv26-silver | 315M | none / known: 307/310 clips in train | 26.06 / 4.02 | 12.59 / 1.86 (leaked) |
| 12 | Organizers: xls-r-qxp-cv26-validated | 315M | none / known: 307/310 clips in train | 30.89 / 5.17 | 5.94 / 0.97 (leaked) |
| 13 | Organizers: xls-r-cpt-qxp-cv26-validated | 315M | none / known: 307/310 clips in train | 31.04 / 5.18 | 4.03 / 0.75 (leaked) |
| 14 | ctaguchi/ssc-qxp-mms-model-mix-adapt-max-lowlr | 965M | none (Dec 2025) / possible (SPS qxp) | 31.19 / 5.17 | 35.05 / 5.62 (?) |
| 15 | ctaguchi/ssc-qxp-mms-model-mix-adapt-max | 965M | none (Dec 2025) / possible (SPS qxp) | 31.80 / 4.92 | 34.39 / 5.52 (?) |
| 16 | Adriatogi/wav2vec2-puno-quechua-ft-cpt-validated (organizer CPT recipe, qxp_v2 train) | 315M | unknown (not in recipe) / likely (organizer qxp_v2 train) | 32.85 / 5.76 | 9.74 / 1.58 (leaked) |
| 17 | MMS-1b-all, `quz` (Cusco) adapter | 965M | none / none (no qxp data) | 34.21 / 5.49 | 46.78 / 8.51 |
| 18 | omniASR-CTC-3B-v2 zero-shot | 3.08B | none / unknown | 34.97 / 5.30 | 23.46 / 3.88 |
| 19 | omniASR-CTC-1B-v2 zero-shot | 976M | none / unknown | 35.42 / 5.27 | 31.50 / 5.98 |
| 20 | ctaguchi/ssc-qxp-mms-model-mix-adapt-max-longcv2 | 965M | none (Dec 2025) / possible (SPS qxp) | 38.22 / 6.14 | not run |
| 21 | Organizers: whisper-base-qxp-finetuned | 73M | none / likely (older split: 161/310 clips) | 38.37 / 7.22 | 24.34 / 9.98 (leaked) |
| 22 | robertp408/mms-1b-all-qxp | 965M | none (Nov 2025) / possible (1,502 MDC qxp clips) | 39.20 / 6.10 | 38.40 / 6.00 (?) |
| 23 | ctaguchi/ssc-qxp-mms-model-mix-adapt-max2-2 | 965M | none (Dec 2025) / possible (SPS qxp) | 39.35 / 6.21 | 41.39 / 6.89 (?) |
| 24 | omniASR-CTC-300M-v2 zero-shot | 326M | none / unknown | 40.71 / 5.89 | 38.01 / 6.15 |
| 25 | MMS-1b-all, `quy` (Ayacucho) adapter | 965M | none / none (no qxp data) | 45.09 / 7.45 | 49.76 / 8.95 |
| 26 | cportoca/MMS_Quechua_finetuned (`qve` vocab) | 965M | none (Jan 2025) / none known | 45.54 / 7.44 | not run |
| 27 | ivangtorre/wav2vec2-xlsr-300m-quechua (AmericasNLP 2022, quy) | 315M | none / none known | 46.00 / 7.91 | 56.75 / 10.87 |
| 28 | Epiph0nE/whisper-quz (Whisper-medium) | 764M | unknown / unknown | 49.55 / 11.03 | 56.44 / 16.07 |
| 29 | Organizers: wav2vec2-base-qxp-finetuned | 94M | none / likely (older split: 161/310 clips) | 50.30 / 9.37 | 39.08 / 7.02 (leaked) |
| 30 | cportoca/CS224S_Quechua_Project (w2v-BERT 2.0) | 606M | none (May 2024) / none known | 54.98 / 9.68 | 62.15 / 11.17 |
| 31 | ppyecom/whisper-small-quechua-yachay | 242M | unknown / unknown | 60.35 / 11.22 | not run |
| 32 | LuiyiA/wav2vec2-large-xls-r-300m-quechua-colab | 315M | unknown / unknown | 88.37 / 21.11 | not run |
| 33 | cportoca/CS224S_Quechua_Project_Bilingual (w2v-BERT 2.0) | 606M | none (Jun 2024) / none known | 97.13 / 41.39 | not run |
| 34 | ctaguchi/ssc-qxp-mms-model-mix-adapt-max2 | 965M | none (Dec 2025) / possible (SPS qxp) | 99.55 / 56.03 | not run |
| 35 | ctaguchi/ssc-qxp-mms-model-mix-adapt-max3 | 965M | none (Dec 2025) / possible (SPS qxp) | 100.00 / 78.30 | not run |
| 36 | cportoca/CS224S_Quechua_Project_Expanded_Dataset (w2v-BERT 2.0) | 606M | none (Jun 2024) / none known | 100.00 / 72.52 | not run |
| 37 | gaalocastillo/wav2vec2-large-xls-r-300m-asp-project-quechua (outputs blank) | 315M | none (Jan 2023) / none known | 100.00 / 100.00 | not run |
| 38 | ctaguchi/ssc-qxp-mms-model-mix-adapt-max-longcv | 965M | none (Dec 2025) / possible (SPS qxp) | 101.51 / 73.68 | not run |
| 39 | LastEngineer01/whisper-quechua (Whisper-small) | 242M | none (Mar 2025) / unknown | 114.58 / 81.42 | not run |
| 40 | LastEngineer01/whisper-quechua-v2 (Whisper-tiny) | 38M | none (Apr 2025) / unknown | 136.78 / 51.26 | not run |
| 41 | pollitoconpapass/whisper-small-cuzco-quechua (quz Bible) | 242M | none (Jun 2024) / none (Bible audio) | 175.91 / 132.60 | not run |
| 42 | cportoca/Quechua_Project_Whisper (Whisper-tiny) | 38M | none (Nov 2024) / none known | 1637.99 / 1938.08 | not run |

Overlap column: "OOD / dev_spont" training-data overlap. **known** = verified clip overlap; **likely** = trained on an organizer split that we know contains the clips; **possible** = trained on SPS qxp data whose split we cannot check; **unknown** = no training-data information published; **selection / tuned** = our systems used `dev_spont` for checkpoint selection (LoRA) or rescoring-weight tuning (joint), but never for training. dev_spont values marked "(leaked)" are training-set or near-training-set numbers. "(?)" marks possible overlap.

## Notes per system

- **omniASR (Meta).** These are the official v2 cards run through the official pipeline.
  - The 300M/1B/3B LLM v2 cards carry the language-marker bug described in `reports/unsloth_vllm_spike.md`: they use a CJK text-frontend row (9812) instead of the LID row. We recorded them as is. Whatever the bug costs, **LLM-1B-v2 still beats LLM-7B-v2 on OOD** (20.85 vs 23.26 WER), though not on dev_spont (18.75 vs 14.25).
  - **LLM-3B-v2 was skipped as too slow.** With `--batch-size 16` it hit the 45 min cap: 5 min of download plus about 39 min of decoding without finishing. LLM-300M and LLM-1B took 34–35 min at batch 4 (RTF 2.2); the decoder dominates.
  - Batch 16 turned out *slower* than batch 4 for the LLM beam search on CPU. LLM-1B dev_spont39 hit a 4 h cap at batch 16 and finished at batch 4 in 4.1 h (RTF 2.8). The LLM-300M dev_spont39 run was therefore dropped.
  - CTC-3B is no better than CTC-1B on OOD (34.97 vs 35.42) but much better on dev_spont (23.46 vs 31.50).
  - Meta's qxp training data is about 9.9 h from Common Voice. Whether it includes Spontaneous Speech clips is not published, so dev_spont overlap is "unknown".
- **Organizer checkpoints.**
  - OOD hypotheses are new (CPU, `bench_organizer.py run --roles ood`). dev_spont hypotheses are the stored ones from `organizer_baselines.md`.
  - The organizers report 27.4 / 4.55 on OOD for their best fine-tuned model (Huaman et al. 2026). We measure 31.04 / 5.18 for `xls-r-cpt-qxp-cv26-validated` against the curated references. Their **silver** checkpoints generalize better to radio speech (24.47 and 26.06 WER) than the validated ones (≈ 31).
- **Adriatogi / ayayon** (`checkpoint_best.pt` only, no card).
  - Their pickled fairseq cfg shows the organizers' own recipe: CPT init `/workspace/checkpoints/cpt/checkpoint_best.pt`, data `/workspace/data/manifests/finetune/qxp_v2`, train subsets `train` and `train_silver` respectively.
  - In effect they re-train the organizers' older qxp_v2 models, so dev_spont is flagged (9.74 and 13.47 WER, far below every clean system). The OOD radio drama is not part of that recipe.
- **MMS-1b-all + `quz`/`quy` adapters.** These are clean (no qxp data). The Cusco adapter (34.21 / 5.49) is about as good as zero-shot omniASR-CTC-1B. Both adapters are worse on spontaneous speech (46.8 / 49.8).
- **MMS-1B community fine-tunes.**
  - `robertp408/mms-1b-all-qxp`: 1,502 training clips from Mozilla Data Collective dataset `cmflnuzw781hfx067jwv4pt9x`, which the MDC API now reports as "no longer available".
  - `ctaguchi/ssc-qxp-*`: Mozilla Spontaneous Speech shared task 2025, data not stated. Both may include SPS `dev` clips, so dev_spont is "possible".
  - `ctaguchi` max2, max3 and longcv score about 100 % WER. For max2 we checked that `processor.decode` equals a manual argmax/collapse through `vocab.json`, so the decoding is faithful. The likely cause is the published `vocab.json` (a set-ordered dict) not matching the trained head, or failed training: the cards themselves report WER 1.0 (max3) and 0.80 (longcv).
- **Whisper community models.** We decoded with each checkpoint's own generation config, as the cards prescribe nothing else.
  - `whisper-small-cuzco-quechua` has `language="latin"` and was trained on Bible audio. `LastEngineer01/*` has no language set, so Whisper auto-detects. These models and `cportoca/Quechua_Project_Whisper` fall into repetition loops on radio speech; WER above 100 % comes from insertions.
  - `Epiph0nE/whisper-quz` (Whisper-medium, prompt `spanish`) is the only usable one (49.55 / 11.03).
- **`gaalocastillo/wav2vec2-large-xls-r-300m-asp-project-quechua`** emits only blanks (collapsed checkpoint).
- **dev_spont<=39 "not run"**: skipped to save CPU time (each run is 5.5× the OOD audio). We ran it for the main contenders of each family: omniASR, MMS, organizer, Adriatogi/ayayon, and the leading MMS and community fine-tunes. We skipped it for the unusable checkpoints and for LLM-300M/3B.

### Not evaluated

| Repo | Reason |
|---|---|
| `ctaguchi/ssc-qxp-mms-model-bl` | tokenizer files only, no weights |
| `cportoca/MMS_Quechua_finetuned_qve` | weights only: no `vocab.json` / tokenizer, so CTC ids cannot be decoded |
| `Epiph0nE/mms-qxp-finetuned`, `Epiph0nE/mms-qxp-v2`, `Epiph0nE/rimay-vits-qxp-v3` | VITS text-to-speech, not ASR |
| `espnet/americasnlp22-asr-quy` | ESPnet only (does not load with transformers) |
| `bellengc/wav2vec2-large-xls-r-300m-asp-project-quechua{,-2}` | empty repos |
| `omniASR_LLM_3B_v2` | > 40 min on the OOD set on CPU (see above) |

The candidate list comes from `https://huggingface.co/api/models?search=` for `qxp`, `quechua`, `quz`, `quy` and `puno` (2026-10-09), restricted to ASR models for Puno or Southern Quechua (qxp, quz, quy, qve) or unspecified Quechua.

## Verdict

1. On the only evaluation set that is clean for every system (OOD radio drama), all five of our systems (14.3–15.1 WER, 2.2–2.7 CER) beat all 37 public systems evaluated. The best public system, zero-shot omniASR-LLM-1B-v2 (20.85 / 2.94), has 6.6 points (46 % relative) higher WER, and every public fine-tuned Quechua checkpoint, the organizers' included, is at ≥ 24.5 WER.
2. Fine-tuned public models trail zero-shot omniASR-LLM on radio speech, and on spontaneous speech they look strong only where they were trained on the test clips (organizer and Adriatogi/ayayon rows, 4–13 % WER on dev_spont<=39).
3. On spontaneous speech without known overlap, our margin over zero-shot omniASR-LLM-7B is small (13.63 vs 14.25 WER) and favours us through tuning. The robust, decision-relevant gap is the OOD one.

## Commands

Server B (`ssh -p <SERVER_B_PORT> root@<SERVER_B_HOST>`) and server A (`ssh -p <SERVER_A_PORT> root@<SERVER_A_HOST>`), all CPU-only:

```bash
# envs (uv only). Server B: new env. Server A: the existing orgbench env was used read-only.
mkdir -p /workspace/simbig/envs/bench_hf && cd /workspace/simbig/envs/bench_hf
uv init --bare --python 3.12
uv add "torch==2.8.0" "transformers>=4.45" soundfile polars jiwer omegaconf huggingface_hub numpy typer safetensors
#   resolved: torch 2.8.0, transformers 5.19.0. qxp_asr is imported read-only via PYTHONPATH.
# scripts/bench_all_systems.py + scripts/bench_organizer.py copied to /workspace/simbig/envs/bench_hf/run/

cd /workspace/simbig/envs/bench_hf/run
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 HF_HOME=/workspace/hf PYTHONPATH=/workspace/simbig/repo/src
OMNI=/workspace/simbig/envs/omni/.venv/bin/python     # existing omni env, used read-only (no sync)
HFPY=/workspace/simbig/envs/bench_hf/.venv/bin/python  # server A: /workspace/simbig/envs/orgbench/.venv/bin/python

# 1. omniASR zero-shot (official pipeline, qxp_Latn, fp32 CPU)
for c in omniASR_CTC_300M_v2 omniASR_CTC_1B_v2 omniASR_CTC_3B_v2 omniASR_LLM_300M_v2 omniASR_LLM_1B_v2; do
  timeout 2700 nice -n 15 $OMNI bench_all_systems.py omni $c --sets ood --out-dir runs
done
timeout 2700 nice -n 15 $OMNI bench_all_systems.py omni omniASR_LLM_3B_v2 --batch-size 16 --sets ood --out-dir runs  # hit the cap
nice -n 15 $OMNI bench_all_systems.py omni omniASR_CTC_300M_v2 --sets dev_spont39 --out-dir runs  # same for CTC_1B/3B, LLM_1B

# 2. organizer checkpoints (server A; dev_spont preds reused from organizer_baselines.md)
for m in xls-r-cpt-qxp-cv26-validated xls-r-cpt-qxp-cv26-silver xls-r-qxp-cv26-validated xls-r-qxp-cv26-silver \
         wav2vec2-base-qxp-finetuned whisper-base-qxp-finetuned; do
  nice -n 15 /workspace/simbig/envs/orgbench/.venv/bin/python /workspace/simbig/repo/scripts/bench_organizer.py run \
      /workspace/simbig/envs/orgbench/checkpoints/$m --name $m --roles ood --out-dir org --device cpu --threads 4 --batch-seconds 60
done
$HFPY bench_all_systems.py import org/*.preds.parquet org_dev/*.preds.parquet --out-dir runs  # org_dev = reports/benchmarks/organizer_preds

# 3. MMS-1b-all with Quechua adapters
nice -n 15 $HFPY bench_all_systems.py hf facebook/mms-1b-all --name mms-1b-all-quz --target-lang quz --sets ood,dev_spont39 --out-dir runs
nice -n 15 $HFPY bench_all_systems.py hf facebook/mms-1b-all --name mms-1b-all-quy --target-lang quy --sets ood,dev_spont39 --out-dir runs

# 4. community checkpoints (one call per repo; fairseq / CTC / Whisper auto-detected)
for m in robertp408/mms-1b-all-qxp ctaguchi/ssc-qxp-mms-model-mix-adapt-max{,-lowlr,2,2-2,-longcv,-longcv2,3} \
         cportoca/MMS_Quechua_finetuned Adriatogi/wav2vec2-puno-quechua-ft-cpt-validated ayayon/asr-puno-quechua-curated \
         ivangtorre/wav2vec2-xlsr-300m-quechua LuiyiA/wav2vec2-large-xls-r-300m-quechua-colab \
         gaalocastillo/wav2vec2-large-xls-r-300m-asp-project-quechua cportoca/CS224S_Quechua_Project{,_Bilingual,_Expanded_Dataset} \
         Epiph0nE/whisper-quz pollitoconpapass/whisper-small-cuzco-quechua ppyecom/whisper-small-quechua-yachay \
         LastEngineer01/whisper-quechua{,-v2} cportoca/Quechua_Project_Whisper; do
  timeout 2700 nice -n 15 $HFPY bench_all_systems.py hf $m --sets ood --out-dir runs
done   # dev_spont39 (--sets dev_spont39) only for the models in the table that have a value

# 5. our systems (stored predictions; no re-run)
P=/workspace/simbig/artifacts/preds
$HFPY bench_all_systems.py import $P/joint_final.parquet $P/joint_v3.parquet $P/llm7b_lora_v1.parquet \
    $P/llm7b_lora_v2.parquet --prefix ours_ --out-dir runs                        # server A
$HFPY bench_all_systems.py import $P/llm7b_textcpt.parquet --prefix ours_ --out-dir runs   # server B
cp $P/zs_llm7b_all.parquet tmp/omniASR_LLM_7B_v2.parquet
$HFPY bench_all_systems.py import tmp/omniASR_LLM_7B_v2.parquet --hyp-col raw --out-dir runs

# 6. merge both servers' runs/ into one directory, then score + table
$HFPY bench_all_systems.py score --out-dir all --csv scores.csv
$HFPY bench_all_systems.py table --csv scores.csv
```

Per-run hypotheses (`<system>__<set>.parquet` + metadata `.json`) and `scores.csv` are in
`reports/benchmarks/all_systems_runs/`.
