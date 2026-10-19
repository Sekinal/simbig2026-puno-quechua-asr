# Video script: SIMBig 2026 Shared Task 1 (15 minutes)

**Format.** The video is 14:40 long: 14 scenes, at about 145 words per minute. Each scene's animation in `scenes.py` runs exactly the length shown in its header, and `script.srt` gives the narration lines with their times. **[Bracketed lines]** describe what is on screen; they are not spoken.

All numbers come from the paper and from `reports/fig_data/`.

---

## S01 — Title (0:00–0:25, 25 s)
**[Title card: "Where should a speech model learn a new language?" Subtitle: "Puno Quechua ASR with provided data only — SIMBig 2026 Shared Task 1". Name, affiliation.]**

Hi, I'm Irving Ernesto. This is our system for the SIMBig 2026 shared task on Puno Quechua speech recognition. It was built using only the data the organizers provided. Along the way it answers a question that matters beyond Quechua: when you adapt a large speech model to a new language, where should the learning happen?

## S02 — The problem (0:25–1:25, 60 s)
**[Map dot over Puno, Peru; "≈474,000 speakers". Then three data blocks fill in: 30.5 h read speech, 77 speakers; 5.2 h transcribed spontaneous speech; 37 h untranscribed.]**

The Puno region of southern Peru has about 474,000 Quechua speakers, yet very little speech technology. The shared task gives us Common Voice data.
- Read speech: about thirty hours from seventy-seven speakers.
- Spontaneous speech: five hours with transcripts, and thirty-seven hours without.

Quechua is a hard case for speech recognition. Words are long, built from many suffixes, so one wrong suffix is a whole wrong word. And its stop consonants carry distinctions Spanish and English lack: plain, aspirated and ejective stops, and a uvular q next to the velar k.

**[Highlight: "only 2,051 distinct sentences".]**

One detail turns out to matter a lot. Those thirty hours of read speech cover only about two thousand distinct sentences, so every sentence is read many times by many people.

**[Test set: 659 clips; "spontaneous test speakers: all 60+"; "training speakers: almost all under 40".]**

The test set is different in two ways. Its sentences are new. And all of its spontaneous speakers are over sixty, while almost all our transcribed spontaneous speech comes from speakers under forty.

## S03 — The sentence-reuse trap (1:25–2:50, 85 s)
**[Slope chart animates: x = "seen sentences" → "new sentences", y = WER. The 4-gram system (violet) starts lowest at 6.0 and climbs to 18.7. Greedy CTC (grey) stays flat, 14.1 → 14.6. Our system (ink) goes 10.0 → 9.7. The lines cross.]**

The official splits keep speakers apart, but they share every sentence between training and test. Here is why that matters. We measure word error rate: the share of words a system gets wrong, so lower is better.

Take one speech model and add a word language model trained on the training prompts. On sentences it has seen, it is the best system we have: six percent word error rate. On new sentences it jumps to almost nineteen percent, worse than the same model with no language model at all.

**[Labels: "memorizes the prompts" on violet; "generalizes" on ink. Callout: "reversal: −12.1 points [−18.0, −7.1]".]**

The language model hasn't learned Quechua. It has memorized the prompts. Its effect reverses between seen and new sentences by twelve points, and the bootstrap interval is far from zero.

**[Test hypotheses: distance to the nearest training prompt 0.495, held-out sentences 0.489, in-pool 0.028.]**

The real test reads new sentences: its transcripts are as far from the training prompts as our held-out sentences are. So the official splits would have picked the wrong system.

## S04 — How we evaluate (2:50–3:35, 45 s)
**[Three evaluation cards slide in: "held-out sentences — 400 sentences, never trained on"; "held-out speakers — internal test, 21 speakers"; "out of domain — radio drama, 109 clips". Under them: "speaker × sentence bootstrap".]**

So we built our own evaluation.
- We hold out four hundred sentences from all training audio and text.
- We test on speakers never seen in training.
- We add an out-of-domain radio drama that we never tune on.

For spontaneous speech we use the official development and test speakers, four and six of them. That is too few for strong conclusions, so we report spontaneous results as descriptive, with an exact sign test over speakers.

**[Small icon row: "95% intervals", "Holm correction".]**

Every comparison in this talk comes with a ninety-five percent interval from a paired bootstrap. The bootstrap resamples speakers and sentences together, because clips from the same speaker or sentence are not independent.

## S05 — The system (3:35–5:10, 95 s)
**[Pipeline diagram builds left to right: audio → two branches.]**

Our system has two branches.

**[Top branch: "CTC 1B, fine-tuned" → "16-best + word 4-gram".]**

The first is a one-billion-parameter CTC model, fine-tuned on our data, which proposes sixteen candidate transcripts per clip.

**[Bottom branch: "omniASR-LLM 7B: speech encoder → LLaMA decoder", LoRA adapters glow.]**

The second is Meta's Omnilingual LLM-ASR model with seven billion parameters: a speech encoder feeding a LLaMA decoder. We adapt it with LoRA, small low-rank updates to its weights, trained on the transcribed data and on pseudo-labels for the untranscribed pool.

Those pseudo-labels come from self-training. Three systems transcribe the untranscribed spontaneous pool, we combine them by voting, and we keep the twenty-seven hours where they mostly agree. That gives us much more spontaneous speech than the five hours with real transcripts.

**[Perturbed copies of the waveform fan out: faster, slower, noisier.]**

We also decode six perturbed copies of every clip, a little faster, slower, or noisier. Each copy can surface a different, sometimes better, transcript.

**[All candidates pour into a "reranker" box: CTC likelihood + character LM + LLM likelihoods.]**

Finally, a reranker scores every candidate on the original audio, combining CTC and LLM likelihoods with a character language model, and picks one. Its weights are tuned on the development set only. The internal test set and the radio drama are never used for tuning, so they stay honest.

## S06 — Results (5:10–6:20, 70 s)
**[Bar race on the radio drama (OOD): 37 public systems as thin grey bars; best public 20.9; organizer best 24.5; ours 14.1 in ink.]**

Out of domain, on the radio drama, our submission reaches fourteen point one percent word error rate. We ran thirty-seven public Quechua-capable models on the same audio; the best of them, Omnilingual's 1B LLM model zero-shot, gets twenty point nine. That's a third fewer errors.

After the experiments, we designated two confirmatory hypotheses and corrected them for multiple testing. The main one holds: on held-out sentences, the full system beats its best single component even after the correction, while out of domain it does not.

**[Held-out sentences: zero-shot 27.9 → LoRA 20.2 → + encoder adaptation 16.0 → full system 15.5.]**

On held-out sentences the gains add up. Zero-shot, the 7B model makes twenty-eight percent errors. LoRA on the decoder brings that to twenty. Adapting the encoder as well brings it to sixteen, and the full reranked system to fifteen and a half.

**[The 20.2 → 16.0 step is highlighted.]**

That middle step is the most interesting part of this work. Adapting the speech encoder, not just the language-model decoder, removed another fifth of the errors. So the rest of this talk asks: where should a speech model learn a new language?

## S07 — The question (6:20–6:55, 35 s)
**[Diagram: encoder (orange) and decoder (blue). Two LoRA switches: "adapt the decoder" vs "adapt the encoder".]**

An LLM-based recognizer has two places to learn. The decoder is a language model that already writes text; the encoder turns sound into representations. The common recipe adapts only the decoder. Recent work showed that adapting the encoder can help on hard, domain-shifted speech. Adaptation here means LoRA: next to each frozen weight matrix we train a small low-rank correction, so we can choose exactly which part of the model learns. We asked when that holds, and why. We compare three options with exactly the same data and training budget: adapt only the decoder, only the encoder, or both.

## S08 — It depends on the model (6:55–8:25, 90 s)
**[Dumbbell chart: four rows (omniASR encoders 300M, 1B, 7B; Whisper large-v3). Each row: blue dot = decoder LoRA, orange dot = encoder LoRA, arrow from blue to orange. Row 7B animates first: 20.2 → 16.4.]**

Same data, same recipe, LoRA on the decoder only or on the encoder only. For the 7B encoder, adapting the encoder wins clearly: sixteen point four against twenty point two.

**[Whisper row: 34.1 → 28.7, two seeds as small dots.]**

Whisper, a different model family that has never seen Quechua, shows the same pattern, and even more strongly out of domain.

**[1B row: 20.9 → 21.6 (no gain). 300M row: 19.8 → 25.4 (worse); arrows point the other way.]**

But the smaller Omnilingual encoders, one billion and three hundred million parameters, with the very same decoder, don't benefit. For the 300M model, adapting the encoder is clearly worse. With the decoder adapted, all three Omnilingual models end up near twenty percent. Only the 7B encoder turns encoder adaptation into a further gain.

**[Hollow rings appear: param-matched encoder LoRA, 22.9 and 21.4.]**

Is that just adapter size? A rank-16 adapter has ten times fewer parameters on the small encoder. So we matched the parameter count. It helps the 300M model a little, but encoder adaptation still doesn't beat the decoder. So it's not adapter capacity.

## S09 — Same information, different plasticity (8:25–9:35, 70 s)
**[Left panel: three bars per contrast (ejective, aspirated, q vs k) for 300M, 1B and 7B, all at 94–97%; dashed line at 50% labelled "shuffled-label control".]**

Maybe the small encoders just don't hear Quechua well enough? We trained linear probes on Quechua stop consonants: ejective versus plain, aspirated versus plain, and the uvular q versus the velar k. All three pretrained encoders separate these at ninety-four to ninety-seven percent, on speakers the probes never saw. The information is there at every size. A probe is a simple linear classifier on the encoder's internal representations: if a straight line can separate ejective from plain stops, the encoder encodes that contrast.

**[Right panel: CKA vs relative depth. 300M and 1B lines hug 1.0 until near the top. 7B line drops to ~0.84 early.]**

What differs is how much the encoder moves when we adapt it. Comparing representations before and after adaptation, the small encoders barely change except in their last few layers. The 7B encoder reorganizes from its lower-middle layers on. We measure this with CKA, a similarity score between two sets of representations of the same audio: one means unchanged, lower means the encoder has moved.

**[Caption: "same information, different plasticity". The 7B CKA used the enc+dec adapter.]**

Same information, different plasticity. That's our working explanation: a hypothesis that fits all our controls, not a proof.

## S10 — Which layers matter (9:35–10:50, 75 s)
**[Schematic: a tall bar of 128 layers split into four quarters. Rows appear one at a time; shaded quarters show where the update is kept; a WER dot slides along an axis with reference lines at 27.9 (zero-shot) and 16.4 (full adapter).]**

Inside the 7B encoder, which layers matter? We took the trained adapter and switched its update off, one block of layers at a time.

**[Row: top quarter removed → 16.2.]**

Remove the top quarter, where the update is largest, and nothing changes.

**[Row: bottom quarter removed → 20.1.]**

Remove the bottom quarter, and most of the gain disappears: twenty point one instead of sixteen point four.

**[Two small markers appear on each row: the two enc+dec seeds.]**

Both enc+dec adapters, trained with different seeds, show exactly the same pattern.

**[Rows: trained on bottom half only → 16.1; trained on top half only → 20.6.]**

And training on the bottom half alone matches training on all layers, while the top half alone gains nothing. The useful adaptation sits low in the encoder, not where the update is biggest.

Out of domain, the picture splits. Layers thirty-two to sixty-three alone recover the whole gain on the radio drama, while held-out read speech also needs the bottom quarter. In-domain and out-of-domain effects live at different depths.

## S11 — Four directions are enough (10:50–11:35, 45 s)
**[Line chart: WER vs directions kept, 1, 2, 4, 8, 16. Points at 17.8, 17.0, 16.35, 16.5, 16.4. Energy labels 61%, 74%, 85%, 93%, 100%. Reference line at decoder LoRA 20.2.]**

How big is that change? Each adapted weight matrix can move in sixteen directions. Keeping only the top four, eighty-five percent of the update's energy, keeps the whole gain: sixteen point three five versus sixteen point four. Even one direction keeps most of it. So a tiny, targeted change in the lower encoder does almost all of the work. The useful change is very low-dimensional. And two training runs with different seeds find almost entirely different directions. Averaging their updates exactly lands between the two, no better than the better seed. There are many equally good small changes, not one special subspace.

## S12 — Ranking, not coverage (11:35–12:25, 50 s)
**[Two-panel reveal. Left: "Is the right answer in the beam?" Oracle WER: decoder LoRA 14.0 vs encoder LoRA 14.0 (equal bars). Right: "Does the model rank it first?" Pairwise accuracy: 69.7% → 76.9%.]**

So what does encoder adaptation buy? Not new answers: the best transcript in the beam is just as good with or without it, fourteen percent either way. What changes is ranking. On pairs of candidates, the encoder-adapted model prefers the better one about seven points more often. That gain is clear for candidates that differ in whole words. For single-letter contrasts, like ejective versus plain, it points the same way but is not significant.

**[Text: "it re-weights what the encoder already knows".]**

It doesn't learn new sounds. It learns to trust the right evidence. And as a fourth scorer in our reranker, the encoder-adapted model lowered the error rate on held-out sentences by another one point four points, which is why our final submission uses it.

## S13 — Exploration is easy, verification is hard (12:25–13:50, 85 s)
**[Bar chart: "a better transcript exists among 8 candidates": beam 6.0%, samples 14.0%, perturbed audio 16.7%. Then: best candidate in our pool 8.6% WER vs our pick 16.9% (held-out sentences).]**

That pattern runs through the whole project. Finding good candidates is easy. Eight beam hypotheses contain a better transcript than the top one for only six percent of clips, but decoding perturbed audio finds one for seventeen percent. And the best candidate already in our pool would cut our errors in half: eight point six instead of sixteen point nine percent.

**[RL icons crossed out: GSPO, MWER, Dr GRPO.]**

Reinforcement learning didn't help. GSPO, MWER and Dr GRPO never beat plain supervised training on held-out sentences. With reference transcripts available, reinforcement learning and ordinary cross-entropy training aim at the same answer, and perturbed transcripts found by exploration have little likelihood on clean audio, so there is little left for the reward to teach.

**[Verifier curve: WER vs pairwise ranking accuracy, synthetic verifiers in aqua, our reranker as a dark point at 71–78%.]**

The limit is choosing among candidates without a reference. Our reranker orders candidate pairs correctly seventy-one to seventy-eight percent of the time. Simulated verifiers show that five more points of ranking accuracy would cut word error rate by between thirteen and fifty percent. The encoder-adapted scorer gave us part of that, exactly as the curve predicts.

## S14 — Takeaways (13:50–14:40, 50 s)
**[Three takeaway cards appear one by one.]**

Three takeaways.
- **Evaluate on new sentences.** Sentence reuse can reverse which method looks best.
- **Where to adapt depends on the model.** The encoder for the 7B model and for Whisper, the decoder for the smaller encoders. Where it helps, a low-dimensional update in the lower layers improves ranking, not coverage.
- **Verification is the bottleneck.** Better candidates are easy to find; choosing them is hard.

For practitioners: evaluate on new sentences and new speakers; try adapting the encoder, but test it, because it depends on the model; and if you have compute to spare, spend it on better verifiers, not more candidates.

**[Limitations line: "single seeds for the smaller models · 4–6 spontaneous evaluation speakers · official test scores pending". Then: code on GitHub · research@irvingernesto.com.]**

The limitations are in the paper. The code and all run scripts will be on GitHub. Thank you.

## S15 — References (silent, 14 s)
**[References card: full list of the sources cited on screen. No narration.]**
