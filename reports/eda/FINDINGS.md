# EDA — key findings and their consequences

Source: `scripts/eda.py` → `reports/eda/EDA.md` (tables) and figures in this folder. Only aggregate
statistics of the shared-task test set are reported (data-use agreement).

1. **Test spontaneous = elderly speakers only.** All 221 spontaneous test clips are `sixties` (60+),
   but labelled training data has ~0.05 h of gold elderly spontaneous speech (1.7 h silver,
   0.3 h unlabelled). → Prioritise pseudo-labelling of elderly/older audio, and select the
   spontaneous dev set to over-represent older speakers.
2. **Test clips are silence-trimmed.** Leading silence median 0.01 s (scripted) / 0.24 s
   (spontaneous) vs 0.5–0.8 s in training; voiced fraction 0.72 vs 0.56. → Random leading/trailing
   silence trimming as training augmentation; never rely on silence context.
   (The lower "SNR" proxy of the test set is mostly a by-product of trimming: the 10th-percentile
   frame is speech rather than background.)
3. **Gender shift.** Scripted training audio is ~72 % female / 27 % unknown / 1.4 % male; test scripted
   is one-third male. Spontaneous gold has 2.3 h male speech, which is the main male source.
   → Speed/pitch perturbation; include spontaneous data when training a scripted model.
4. **Shorter test clips.** Test spontaneous mean 9.1 s (max 26 s) vs 15–17 s in training.
5. **Split semantics.** Official CV splits are *sentence-disjoint* (≤0.7 % dev/test text seen in
   train); the organizer `splits_cv26` scripted dev/test are *speaker-disjoint but 100 % sentence
   overlapping*. Whether the hidden test reuses the 2,051-sentence pool decides how hard to lean on a
   closed-set sentence LM — to be measured empirically on first-pass hypotheses.
6. **Open vocabulary for spontaneous.** 22.5 % of spontaneous-gold tokens are unseen in all other
   training text (71 % of word types are hapax). → Character / subword LM, not a closed word LM.
7. **Label noise.** 29 invalidated scripted clips have implausible speaking rates (<3 or >25
   chars/voiced-s); validated scripted only 2. Unlabelled spontaneous includes a 1,716 s clip →
   duration filter (≤ 60 s) for training / pseudo-labelling.
8. **Alphabet.** Targets fit `[a-zñ' ]`; Spanish letters `e o d j g v b f z x` are rare (loanwords,
   names) but must stay in the vocabulary.
