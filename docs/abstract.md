# Puno Quechua ASR: When to Adapt the Speech Encoder, and Verification as the Bottleneck

Irving Ernesto Quezada Ramírez, Independent Researcher (research@irvingernesto.com)

We describe our system for Task 1 of the SIMBig 2026 shared task, Puno Quechua (qxp) ASR from the provided data and public models only, with four findings.

1. The official splits reuse every prompt sentence between training and test, which can reverse the ranking of systems; we evaluate on held-out sentences, speakers and an out-of-domain (OOD) radio drama.
2. Hybrid rescoring of LLM-ASR, CTC and perturbed-audio candidates reaches 14.1% WER on the radio drama vs. 20.9% for the best of 37 public checkpoints.
3. LoRA on the speech encoder instead of the LLM decoder cuts WER on new sentences by 3.7 points for omniASR-LLM-7B and by 6.4 and 4.6 for two Whisper seeds (the second n.s.), not for smaller omniASR encoders, which encode Quechua stop contrasts almost as well. In the 7B, the gain needs the lower layers, survives truncation to 4 of 16 singular directions per projection, and leaves the N-best oracle unchanged while ranking candidates better.
4. Verification is the bottleneck: sampling and perturbed audio propose better transcripts than beam search, but our reranker orders candidate pairs correctly only 71–78% of the time.
