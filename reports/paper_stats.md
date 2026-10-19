# Paper statistics (scripts/paper_stats.py)

Paired bootstrap, 10000 resamples; differences in WER points (a - b); `lo`/`hi`/`p` use the set's resampling design, `clip_*` resample clips.

## 1 sentence reuse

| comparison | set | clips | wer_a | wer_b | diff | clip_lo | clip_hi | design | units | lo | hi | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctc_4gram - ctc1b_v3_soup_wise0.85_greedy | dev_seen | 670 | 5.99 | 14.11 | -8.12 | -9.37 | -6.92 | speaker x sentence | 19 | -13.22 | -4.36 | 0.00 |
| ctc_4gram - ctc1b_v3_soup_wise0.85_greedy | dev_unseen | 278 | 18.66 | 14.64 | 4.02 | 2.19 | 5.84 | speaker x sentence | 17 | 0.92 | 7.16 | 0.01 |
| ctc_4gram - ctc1b_v3_soup_wise0.85_greedy | itest_seen | 612 | 7.54 | 16.70 | -9.16 | -10.46 | -7.86 | speaker x sentence | 21 | -13.13 | -4.90 | 0.00 |
| ctc_4gram - ctc1b_v3_soup_wise0.85_greedy | itest_unseen | 297 | 25.04 | 21.30 | 3.74 | 1.72 | 5.82 | speaker x sentence | 19 | 0.00 | 8.03 | 0.05 |
| hybrid_dev_aug - ctc_4gram | dev_seen | 670 | 10.00 | 5.99 | 4.01 | 3.06 | 4.99 | speaker x sentence | 19 | 1.88 | 6.85 | 0.00 |
| hybrid_dev_aug - ctc_4gram | dev_unseen | 278 | 9.73 | 18.66 | -8.93 | -11.16 | -6.75 | speaker x sentence | 17 | -12.82 | -5.44 | 0.00 |
| hybrid_dev_aug - ctc_4gram | itest_seen | 612 | 11.91 | 7.54 | 4.37 | 3.20 | 5.55 | speaker x sentence | 21 | 2.37 | 6.81 | 0.00 |
| hybrid_dev_aug - ctc_4gram | itest_unseen | 297 | 16.87 | 25.04 | -8.17 | -10.31 | -6.07 | speaker x sentence | 19 | -13.02 | -4.18 | 0.0002 |
| interaction (dev_seen - dev_unseen) | dev_seen/dev_unseen |  |  |  | -12.14 |  |  | speaker x sentence |  | -17.96 | -7.14 | 0.00 |
| interaction (itest_seen - itest_unseen) | itest_seen/itest_unseen |  |  |  | -12.90 |  |  | speaker x sentence |  | -18.73 | -7.33 | 0.00 |

## 2 components

| comparison | set | clips | wer_a | wer_b | diff | clip_lo | clip_hi | design | units | lo | hi | p | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LoRA v1 - zero-shot | dev_unseen | 278 | 12.59 | 21.25 | -8.66 | -10.69 | -6.71 | speaker x sentence | 17 | -12.03 | -5.63 | 0.00 |  |
| LoRA v1 - zero-shot | itest_unseen | 297 | 20.17 | 27.91 | -7.74 | -9.64 | -5.93 | speaker x sentence | 19 | -11.76 | -4.61 | 0.0002 |  |
| LoRA v1 - zero-shot | dev_spont | 310 | 14.79 | 14.31 | 0.48 | -0.10 | 1.08 | speaker sign test (2/4 better) | 4 | -0.10 | 1.08 | 1.00 |  |
| LoRA v1 - zero-shot | itest_spont | 304 | 6.81 | 7.25 | -0.44 | -0.94 | 0.05 | speaker sign test (4/6 better) | 6 | -0.94 | 0.05 | 0.38 |  |
| LoRA v1 - zero-shot | ood | 109 | 14.88 | 23.26 | -8.38 | -10.74 | -6.03 | clip | 109 | -10.74 | -6.03 | 0.00 |  |
| LoRA v2 - LoRA v1 | dev_unseen | 278 | 13.12 | 12.59 | 0.54 | -0.44 | 1.49 | speaker x sentence | 17 | -0.86 | 2.09 | 0.45 |  |
| LoRA v2 - LoRA v1 | itest_unseen | 297 | 20.17 | 20.17 | 0.00 | -1.27 | 1.32 | speaker x sentence | 19 | -1.96 | 2.15 | 1.00 |  |
| LoRA v2 - LoRA v1 | dev_spont | 310 | 14.93 | 14.79 | 0.14 | -0.23 | 0.52 | speaker sign test (0/4 better) | 4 | -0.23 | 0.52 | 0.25 |  |
| LoRA v2 - LoRA v1 | itest_spont | 304 | 6.62 | 6.81 | -0.19 | -0.52 | 0.12 | speaker sign test (3/6 better) | 6 | -0.52 | 0.12 | 1.00 |  |
| LoRA v2 - LoRA v1 | ood | 109 | 15.11 | 14.88 | 0.23 | -1.00 | 1.45 | clip | 109 | -1.00 | 1.45 | 0.76 |  |
| hybrid - LoRA v1 | dev_unseen | 278 | 9.82 | 12.59 | -2.77 | -4.07 | -1.59 | speaker x sentence | 17 | -5.39 | -0.75 | 0.007 |  |
| hybrid - LoRA v1 | itest_unseen | 297 | 17.39 | 20.17 | -2.78 | -4.46 | -1.14 | speaker x sentence | 19 | -5.69 | -0.38 | 0.03 |  |
| hybrid - LoRA v1 | dev_spont | 310 | 13.67 | 14.79 | -1.12 | -1.86 | -0.42 | speaker sign test (2/4 better) | 4 | -1.86 | -0.42 | 1.00 |  |
| hybrid - LoRA v1 | itest_spont | 304 | 6.20 | 6.81 | -0.61 | -1.12 | -0.14 | speaker sign test (3/6 better) | 6 | -1.12 | -0.14 | 1.00 |  |
| hybrid - LoRA v1 | ood | 109 | 14.50 | 14.88 | -0.38 | -1.31 | 0.51 | clip | 109 | -1.31 | 0.51 | 0.45 |  |
| hybrid - LoRA v2 | dev_unseen | 278 | 9.82 | 13.12 | -3.30 | -4.74 | -2.02 | speaker x sentence | 17 | -6.36 | -1.10 | 0.0032 |  |
| hybrid - LoRA v2 | itest_unseen | 297 | 17.39 | 20.17 | -2.78 | -4.25 | -1.40 | speaker x sentence | 19 | -5.30 | -0.69 | 0.02 |  |
| hybrid - LoRA v2 | dev_spont | 310 | 13.67 | 14.93 | -1.25 | -2.03 | -0.53 | speaker sign test (3/4 better) | 4 | -2.03 | -0.53 | 0.62 |  |
| hybrid - LoRA v2 | itest_spont | 304 | 6.20 | 6.62 | -0.42 | -0.92 | 0.05 | speaker sign test (3/6 better) | 6 | -0.92 | 0.05 | 1.00 |  |
| hybrid - LoRA v2 | ood | 109 | 14.50 | 15.11 | -0.60 | -1.79 | 0.58 | clip | 109 | -1.79 | 0.58 | 0.35 |  |
| views - hybrid | dev_unseen | 278 | 9.73 | 9.82 | -0.09 | -0.27 | 0.00 | speaker x sentence | 17 | -0.60 | 0.00 | 1.00 |  |
| views - hybrid | itest_unseen | 297 | 16.87 | 17.39 | -0.52 | -1.14 | -0.09 | speaker x sentence | 19 | -1.77 | 0.00 | 0.36 | 0.36 |
| views - hybrid | dev_spont | 310 | 13.52 | 13.67 | -0.15 | -0.31 | -0.03 | speaker sign test (2/4 better) | 4 | -0.31 | -0.03 | 0.50 |  |
| views - hybrid | itest_spont | 304 | 6.21 | 6.20 | 0.02 | -0.07 | 0.15 | speaker sign test (2/6 better) | 6 | -0.07 | 0.15 | 1.00 |  |
| views - hybrid | ood | 109 | 13.67 | 14.50 | -0.83 | -1.80 | 0.08 | clip | 109 | -1.80 | 0.08 | 0.08 | 0.16 |
| full - zero-shot | dev_unseen | 278 | 9.73 | 21.25 | -11.52 | -13.70 | -9.39 | speaker x sentence | 17 | -15.59 | -7.95 | 0.00 |  |
| full - zero-shot | itest_unseen | 297 | 16.87 | 27.91 | -11.04 | -13.30 | -8.89 | speaker x sentence | 19 | -16.08 | -7.10 | 0.00 |  |
| full - zero-shot | dev_spont | 310 | 13.52 | 14.31 | -0.79 | -1.62 | 0.00 | speaker sign test (4/4 better) | 4 | -1.62 | 0.00 | 0.12 |  |
| full - zero-shot | itest_spont | 304 | 6.21 | 7.25 | -1.04 | -1.64 | -0.49 | speaker sign test (3/6 better) | 6 | -1.64 | -0.49 | 1.00 |  |
| full - zero-shot | ood | 109 | 13.67 | 23.26 | -9.59 | -12.15 | -6.99 | clip | 109 | -12.15 | -6.99 | 0.00 |  |
| full - CTC | dev_unseen | 278 | 9.73 | 14.64 | -4.91 | -6.64 | -3.18 | speaker x sentence | 17 | -8.44 | -1.78 | 0.0032 |  |
| full - CTC | itest_unseen | 297 | 16.87 | 21.30 | -4.43 | -6.40 | -2.45 | speaker x sentence | 19 | -8.22 | -0.97 | 0.02 |  |
| full - CTC | dev_spont | 310 | 13.52 | 19.08 | -5.57 | -6.57 | -4.57 | speaker sign test (3/4 better) | 4 | -6.57 | -4.57 | 0.62 |  |
| full - CTC | itest_spont | 304 | 6.21 | 8.16 | -1.95 | -2.53 | -1.38 | speaker sign test (5/6 better) | 6 | -2.53 | -1.38 | 0.22 |  |
| full - CTC | ood | 109 | 13.67 | 19.34 | -5.66 | -8.33 | -3.20 | clip | 109 | -8.33 | -3.20 | 0.00 |  |
| full - LoRA v1 | dev_unseen | 278 | 9.73 | 12.59 | -2.86 | -4.18 | -1.67 | speaker x sentence | 17 | -5.52 | -0.81 | 0.0048 |  |
| full - LoRA v1 | itest_unseen | 297 | 16.87 | 20.17 | -3.30 | -5.08 | -1.63 | speaker x sentence | 19 | -6.51 | -0.79 | 0.01 | 0.04 |
| full - LoRA v1 | dev_spont | 310 | 13.52 | 14.79 | -1.27 | -2.01 | -0.58 | speaker sign test (2/4 better) | 4 | -2.01 | -0.58 | 0.50 |  |
| full - LoRA v1 | itest_spont | 304 | 6.21 | 6.81 | -0.60 | -1.12 | -0.11 | speaker sign test (3/6 better) | 6 | -1.12 | -0.11 | 1.00 |  |
| full - LoRA v1 | ood | 109 | 13.67 | 14.88 | -1.21 | -2.35 | -0.15 | clip | 109 | -2.35 | -0.15 | 0.04 | 0.12 |
| full - LoRA v2 | dev_unseen | 278 | 9.73 | 13.12 | -3.39 | -4.83 | -2.11 | speaker x sentence | 17 | -6.49 | -1.18 | 0.003 |  |
| full - LoRA v2 | itest_unseen | 297 | 16.87 | 20.17 | -3.30 | -4.79 | -1.91 | speaker x sentence | 19 | -6.17 | -1.04 | 0.009 |  |
| full - LoRA v2 | dev_spont | 310 | 13.52 | 14.93 | -1.41 | -2.19 | -0.67 | speaker sign test (3/4 better) | 4 | -2.19 | -0.67 | 0.25 |  |
| full - LoRA v2 | itest_spont | 304 | 6.21 | 6.62 | -0.40 | -0.92 | 0.09 | speaker sign test (3/6 better) | 6 | -0.92 | 0.09 | 1.00 |  |
| full - LoRA v2 | ood | 109 | 13.67 | 15.11 | -1.44 | -2.76 | -0.15 | clip | 109 | -2.76 | -0.15 | 0.03 |  |
| soup - LoRA v1 | dev_unseen | 278 | 12.50 | 12.59 | -0.09 | -0.92 | 0.78 | speaker x sentence | 17 | -1.42 | 1.39 | 0.95 |  |
| soup - LoRA v1 | itest_unseen | 297 | 19.22 | 20.17 | -0.96 | -1.90 | -0.09 | speaker x sentence | 19 | -2.56 | 0.28 | 0.16 |  |
| soup - LoRA v1 | dev_spont | 310 | 14.77 | 14.79 | -0.02 | -0.36 | 0.33 | speaker sign test (2/4 better) | 4 | -0.36 | 0.33 | 1.00 |  |
| soup - LoRA v1 | itest_spont | 304 | 6.55 | 6.81 | -0.26 | -0.59 | 0.05 | speaker sign test (3/6 better) | 6 | -0.59 | 0.05 | 1.00 |  |
| soup - LoRA v1 | ood | 109 | 14.73 | 14.88 | -0.15 | -1.20 | 0.89 | clip | 109 | -1.20 | 0.89 | 0.85 |  |
| soup - LoRA v2 | dev_unseen | 278 | 12.50 | 13.12 | -0.62 | -1.42 | 0.09 | speaker x sentence | 17 | -1.84 | 0.56 | 0.26 |  |
| soup - LoRA v2 | itest_unseen | 297 | 19.22 | 20.17 | -0.96 | -2.01 | 0.00 | speaker x sentence | 19 | -2.91 | 0.52 | 0.24 |  |
| soup - LoRA v2 | dev_spont | 310 | 14.77 | 14.93 | -0.15 | -0.52 | 0.20 | speaker sign test (3/4 better) | 4 | -0.52 | 0.20 | 0.62 |  |
| soup - LoRA v2 | itest_spont | 304 | 6.55 | 6.62 | -0.07 | -0.29 | 0.16 | speaker sign test (2/6 better) | 6 | -0.29 | 0.16 | 1.00 |  |
| soup - LoRA v2 | ood | 109 | 14.73 | 15.11 | -0.38 | -1.51 | 0.76 | clip | 109 | -1.51 | 0.76 | 0.57 |  |
| LoRA v5 (cross-fit labels) - LoRA v2 | dev_unseen | 278 | 13.48 | 13.12 | 0.36 | -0.73 | 1.52 | speaker x sentence | 17 | -1.62 | 2.25 | 0.77 |  |
| LoRA v5 (cross-fit labels) - LoRA v2 | itest_unseen | 297 | 20.17 | 20.17 | 0.00 | -1.30 | 1.23 | speaker x sentence | 19 | -1.91 | 1.96 | 1.00 |  |
| LoRA v5 (cross-fit labels) - LoRA v2 | dev_spont | 310 | 14.20 | 14.93 | -0.72 | -1.24 | -0.21 | speaker sign test (2/4 better) | 4 | -1.24 | -0.21 | 1.00 |  |
| LoRA v5 (cross-fit labels) - LoRA v2 | itest_spont | 304 | 6.98 | 6.62 | 0.37 | -0.14 | 0.90 | speaker sign test (1/6 better) | 6 | -0.14 | 0.90 | 0.38 |  |
| LoRA v5 (cross-fit labels) - LoRA v2 | ood | 109 | 15.48 | 15.11 | 0.38 | -1.05 | 1.79 | clip | 109 | -1.05 | 1.79 | 0.64 |  |
| LoRA v2 seed 1 - seed 0 | dev_unseen | 278 | 13.66 | 13.12 | 0.54 | -0.73 | 1.87 | speaker x sentence | 17 | -1.79 | 2.85 | 0.69 |  |
| LoRA v2 seed 1 - seed 0 | itest_unseen | 297 | 20.09 | 20.17 | -0.09 | -1.27 | 1.04 | speaker x sentence | 19 | -2.05 | 1.60 | 0.96 |  |
| LoRA v2 seed 1 - seed 0 | dev_spont | 310 | 14.55 | 14.93 | -0.38 | -0.81 | 0.05 | speaker sign test (2/4 better) | 4 | -0.81 | 0.05 | 1.00 |  |
| LoRA v2 seed 1 - seed 0 | itest_spont | 304 | 6.83 | 6.62 | 0.21 | -0.13 | 0.55 | speaker sign test (1/6 better) | 6 | -0.13 | 0.55 | 0.38 |  |
| LoRA v2 seed 1 - seed 0 | ood | 109 | 14.43 | 15.11 | -0.68 | -2.07 | 0.70 | clip | 109 | -2.07 | 0.70 | 0.37 |  |
| LoRA v2 seed 2 - seed 0 | dev_unseen | 278 | 12.86 | 13.12 | -0.27 | -1.26 | 0.70 | speaker x sentence | 17 | -2.04 | 1.17 | 0.71 |  |
| LoRA v2 seed 2 - seed 0 | itest_unseen | 297 | 20.26 | 20.17 | 0.09 | -1.22 | 1.37 | speaker x sentence | 19 | -2.28 | 2.19 | 1.00 |  |
| LoRA v2 seed 2 - seed 0 | dev_spont | 310 | 14.77 | 14.93 | -0.15 | -0.63 | 0.30 | speaker sign test (3/4 better) | 4 | -0.63 | 0.30 | 0.25 |  |
| LoRA v2 seed 2 - seed 0 | itest_spont | 304 | 7.09 | 6.62 | 0.47 | 0.10 | 0.87 | speaker sign test (1/6 better) | 6 | 0.10 | 0.87 | 0.62 |  |
| LoRA v2 seed 2 - seed 0 | ood | 109 | 15.63 | 15.11 | 0.53 | -0.78 | 1.81 | clip | 109 | -0.78 | 1.81 | 0.44 |  |
| LoRA v2 seed 2 - seed 1 | dev_unseen | 278 | 12.86 | 13.66 | -0.80 | -2.02 | 0.36 | speaker x sentence | 17 | -3.43 | 1.30 | 0.49 |  |
| LoRA v2 seed 2 - seed 1 | itest_unseen | 297 | 20.26 | 20.09 | 0.17 | -1.11 | 1.46 | speaker x sentence | 19 | -2.11 | 2.46 | 0.94 |  |
| LoRA v2 seed 2 - seed 1 | dev_spont | 310 | 14.77 | 14.55 | 0.22 | -0.25 | 0.67 | speaker sign test (2/4 better) | 4 | -0.25 | 0.67 | 1.00 |  |
| LoRA v2 seed 2 - seed 1 | itest_spont | 304 | 7.09 | 6.83 | 0.26 | -0.14 | 0.70 | speaker sign test (1/6 better) | 6 | -0.14 | 0.70 | 0.38 |  |
| LoRA v2 seed 2 - seed 1 | ood | 109 | 15.63 | 14.43 | 1.21 | -0.08 | 2.53 | clip | 109 | -0.08 | 2.53 | 0.09 |  |
| enc+dec LoRA - LoRA v2 | dev_unseen | 278 | 10.89 | 13.12 | -2.23 | -3.71 | -0.80 | speaker x sentence | 17 | -5.20 | -0.21 | 0.03 |  |
| enc+dec LoRA - LoRA v2 | itest_unseen | 297 | 16.00 | 20.17 | -4.17 | -6.06 | -2.38 | speaker x sentence | 19 | -7.32 | -1.46 | 0.0062 |  |
| enc+dec LoRA - LoRA v2 | dev_spont | 310 | 14.57 | 14.93 | -0.36 | -1.17 | 0.37 | speaker sign test (2/4 better) | 4 | -1.17 | 0.37 | 1.00 |  |
| enc+dec LoRA - LoRA v2 | itest_spont | 304 | 6.79 | 6.62 | 0.18 | -0.30 | 0.65 | speaker sign test (4/6 better) | 6 | -0.30 | 0.65 | 0.69 |  |
| enc+dec LoRA - LoRA v2 | ood | 109 | 15.41 | 15.11 | 0.30 | -1.49 | 2.15 | clip | 109 | -1.49 | 2.15 | 0.76 |  |
| enc+dec LoRA seed 1 - LoRA v2 | dev_unseen | 278 | 10.54 | 13.12 | -2.59 | -4.19 | -1.07 | speaker x sentence | 17 | -5.49 | -0.39 | 0.03 |  |
| enc+dec LoRA seed 1 - LoRA v2 | itest_unseen | 297 | 15.30 | 20.17 | -4.87 | -6.80 | -3.03 | speaker x sentence | 19 | -8.37 | -1.92 | 0.0016 |  |
| enc+dec LoRA seed 1 - LoRA v2 | dev_spont | 310 | 14.53 | 14.93 | -0.40 | -1.13 | 0.27 | speaker sign test (3/4 better) | 4 | -1.13 | 0.27 | 0.25 |  |
| enc+dec LoRA seed 1 - LoRA v2 | itest_spont | 304 | 7.27 | 6.62 | 0.65 | 0.14 | 1.18 | speaker sign test (1/6 better) | 6 | 0.14 | 1.18 | 0.62 |  |
| enc+dec LoRA seed 1 - LoRA v2 | ood | 109 | 14.95 | 15.11 | -0.15 | -1.72 | 1.45 | clip | 109 | -1.72 | 1.45 | 0.91 |  |
| enc+dec LoRA seed 1 - seed 0 | dev_unseen | 278 | 10.54 | 10.89 | -0.36 | -1.45 | 0.71 | speaker x sentence | 17 | -2.37 | 1.52 | 0.74 |  |
| enc+dec LoRA seed 1 - seed 0 | itest_unseen | 297 | 15.30 | 16.00 | -0.70 | -1.84 | 0.43 | speaker x sentence | 19 | -2.64 | 1.05 | 0.46 |  |
| enc+dec LoRA seed 1 - seed 0 | dev_spont | 310 | 14.53 | 14.57 | -0.03 | -0.54 | 0.47 | speaker sign test (2/4 better) | 4 | -0.54 | 0.47 | 1.00 |  |
| enc+dec LoRA seed 1 - seed 0 | itest_spont | 304 | 7.27 | 6.79 | 0.47 | -0.11 | 1.08 | speaker sign test (2/6 better) | 6 | -0.11 | 1.08 | 0.69 |  |
| enc+dec LoRA seed 1 - seed 0 | ood | 109 | 14.95 | 15.41 | -0.45 | -1.92 | 1.00 | clip | 109 | -1.92 | 1.00 | 0.59 |  |
| full + enc+dec scorer - full | dev_unseen | 278 | 9.20 | 9.73 | -0.54 | -1.17 | 0.00 | speaker x sentence | 17 | -1.73 | 0.38 | 0.30 |  |
| full + enc+dec scorer - full | itest_unseen | 297 | 15.48 | 16.87 | -1.39 | -2.40 | -0.45 | speaker x sentence | 19 | -3.08 | 0.16 | 0.09 |  |
| full + enc+dec scorer - full | dev_spont | 310 | 13.52 | 13.52 | 0.00 | 0.00 | 0.00 | speaker sign test (0/4 better) | 4 | 0.00 | 0.00 | 1.00 |  |
| full + enc+dec scorer - full | itest_spont | 304 | 6.18 | 6.21 | -0.04 | -0.11 | 0.00 | speaker sign test (1/6 better) | 6 | -0.11 | 0.00 | 1.00 |  |
| full + enc+dec scorer - full | ood | 109 | 13.44 | 13.67 | -0.23 | -0.61 | 0.00 | clip | 109 | -0.61 | 0.00 | 0.25 |  |
| encoder-only LoRA - LoRA v2 | dev_unseen | 278 | 11.70 | 13.12 | -1.43 | -3.01 | 0.09 | speaker x sentence | 17 | -4.46 | 0.94 | 0.26 |  |
| encoder-only LoRA - LoRA v2 | itest_unseen | 297 | 16.43 | 20.17 | -3.74 | -5.67 | -1.91 | speaker x sentence | 19 | -7.22 | -0.72 | 0.02 |  |
| encoder-only LoRA - LoRA v2 | dev_spont | 310 | 14.75 | 14.93 | -0.17 | -1.01 | 0.63 | speaker sign test (3/4 better) | 4 | -1.01 | 0.63 | 0.62 |  |
| encoder-only LoRA - LoRA v2 | itest_spont | 304 | 6.70 | 6.62 | 0.09 | -0.39 | 0.55 | speaker sign test (3/6 better) | 6 | -0.39 | 0.55 | 1.00 |  |
| encoder-only LoRA - LoRA v2 | ood | 109 | 14.95 | 15.11 | -0.15 | -1.76 | 1.49 | clip | 109 | -1.76 | 1.49 | 0.91 |  |
| encoder-only LoRA - enc+dec LoRA | dev_unseen | 278 | 11.70 | 10.89 | 0.80 | -0.18 | 1.78 | speaker x sentence | 17 | -0.94 | 2.66 | 0.34 |  |
| encoder-only LoRA - enc+dec LoRA | itest_unseen | 297 | 16.43 | 16.00 | 0.43 | -0.35 | 1.24 | speaker x sentence | 19 | -0.97 | 1.89 | 0.53 |  |
| encoder-only LoRA - enc+dec LoRA | dev_spont | 310 | 14.75 | 14.57 | 0.19 | -0.38 | 0.76 | speaker sign test (1/4 better) | 4 | -0.38 | 0.76 | 1.00 |  |
| encoder-only LoRA - enc+dec LoRA | itest_spont | 304 | 6.70 | 6.79 | -0.09 | -0.46 | 0.27 | speaker sign test (2/6 better) | 6 | -0.46 | 0.27 | 1.00 |  |
| encoder-only LoRA - enc+dec LoRA | ood | 109 | 14.95 | 15.41 | -0.45 | -1.82 | 0.87 | clip | 109 | -1.82 | 0.87 | 0.54 |  |
| enc+dec LoRA rank 64 - rank 16 | dev_unseen | 278 | 10.98 | 10.89 | 0.09 | -0.99 | 1.17 | speaker x sentence | 17 | -1.55 | 2.17 | 0.96 |  |
| enc+dec LoRA rank 64 - rank 16 | itest_unseen | 297 | 14.78 | 16.00 | -1.22 | -2.43 | -0.09 | speaker x sentence | 19 | -3.12 | 0.55 | 0.18 |  |
| enc+dec LoRA rank 64 - rank 16 | dev_spont | 310 | 14.55 | 14.57 | -0.02 | -0.51 | 0.49 | speaker sign test (2/4 better) | 4 | -0.51 | 0.49 | 1.00 |  |
| enc+dec LoRA rank 64 - rank 16 | itest_spont | 304 | 7.09 | 6.79 | 0.30 | -0.28 | 0.89 | speaker sign test (2/6 better) | 6 | -0.28 | 0.89 | 1.00 |  |
| enc+dec LoRA rank 64 - rank 16 | ood | 109 | 14.95 | 15.41 | -0.45 | -1.92 | 1.02 | clip | 109 | -1.92 | 1.02 | 0.57 |  |
| final: views - no views | dev_spont | 310 | 13.76 | 13.93 | -0.17 | -0.33 | -0.05 | speaker sign test (2/4 better) | 4 | -0.33 | -0.05 | 0.50 |  |
| final: views - no views | ood | 109 | 14.12 | 14.43 | -0.30 | -0.79 | 0.15 | clip | 109 | -0.79 | 0.15 | 0.25 |  |

## 3 public systems

| comparison | set | clips | wer_a | wer_b | diff | clip_lo | clip_hi | design | units | lo | hi | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| submission - omniASR-LLM-1B zero-shot | ood | 109 | 14.12 | 20.85 | -6.72 | -9.24 | -4.39 | clip | 109 | -9.24 | -4.39 | 0.00 |
| submission - omniASR-LLM-7B zero-shot | ood | 109 | 14.12 | 23.26 | -9.14 | -11.63 | -6.72 | clip | 109 | -11.63 | -6.72 | 0.00 |
| submission - organizer XLS-R CPT silver | ood | 109 | 14.12 | 24.47 | -10.35 | -13.23 | -7.62 | clip | 109 | -13.23 | -7.62 | 0.00 |
| dev-mode full - organizer XLS-R CPT validated | itest_all | 909 | 13.52 | 17.46 | -3.94 | -5.28 | -2.63 | speaker x sentence | 21 | -6.66 | -1.18 | 0.008 |
| dev-mode full - organizer XLS-R CPT validated | itest_unseen | 297 | 16.87 | 17.74 | -0.87 | -3.23 | 1.56 | speaker x sentence | 19 | -5.14 | 3.57 | 0.69 |

## 4 ablation

| comparison | set | clips | wer_a | wer_b | diff | clip_lo | clip_hi | design | units | lo | hi | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - CTC log-likelihood (vs full) | dev_unseen | 278 | 10.77 | 9.64 | 1.13 | 0.48 | 1.87 | speaker x sentence | 17 | 0.08 | 2.58 | 0.04 |
| - CTC log-likelihood (vs full) | itest_unseen | 297 | 18.67 | 16.99 | 1.68 | 0.51 | 2.92 | speaker x sentence | 19 | -0.02 | 3.77 | 0.05 |
| - CTC log-likelihood (vs full) | dev_spont | 310 | 13.48 | 13.52 | -0.05 | -0.48 | 0.44 | speaker sign test (1/4 better) | 4 | -0.48 | 0.44 | 1.00 |
| - CTC log-likelihood (vs full) | itest_spont | 304 | 6.35 | 6.33 | 0.02 | -0.23 | 0.26 | speaker sign test (1/6 better) | 6 | -0.23 | 0.26 | 1.00 |
| - CTC log-likelihood (vs full) | ood | 109 | 14.43 | 14.22 | 0.20 | -0.80 | 1.28 | clip | 109 | -0.80 | 1.28 | 0.71 |
| - char 10-gram LM (vs full) | dev_unseen | 278 | 10.30 | 9.64 | 0.65 | 0.09 | 1.22 | speaker x sentence | 17 | -0.35 | 1.85 | 0.18 |
| - char 10-gram LM (vs full) | itest_unseen | 297 | 17.94 | 16.99 | 0.96 | 0.23 | 1.77 | speaker x sentence | 19 | -0.40 | 2.40 | 0.14 |
| - char 10-gram LM (vs full) | dev_spont | 310 | 13.86 | 13.52 | 0.34 | -0.02 | 0.74 | speaker sign test (0/4 better) | 4 | -0.02 | 0.74 | 0.25 |
| - char 10-gram LM (vs full) | itest_spont | 304 | 6.45 | 6.33 | 0.12 | -0.12 | 0.38 | speaker sign test (2/6 better) | 6 | -0.12 | 0.38 | 1.00 |
| - char 10-gram LM (vs full) | ood | 109 | 14.22 | 14.22 | 0.00 | -0.88 | 0.92 | clip | 109 | -0.88 | 0.92 | 1.00 |
| - length (vs full) | dev_unseen | 278 | 9.85 | 9.64 | 0.21 | -0.18 | 0.64 | speaker x sentence | 17 | -0.47 | 1.06 | 0.57 |
| - length (vs full) | itest_unseen | 297 | 16.84 | 16.99 | -0.14 | -0.64 | 0.38 | speaker x sentence | 19 | -1.28 | 0.79 | 0.68 |
| - length (vs full) | dev_spont | 310 | 13.86 | 13.52 | 0.33 | 0.01 | 0.69 | speaker sign test (1/4 better) | 4 | 0.01 | 0.69 | 1.00 |
| - length (vs full) | itest_spont | 304 | 6.34 | 6.33 | 0.01 | -0.22 | 0.24 | speaker sign test (2/6 better) | 6 | -0.22 | 0.24 | 1.00 |
| - length (vs full) | ood | 109 | 13.92 | 14.22 | -0.30 | -1.00 | 0.33 | clip | 109 | -1.00 | 0.33 | 0.39 |
| - all LLM likelihoods (vs full) | dev_unseen | 278 | 10.36 | 9.64 | 0.71 | -0.12 | 1.53 | speaker x sentence | 17 | -0.75 | 2.09 | 0.27 |
| - all LLM likelihoods (vs full) | itest_unseen | 297 | 17.04 | 16.99 | 0.06 | -1.14 | 1.16 | speaker x sentence | 19 | -1.98 | 2.17 | 0.95 |
| - all LLM likelihoods (vs full) | dev_spont | 310 | 14.62 | 13.52 | 1.09 | 0.58 | 1.66 | speaker sign test (1/4 better) | 4 | 0.58 | 1.66 | 1.00 |
| - all LLM likelihoods (vs full) | itest_spont | 304 | 6.55 | 6.33 | 0.22 | -0.05 | 0.50 | speaker sign test (2/6 better) | 6 | -0.05 | 0.50 | 1.00 |
| - all LLM likelihoods (vs full) | ood | 109 | 16.57 | 14.22 | 2.34 | 0.65 | 4.34 | clip | 109 | 0.65 | 4.34 | 0.005 |
| - zero-shot likelihood (vs full) | dev_unseen | 278 | 9.67 | 9.64 | 0.03 | -0.24 | 0.32 | speaker x sentence | 17 | -0.46 | 0.55 | 0.95 |
| - zero-shot likelihood (vs full) | itest_unseen | 297 | 17.01 | 16.99 | 0.03 | -0.27 | 0.32 | speaker x sentence | 19 | -0.52 | 0.54 | 0.92 |
| - zero-shot likelihood (vs full) | dev_spont | 310 | 13.49 | 13.52 | -0.03 | -0.15 | 0.10 | speaker sign test (3/4 better) | 4 | -0.15 | 0.10 | 0.25 |
| - zero-shot likelihood (vs full) | itest_spont | 304 | 6.32 | 6.33 | -0.01 | -0.10 | 0.08 | speaker sign test (2/6 better) | 6 | -0.10 | 0.08 | 1.00 |
| - zero-shot likelihood (vs full) | ood | 109 | 14.43 | 14.22 | 0.20 | -0.16 | 0.58 | clip | 109 | -0.16 | 0.58 | 0.30 |
| - LoRA v1 likelihood (vs full) | dev_unseen | 278 | 10.06 | 9.64 | 0.42 | 0.03 | 0.84 | speaker x sentence | 17 | -0.22 | 1.26 | 0.20 |
| - LoRA v1 likelihood (vs full) | itest_unseen | 297 | 16.81 | 16.99 | -0.17 | -0.81 | 0.40 | speaker x sentence | 19 | -1.19 | 0.69 | 0.69 |
| - LoRA v1 likelihood (vs full) | dev_spont | 310 | 13.52 | 13.52 | 0.00 | -0.13 | 0.13 | speaker sign test (2/4 better) | 4 | -0.13 | 0.13 | 1.00 |
| - LoRA v1 likelihood (vs full) | itest_spont | 304 | 6.36 | 6.33 | 0.04 | -0.07 | 0.14 | speaker sign test (1/6 better) | 6 | -0.07 | 0.14 | 0.62 |
| - LoRA v1 likelihood (vs full) | ood | 109 | 14.55 | 14.22 | 0.33 | -0.03 | 0.76 | clip | 109 | -0.03 | 0.76 | 0.08 |
| - LoRA v2 likelihood (vs full) | dev_unseen | 278 | 9.58 | 9.64 | -0.06 | -0.35 | 0.21 | speaker x sentence | 17 | -0.52 | 0.38 | 0.80 |
| - LoRA v2 likelihood (vs full) | itest_unseen | 297 | 17.07 | 16.99 | 0.09 | -0.23 | 0.42 | speaker x sentence | 19 | -0.50 | 0.77 | 0.80 |
| - LoRA v2 likelihood (vs full) | dev_spont | 310 | 13.60 | 13.52 | 0.08 | -0.04 | 0.20 | speaker sign test (0/4 better) | 4 | -0.04 | 0.20 | 0.25 |
| - LoRA v2 likelihood (vs full) | itest_spont | 304 | 6.37 | 6.33 | 0.04 | -0.07 | 0.16 | speaker sign test (1/6 better) | 6 | -0.07 | 0.16 | 0.38 |
| - LoRA v2 likelihood (vs full) | ood | 109 | 14.17 | 14.22 | -0.05 | -0.47 | 0.41 | clip | 109 | -0.47 | 0.41 | 0.84 |
| - view-count feature (vs full) | dev_unseen | 278 | 10.00 | 9.64 | 0.36 | 0.03 | 0.78 | speaker x sentence | 17 | -0.08 | 1.27 | 0.19 |
| - view-count feature (vs full) | itest_unseen | 297 | 17.54 | 16.99 | 0.55 | 0.14 | 1.08 | speaker x sentence | 19 | -0.10 | 1.54 | 0.12 |
| - view-count feature (vs full) | dev_spont | 310 | 13.78 | 13.52 | 0.25 | -0.07 | 0.59 | speaker sign test (0/4 better) | 4 | -0.07 | 0.59 | 0.12 |
| - view-count feature (vs full) | itest_spont | 304 | 6.32 | 6.33 | -0.01 | -0.30 | 0.27 | speaker sign test (3/6 better) | 6 | -0.30 | 0.27 | 1.00 |
| - view-count feature (vs full) | ood | 109 | 14.02 | 14.22 | -0.20 | -0.89 | 0.56 | clip | 109 | -0.89 | 0.56 | 0.56 |
| - perturbed-view candidates (vs full) | dev_unseen | 278 | 9.73 | 9.64 | 0.09 | 0.00 | 0.27 | speaker x sentence | 17 | 0.00 | 0.60 | 1.00 |
| - perturbed-view candidates (vs full) | itest_unseen | 297 | 17.13 | 16.99 | 0.14 | 0.00 | 0.38 | speaker x sentence | 19 | 0.00 | 0.61 | 0.68 |
| - perturbed-view candidates (vs full) | dev_spont | 310 | 13.74 | 13.52 | 0.21 | -0.01 | 0.47 | speaker sign test (2/4 better) | 4 | -0.01 | 0.47 | 1.00 |
| - perturbed-view candidates (vs full) | itest_spont | 304 | 6.33 | 6.33 | 0.00 | -0.11 | 0.10 | speaker sign test (1/6 better) | 6 | -0.11 | 0.10 | 0.62 |
| - perturbed-view candidates (vs full) | ood | 109 | 15.28 | 14.22 | 1.06 | 0.22 | 1.97 | clip | 109 | 0.22 | 1.97 | 0.01 |
| - CTC candidates (vs full) | dev_unseen | 278 | 9.64 | 9.64 | -0.00 | -0.34 | 0.40 | speaker x sentence | 17 | -0.58 | 0.78 | 0.86 |
| - CTC candidates (vs full) | itest_unseen | 297 | 17.16 | 16.99 | 0.17 | -0.37 | 0.68 | speaker x sentence | 19 | -0.81 | 1.00 | 0.62 |
| - CTC candidates (vs full) | dev_spont | 310 | 13.69 | 13.52 | 0.17 | -0.07 | 0.44 | speaker sign test (1/4 better) | 4 | -0.07 | 0.44 | 1.00 |
| - CTC candidates (vs full) | itest_spont | 304 | 6.37 | 6.33 | 0.04 | -0.08 | 0.16 | speaker sign test (1/6 better) | 6 | -0.08 | 0.16 | 0.62 |
| - CTC candidates (vs full) | ood | 109 | 14.15 | 14.22 | -0.08 | -0.60 | 0.44 | clip | 109 | -0.60 | 0.44 | 0.83 |
| - LLM beam candidates (vs full) | dev_unseen | 278 | 10.00 | 9.64 | 0.36 | -0.00 | 0.85 | speaker x sentence | 17 | -0.12 | 1.19 | 0.23 |
| - LLM beam candidates (vs full) | itest_unseen | 297 | 17.39 | 16.99 | 0.41 | 0.09 | 0.78 | speaker x sentence | 19 | -0.25 | 1.35 | 0.33 |
| - LLM beam candidates (vs full) | dev_spont | 310 | 13.73 | 13.52 | 0.21 | 0.01 | 0.41 | speaker sign test (0/4 better) | 4 | 0.01 | 0.41 | 0.25 |
| - LLM beam candidates (vs full) | itest_spont | 304 | 6.42 | 6.33 | 0.09 | -0.02 | 0.20 | speaker sign test (1/6 better) | 6 | -0.02 | 0.20 | 0.38 |
| - LLM beam candidates (vs full) | ood | 109 | 14.05 | 14.22 | -0.18 | -0.63 | 0.30 | clip | 109 | -0.63 | 0.30 | 0.45 |
| CTC candidates only (vs full) | dev_unseen | 278 | 15.33 | 9.64 | 5.68 | 3.99 | 7.46 | speaker x sentence | 17 | 2.76 | 8.96 | 0.0002 |
| CTC candidates only (vs full) | itest_unseen | 297 | 22.90 | 16.99 | 5.91 | 3.97 | 7.91 | speaker x sentence | 19 | 2.00 | 10.21 | 0.0052 |
| CTC candidates only (vs full) | dev_spont | 310 | 16.99 | 13.52 | 3.47 | 2.65 | 4.31 | speaker sign test (2/4 better) | 4 | 2.65 | 4.31 | 1.00 |
| CTC candidates only (vs full) | itest_spont | 304 | 7.20 | 6.33 | 0.87 | 0.42 | 1.33 | speaker sign test (0/6 better) | 6 | 0.42 | 1.33 | 0.06 |
| CTC candidates only (vs full) | ood | 109 | 19.56 | 14.22 | 5.34 | 3.35 | 7.36 | clip | 109 | 3.35 | 7.36 | 0.00 |
| control: views - none | dev_unseen | 278 | 9.64 | 10.09 | -0.45 | -0.91 | -0.09 | speaker x sentence | 17 | -1.47 | 0.06 | 0.12 |
| control: views - none | itest_unseen | 297 | 16.99 | 17.54 | -0.55 | -1.12 | -0.09 | speaker x sentence | 19 | -1.58 | 0.23 | 0.18 |
| control: views - none | dev_spont | 310 | 13.52 | 13.76 | -0.24 | -0.58 | 0.09 | speaker sign test (3/4 better) | 4 | -0.58 | 0.09 | 0.62 |
| control: views - none | itest_spont | 304 | 6.33 | 6.28 | 0.05 | -0.21 | 0.31 | speaker sign test (3/6 better) | 6 | -0.21 | 0.31 | 1.00 |
| control: views - none | ood | 109 | 14.22 | 14.63 | -0.40 | -1.50 | 0.61 | clip | 109 | -1.50 | 0.61 | 0.47 |
| control: samples - none | dev_unseen | 278 | 9.79 | 10.09 | -0.30 | -0.75 | 0.09 | speaker x sentence | 17 | -1.20 | 0.34 | 0.41 |
| control: samples - none | itest_unseen | 297 | 17.25 | 17.54 | -0.29 | -0.70 | 0.12 | speaker x sentence | 19 | -1.00 | 0.36 | 0.37 |
| control: samples - none | dev_spont | 310 | 13.68 | 13.76 | -0.09 | -0.31 | 0.14 | speaker sign test (4/4 better) | 4 | -0.31 | 0.14 | 0.12 |
| control: samples - none | itest_spont | 304 | 6.36 | 6.28 | 0.08 | -0.06 | 0.22 | speaker sign test (2/6 better) | 6 | -0.06 | 0.22 | 1.00 |
| control: samples - none | ood | 109 | 14.73 | 14.63 | 0.10 | -0.82 | 0.98 | clip | 109 | -0.82 | 0.98 | 0.82 |
| control: views - samples | dev_unseen | 278 | 9.64 | 9.79 | -0.15 | -0.50 | 0.21 | speaker x sentence | 17 | -0.77 | 0.46 | 0.60 |
| control: views - samples | itest_unseen | 297 | 16.99 | 17.25 | -0.26 | -0.84 | 0.26 | speaker x sentence | 19 | -1.22 | 0.57 | 0.59 |
| control: views - samples | dev_spont | 310 | 13.52 | 13.68 | -0.15 | -0.50 | 0.18 | speaker sign test (2/4 better) | 4 | -0.50 | 0.18 | 1.00 |
| control: views - samples | itest_spont | 304 | 6.33 | 6.36 | -0.03 | -0.25 | 0.20 | speaker sign test (3/6 better) | 6 | -0.25 | 0.20 | 1.00 |
| control: views - samples | ood | 109 | 14.22 | 14.73 | -0.50 | -1.56 | 0.48 | clip | 109 | -1.56 | 0.48 | 0.34 |
| soup added (vs full) | dev_unseen | 278 | 9.70 | 9.64 | 0.06 | -0.15 | 0.26 | speaker x sentence | 17 | -0.25 | 0.38 | 0.70 |
| soup added (vs full) | itest_unseen | 297 | 16.99 | 16.99 | 0.00 | -0.35 | 0.34 | speaker x sentence | 19 | -0.65 | 0.75 | 0.98 |
| soup added (vs full) | dev_spont | 310 | 13.60 | 13.52 | 0.07 | -0.06 | 0.22 | speaker sign test (0/4 better) | 4 | -0.06 | 0.22 | 0.50 |
| soup added (vs full) | itest_spont | 304 | 6.32 | 6.33 | -0.01 | -0.15 | 0.13 | speaker sign test (2/6 better) | 6 | -0.15 | 0.13 | 1.00 |
| soup added (vs full) | ood | 109 | 14.25 | 14.22 | 0.03 | -0.38 | 0.49 | clip | 109 | -0.38 | 0.49 | 0.99 |
| soup instead of v1+v2 (vs full) | dev_unseen | 278 | 9.79 | 9.64 | 0.15 | -0.18 | 0.49 | speaker x sentence | 17 | -0.41 | 0.74 | 0.63 |
| soup instead of v1+v2 (vs full) | itest_unseen | 297 | 16.70 | 16.99 | -0.29 | -0.74 | 0.15 | speaker x sentence | 19 | -1.21 | 0.51 | 0.46 |
| soup instead of v1+v2 (vs full) | dev_spont | 310 | 13.62 | 13.52 | 0.09 | 0.01 | 0.19 | speaker sign test (0/4 better) | 4 | 0.01 | 0.19 | 0.50 |
| soup instead of v1+v2 (vs full) | itest_spont | 304 | 6.35 | 6.33 | 0.02 | -0.05 | 0.08 | speaker sign test (2/6 better) | 6 | -0.05 | 0.08 | 1.00 |
| soup instead of v1+v2 (vs full) | ood | 109 | 14.33 | 14.22 | 0.10 | -0.21 | 0.40 | clip | 109 | -0.21 | 0.40 | 0.51 |

## 5 RL

| comparison | set | clips | wer_a | wer_b | diff | clip_lo | clip_hi | design | units | lo | hi | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decoding noise (LoRA v2 decoded again) - LoRA v2 | rl_unseen | 290 | 17.44 | 17.44 | 0.00 | 0.00 | 0.00 | speaker x sentence | 31 | 0.00 | 0.00 | 1.00 |
| decoding noise (LoRA v2 decoded again) - LoRA v2 | rl_spont | 295 | 14.43 | 14.45 | -0.02 | -0.12 | 0.06 | speaker sign test (1/4 better) | 4 | -0.12 | 0.06 | 1.00 |
| decoding noise (LoRA v2 decoded again) - LoRA v2 | ood | 109 | 15.11 | 15.11 | 0.00 | 0.00 | 0.00 | clip | 109 | 0.00 | 0.00 | 1.00 |
| SFT 400 - LoRA v2 | rl_unseen | 290 | 18.12 | 17.44 | 0.68 | -0.09 | 1.47 | speaker x sentence | 31 | -0.60 | 2.12 | 0.32 |
| SFT 400 - LoRA v2 | rl_spont | 295 | 14.76 | 14.45 | 0.31 | -0.18 | 0.80 | speaker sign test (0/4 better) | 4 | -0.18 | 0.80 | 0.25 |
| SFT 400 - LoRA v2 | ood | 109 | 16.16 | 15.11 | 1.06 | 0.00 | 2.11 | clip | 109 | 0.00 | 2.11 | 0.06 |
| GSPO 400 - LoRA v2 | rl_unseen | 290 | 18.12 | 17.44 | 0.68 | -0.26 | 1.69 | speaker x sentence | 31 | -0.96 | 2.50 | 0.45 |
| GSPO 400 - LoRA v2 | rl_spont | 295 | 14.66 | 14.45 | 0.22 | -0.38 | 0.79 | speaker sign test (0/4 better) | 4 | -0.38 | 0.79 | 0.25 |
| GSPO 400 - LoRA v2 | ood | 109 | 15.48 | 15.11 | 0.38 | -0.86 | 1.64 | clip | 109 | -0.86 | 1.64 | 0.60 |
| MWER 400 - LoRA v2 | rl_unseen | 290 | 17.69 | 17.44 | 0.26 | -0.52 | 1.04 | speaker x sentence | 31 | -1.03 | 1.72 | 0.78 |
| MWER 400 - LoRA v2 | rl_spont | 295 | 14.47 | 14.45 | 0.02 | -0.31 | 0.35 | speaker sign test (2/4 better) | 4 | -0.31 | 0.35 | 1.00 |
| MWER 400 - LoRA v2 | ood | 109 | 15.33 | 15.11 | 0.23 | -0.51 | 1.02 | clip | 109 | -0.51 | 1.02 | 0.62 |
| Dr. GRPO 400 - LoRA v2 | rl_unseen | 290 | 19.06 | 17.44 | 1.62 | 0.52 | 2.78 | speaker x sentence | 31 | -0.21 | 3.85 | 0.09 |
| Dr. GRPO 400 - LoRA v2 | rl_spont | 295 | 14.58 | 14.45 | 0.14 | -0.29 | 0.57 | speaker sign test (1/4 better) | 4 | -0.29 | 0.57 | 0.62 |
| Dr. GRPO 400 - LoRA v2 | ood | 109 | 15.48 | 15.11 | 0.38 | -0.93 | 1.84 | clip | 109 | -0.93 | 1.84 | 0.64 |
| MWER mixed rollouts 150 - LoRA v2 | rl_unseen | 290 | 17.69 | 17.44 | 0.26 | -0.43 | 0.95 | speaker x sentence | 31 | -1.01 | 1.44 | 0.68 |
| MWER mixed rollouts 150 - LoRA v2 | rl_spont | 295 | 14.47 | 14.45 | 0.02 | -0.18 | 0.21 | speaker sign test (1/4 better) | 4 | -0.18 | 0.21 | 1.00 |
| MWER mixed rollouts 150 - LoRA v2 | ood | 109 | 15.33 | 15.11 | 0.23 | -0.53 | 1.01 | clip | 109 | -0.53 | 1.01 | 0.64 |
| enc+dec start: SFT 400 - enc+dec LoRA | rl_unseen | 290 | 14.27 | 14.27 | 0.00 | -0.70 | 0.69 | speaker x sentence | 31 | -1.30 | 1.12 | 1.00 |
| enc+dec start: SFT 400 - enc+dec LoRA | rl_spont | 295 | 14.51 | 14.27 | 0.23 | -0.15 | 0.64 | speaker sign test (0/4 better) | 4 | -0.15 | 0.64 | 0.25 |
| enc+dec start: SFT 400 - enc+dec LoRA | ood | 109 | 16.16 | 15.41 | 0.76 | -0.16 | 1.71 | clip | 109 | -0.16 | 1.71 | 0.13 |
| enc+dec start: GSPO 400 - enc+dec LoRA | rl_unseen | 290 | 15.73 | 14.27 | 1.45 | 0.26 | 2.68 | speaker x sentence | 31 | -0.61 | 3.48 | 0.17 |
| enc+dec start: GSPO 400 - enc+dec LoRA | rl_spont | 295 | 14.53 | 14.27 | 0.25 | -0.21 | 0.72 | speaker sign test (0/4 better) | 4 | -0.21 | 0.72 | 0.25 |
| enc+dec start: GSPO 400 - enc+dec LoRA | ood | 109 | 17.52 | 15.41 | 2.11 | 0.78 | 3.51 | clip | 109 | 0.78 | 3.51 | 0.0018 |

## 6 proportions (Wilson 95 % intervals)

| proportion | k | n | share | lo | hi |
|---|---|---|---|---|---|
| beam contains a better transcript (LoRA v2, 150 held-out-sentence clips) | 9 | 150 | 0.06 | 0.03 | 0.11 |
| T=1 samples contain a better transcript | 21 | 150 | 0.14 | 0.09 | 0.20 |
| perturbed views contain a better transcript | 25 | 150 | 0.17 | 0.12 | 0.23 |
| union contains a better transcript | 36 | 150 | 0.24 | 0.18 | 0.31 |
| error events specific to one reading (146 multi-reading sentences) | 104 | 145 | 0.72 | 0.64 | 0.78 |
