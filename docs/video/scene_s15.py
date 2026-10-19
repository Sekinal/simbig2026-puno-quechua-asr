"""S15: references (silent card at the end of the talk)."""

from __future__ import annotations

from manim import DOWN, LEFT, SEMIBOLD, UP, FadeIn, LaggedStart, VGroup
from style import INK, INK2, MUTED, SMOOTH, TalkScene, txt

REFS = [
    "Adelaar, W.F.H., Muysken, P.C.: The Languages of the Andes. Cambridge University Press (2004)",
    "Aghajanyan, A., et al.: Intrinsic dimensionality explains the effectiveness of language model"
    " fine-tuning. ACL-IJCNLP (2021)",
    "Ardila, R., et al.: Common Voice: a massively-multilingual speech corpus. LREC (2020)",
    "Bisani, M., Ney, H.: Bootstrap estimates for confidence intervals in ASR performance"
    " evaluation. ICASSP (2004)",
    "Fiscus, J.G.: A post-processing system to yield reduced word error rates: ROVER."
    " IEEE ASRU (1997)",
    "Hewitt, J., Liang, P.: Designing and interpreting probes with control tasks."
    " EMNLP-IJCNLP (2019)",
    "Holm, S.: A simple sequentially rejective multiple test procedure. Scand. J. Stat. (1979)",
    "Hu, E.J., et al.: LoRA: low-rank adaptation of large language models. ICLR (2022)",
    "Huaman, E., et al.: Quechua speech datasets in Common Voice: the case of Puno Quechua."
    " SIMBig 2025, CCIS 2895 (2026)",
    "Kornblith, S., et al.: Similarity of neural network representations revisited. ICML (2019)",
    "Liu, Z., et al.: Understanding R1-Zero-like training: a critical perspective. COLM (2025)",
    "Omnilingual ASR team: Omnilingual ASR: open-source multilingual speech recognition for"
    " 1600+ languages. arXiv:2511.09690 (2025)",
    "Owen, A.B.: The pigeonhole bootstrap. Ann. Appl. Stat. (2007)",
    "Prabhavalkar, R., et al.: Minimum word error rate training for attention-based"
    " sequence-to-sequence models. ICASSP (2018)",
    "Radford, A., et al.: Robust speech recognition via large-scale weak supervision. ICML (2023)",
    "Shi, M., et al.: Encoder awakening via adapters: effective domain-adaptive fine-tuning of"
    " speech-LLMs. IEEE SLT (2026)",
    "Zheng, C., et al.: Group sequence policy optimization. arXiv:2507.18071 (2025)",
    "Map data: Natural Earth (naturalearthdata.com), public domain",
    "Music: “Clean Soul” by Kevin MacLeod (incompetech.com), licensed under CC BY 4.0",
]
PAPER_1 = "Quezada Ramírez, I.E.: Puno Quechua ASR: when to adapt the speech encoder, and"
PAPER_2 = "verification as the bottleneck. SIMBig 2026, Shared Task 1 system description"


class S15(TalkScene):
    sid = "S15"

    def construct(self) -> None:
        self.intro_chip()
        paper = VGroup(
            txt("This talk", size=15, color=MUTED, weight=SEMIBOLD),
            txt(PAPER_1, size=15, color=INK),
            txt(PAPER_2, size=15, color=INK),
        ).arrange(DOWN, buff=0.07, aligned_edge=LEFT)
        paper.to_corner(UP + LEFT, buff=0.45).shift(0.55 * DOWN)
        rows = [txt(r, size=13.5, color=INK2) for r in REFS]
        body = VGroup(*rows).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        body.scale_to_fit_height(min(body.height, 5.2))
        if body.width > 12.9:
            body.scale_to_fit_width(12.9)
        body.next_to(paper, DOWN, buff=0.32, aligned_edge=LEFT)
        self.play(FadeIn(paper, shift=0.1 * UP), run_time=0.8, rate_func=SMOOTH)
        self.play(LaggedStart(*[FadeIn(r, shift=0.05 * LEFT) for r in rows], lag_ratio=0.04),
                  run_time=1.6, rate_func=SMOOTH)  # fmt: skip
        self.end(fade=1.2)
