# How to animate a scene (shared spec for the scene agents)

You animate **some scenes** of a 14.5-minute conference talk video: SIMBig 2026, Puno Quechua ASR. The video is made with Manim CE 0.22. All work happens in `<repo>/docs/video`.

The goal is **beautiful, smooth, calm, and obvious at a glance**. Think of the style of 3Blue1Brown and Distill: geometry first, numbers second, one idea on screen at a time, and generous pauses after each reveal.

## Read first

- `style.py`: palette, `txt()` helper, `TalkScene` base class, timing helpers. Use it; do not edit it. If you think it needs a change, say so in your final report.
- `scene_s01.py`: a finished example (title card).
- `script.md`: per scene, the narration and the **[bracketed visual directions]**. The directions are the plan; improve on them freely, but keep every number exactly as written.
- `out/timing.json`: `duration[Sxx]` is the scene length in seconds. `cues[Sxx]` lists `{t, text}` for each narration sentence, where `t` is the second within the scene at which that sentence starts. **Sync reveals to the words:** call `self.at(i)` (or `self.at(i, offset)`) before the animation that illustrates sentence i.
- Exact data, if you need more than the script gives:
  - `../../reports/fig_data/*.csv` (where_to_adapt, layer_ablation, rank_truncation, sentence_reuse)
  - `../../reports/probe_acc*.csv`, `../../reports/probe_cka*.csv`, `../../reports/verifier_curve.csv`, `../../reports/verifier_ours_ci.csv`, `../../reports/mechanism_*.csv`
  - `../paper/main.tex`

  **Never invent a number.**

## Structure of every scene

- One file per scene: `scene_sXX.py` containing `class SXX(TalkScene): sid = "SXX"`.
- `construct()` must:
  1. call `self.intro_chip()` first. It morphs the previous scene's section label into this one.
  2. build and animate the content, synced with `self.at(i)`;
  3. end with `self.end()`. It fades out all content and pads the scene to its **exact** duration.
- Never leave content that should persist outside `self.end()`'s fade. The next scene must start from a bare background, the progress line, and the chip.
- Do not call `self.wait()` past the scene duration. Before `self.end()`, keep at least about 1 s of the scene for the fade. If your animations are too long for the narration, shorten them.
- Keep the safe area: content within x ∈ [-6.6, 6.6] and y ∈ [-3.4, 3.2]. The chip sits at the top left, and the progress line along the very bottom.

## Motion language

**Smooth and purposeful, never flashy.**

- Default easing is `rate_func=SMOOTH` (ease-in-out cubic). Typical `run_time` is 0.8–1.6 s. Use `LaggedStart(..., lag_ratio=0.1–0.3)` for groups.
- Prefer **transformations over cuts**: `Transform` / `ReplacementTransform` / `TransformMatchingShapes` / `.animate` moves.
  - Morph one state into the next: a diagram becomes a chart, or a number counts to its value with `DecimalNumber` + `ChangeDecimalToValue`.
  - Dots slide along axes (`ValueTracker` + `always_redraw` where useful).
  - Bars grow from the baseline (`GrowFromEdge`), lines draw on (`Create`), arrows grow (`GrowArrow`).
- Direct attention with **opacity layering**: when you focus on one element, dim the others to about 0.25 opacity (`.animate.set_opacity(0.25)`), then restore them. A soft rounded highlight is fine: `SurroundingRectangle(corner_radius=0.1, stroke_width=2)` in a palette colour, faded in.
- **Avoid:**
  - `Write` on more than a short title;
  - spinning, bouncing, flashing, or `Indicate` spam;
  - rainbow colours;
  - 3D;
  - more than about 12 words of running text on screen (labels excepted);
  - text in series colours. Text is ink, INK2 or MUTED. Colour belongs to marks and to keywords via `t2c`.
- **Charts:**
  - thin axes in INK2 / GRID;
  - direct labels next to marks instead of legends;
  - value labels in ink;
  - a light reference line where the script mentions one;
  - fixed meanings for colours: DEC blue = decoder LoRA, ENC orange = encoder LoRA, REF grey = zero-shot / reference, LM4 violet = the word 4-gram system, SYN aqua = synthetic verifiers, INK = our system. Use the ENC_RAMP shades for encoder sizes.
- **Text:**
  - `txt()` uses Inter. Manim `Text` indexing skips spaces, so colour words with `t2c={"word": COLOR}`, not with slices.
  - Sizes: headline 40–48, body 28–32, labels 20–24, fine print ≥ 17.
- Build reusable pieces (axes, dumbbells) as small functions inside your scene file. Do not edit other agents' files.

## Check your work (required)

Render low quality into **your own media dir** (the scenes are worked on in parallel):

```
uv run manim -ql --disable_caching --media_dir media_<you> scene_sXX.py SXX
```

1. **Duration.** It must equal `duration[SXX]` within 0.1 s. Check with `ffprobe -v error -show_entries format=duration -of csv=p=0 <mp4>`.
2. **Frames.** Extract 6–10 frames at key moments (`ffmpeg -ss <t> -i <mp4> -frames:v 1 <png>`) into `/tmp/` and Read them. Look for overlaps, clipped text, cramped or empty layout, wrong numbers, and weak hierarchy. Iterate until every frame looks like a polished keynote slide.
3. **Clean up.** When done, delete your media dir and frame PNGs: the disk is nearly full. **Do not** render at high quality; the final 1080p60 render is done centrally.

**Constraints:**
- Do not touch the GPU servers, git, or any file outside `docs/video/scene_sXX.py` for your scenes. Read-only access elsewhere is fine.
- Python via `uv run` only.

## Report (final message)

Report per scene:
- the file;
- the measured duration vs target;
- 2–3 lines on the visual design;
- anything you could not resolve.
