"""Exploratory data analysis of the qxp corpora and the shared-task test set.

Writes figures (SVG + PNG) and a markdown report to ``reports/eda``. Acoustic descriptors are
cached in ``reports/eda/features.parquet``. Only aggregate statistics of the shared-task test set
are reported (data-use agreement).
"""

from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import duckdb
import polars as pl
import typer
from lets_plot import (
    LetsPlot,
    aes,
    element_blank,
    facet_wrap,
    geom_bar,
    geom_boxplot,
    geom_histogram,
    geom_line,
    geom_point,
    ggplot,
    ggsave,
    ggsize,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_log10,
    scale_y_log10,
    theme,
    theme_minimal2,
)

from qxp_asr.features import clip_stats
from qxp_asr.paths import MANIFESTS, REPORTS

LetsPlot.setup_html(no_js=True)

# Validated categorical palette (dataviz reference instance), fixed order per domain.
DOMAINS = [
    "scripted",
    "scripted_invalidated",
    "spont_gold",
    "spont_silver",
    "spont_unlabelled",
    "test_scripted",
    "test_spontaneous",
]
PALETTE = ["#2a78d6", "#4a3aa7", "#eb6834", "#eda100", "#e87ba4", "#1baf7a", "#008300"]
COLORS = dict(zip(DOMAINS, PALETTE, strict=True))
OUT = REPORTS / "eda"


def domain_expr() -> pl.Expr:
    c, b, lab = pl.col("corpus"), pl.col("bucket"), pl.col("label")
    return (
        pl.when(c == "test")
        .then(pl.lit("test_") + pl.col("test_type"))
        .when((c == "scripted") & (b == "validated"))
        .then(pl.lit("scripted"))
        .when(c == "scripted")
        .then(pl.lit("scripted_invalidated"))
        .when(lab.is_in(["gold", "pending"]))
        .then(pl.lit("spont_gold"))
        .when(lab == "silver")
        .then(pl.lit("spont_silver"))
        .otherwise(pl.lit("spont_unlabelled"))
    )


def acoustic_features(df: pl.DataFrame, workers: int) -> pl.DataFrame:
    cache = OUT / "features.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    paths = df["wav"].to_list()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        stats = list(pool.map(clip_stats, paths, chunksize=64))
    feats = pl.DataFrame(stats).with_columns(clip_id=df["clip_id"])
    feats.write_parquet(cache)
    return feats


def save(plot, name: str) -> None:
    for ext in ("svg", "png"):
        ggsave(plot, f"{name}.{ext}", path=str(OUT))


def style(plot):
    return plot + theme_minimal2() + theme(panel_grid_minor=element_blank())


def md_table(df: pl.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.iter_rows()]
    return "\n".join([head, sep, *rows])


def main(workers: int = typer.Option(os.cpu_count() or 8)) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = pl.read_parquet(MANIFESTS / "all.parquet").with_columns(domain=domain_expr())
    feats = acoustic_features(man, workers)
    df = man.join(feats, on="clip_id").with_columns(
        n_words=pl.col("text").str.split(" ").list.len(),
        n_chars=pl.col("text").str.replace_all(" ", "").str.len_chars(),
    )
    df = df.with_columns(
        chars_per_s=pl.col("n_chars") / (pl.col("duration_s") * pl.col("speech_frac")),
        silver_cps=pl.col("silver").str.replace_all(" ", "").str.len_chars()
        / (pl.col("duration_s") * pl.col("speech_frac")),
    )
    report: list[str] = ["# EDA — Puno Quechua (qxp) corpora and Task 1 test set\n"]

    # 1. Inventory -------------------------------------------------------------------------
    inv = (
        df.group_by("domain")
        .agg(
            clips=pl.len(),
            hours=(pl.col("duration_s").sum() / 3600).round(2),
            speakers=pl.col("speaker").n_unique(),
            mean_dur_s=pl.col("duration_s").mean().round(2),
            p95_dur_s=pl.col("duration_s").quantile(0.95).round(2),
            max_dur_s=pl.col("duration_s").max().round(1),
        )
        .sort(pl.col("domain").replace_strict(DOMAINS, list(range(len(DOMAINS)))))
    )
    report += ["## 1. Inventory\n", md_table(inv), ""]

    # 2. Durations -------------------------------------------------------------------------
    p = style(
        ggplot(df, aes("duration_s", fill="domain"))
        + geom_histogram(bins=60, alpha=0.85, color="white", size=0.2)
        + facet_wrap("domain", ncol=2, scales="free_y", order=1)
        + scale_fill_manual(values=COLORS)
        + scale_x_log10()
        + labs(title="Clip duration by domain (log x)", x="duration (s)", y="clips")
        + ggsize(900, 800)
    )
    save(p, "durations")

    # 3. Acoustic condition shift ----------------------------------------------------------
    long = df.unpivot(
        index=["domain"],
        on=["snr_db", "rms_dbfs", "speech_frac", "lead_sil_s", "chars_per_s"],
        variable_name="feature",
    ).filter(pl.col("value").is_finite())
    p = style(
        ggplot(long, aes("value", "domain", fill="domain"))
        + geom_boxplot(outlier_size=0.3, alpha=0.85, orientation="y")
        + facet_wrap("feature", ncol=1, scales="free_x")
        + scale_fill_manual(values=COLORS)
        + labs(title="Acoustic / rate descriptors: training domains vs test", x="", y="")
        + ggsize(900, 1300)
    )
    save(p, "acoustics")
    acou = (
        df.group_by("domain")
        .agg(
            [
                pl.col(c).median().round(3).alias(f"median_{c}")
                for c in ["snr_db", "rms_dbfs", "speech_frac", "lead_sil_s", "trail_sil_s"]
            ]
            + [
                (pl.col("clip_frac") > 1e-3).mean().round(4).alias("frac_clipped"),
                pl.col("chars_per_s").median().round(2).alias("median_chars_per_s"),
            ]
        )
        .sort("domain")
    )
    report += ["## 2. Acoustic conditions (medians)\n", md_table(acou), ""]

    # 4. Demographics ----------------------------------------------------------------------
    demo = (
        df.with_columns(pl.col("age").fill_null("unknown"), pl.col("gender").fill_null("unknown"))
        .group_by("domain", "age")
        .agg(hours=pl.col("duration_s").sum() / 3600)
        .with_columns(share=pl.col("hours") / pl.col("hours").sum().over("domain"))
    )
    p = style(
        ggplot(demo, aes("age", "share", fill="domain"))
        + geom_bar(stat="identity", position="dodge", width=0.8)
        + scale_fill_manual(values=COLORS)
        + labs(title="Share of audio hours by self-declared age", x="", y="share of hours")
        + ggsize(1000, 450)
    )
    save(p, "age")
    gender = (
        df.with_columns(pl.col("gender").fill_null("unknown"))
        .group_by("domain", "gender")
        .agg(h=(pl.col("duration_s").sum() / 3600).round(2))
        .pivot(on="gender", index="domain", values="h")
        .fill_null(0)
        .sort("domain")
    )
    age = (
        demo.with_columns(pl.col("hours").round(2))
        .pivot(on="age", index="domain", values="hours")
        .fill_null(0)
        .sort("domain")
    )
    report += ["## 3. Demographics (hours)\n", md_table(gender), "", md_table(age), ""]

    # 5. Speaker concentration ---------------------------------------------------------------
    spk = (
        df.filter(pl.col("speaker").is_not_null())
        .group_by("corpus", "speaker")
        .agg(hours=pl.col("duration_s").sum() / 3600)
        .with_columns(rank=pl.col("hours").rank("ordinal", descending=True).over("corpus"))
        .sort("corpus", "rank")
        .with_columns(
            cum_share=pl.col("hours").cum_sum().over("corpus")
            / pl.col("hours").sum().over("corpus")
        )
    )
    p = style(
        ggplot(spk, aes("rank", "cum_share", color="corpus"))
        + geom_line(size=1.2)
        + geom_point(size=2)
        + scale_color_manual(values={"scripted": PALETTE[0], "spontaneous": PALETTE[2]})
        + labs(
            title="Speaker concentration: cumulative share of hours",
            x="speaker rank",
            y="cumulative share",
        )
        + ggsize(800, 450)
    )
    save(p, "speakers")
    top = spk.filter(pl.col("rank") <= 5).group_by("corpus").agg(top5_share=pl.col("hours").sum())
    tot = spk.group_by("corpus").agg(total=pl.col("hours").sum(), n_speakers=pl.len())
    conc = top.join(tot, on="corpus").with_columns(
        top5_share=(pl.col("top5_share") / pl.col("total")).round(3), total=pl.col("total").round(2)
    )
    report += ["## 4. Speaker concentration\n", md_table(conc), ""]

    # 6. Split hygiene (organizer cv26 splits and official CV splits) -----------------------------
    con = duckdb.connect()
    cols = ["clip_id", "corpus", "speaker", "text", "org_split", "cv_split", "label"]
    con.register("m", df.select(cols).to_arrow())
    hyg = []
    for split_col in ("org_split", "cv_split"):
        for corpus in ("scripted", "spontaneous"):
            r = con.sql(
                f"""
                with s as (select * from m where corpus='{corpus}' and {split_col} is not null)
                , tr as (select * from s where {split_col} = 'train')
                select '{split_col}' as split_scheme, '{corpus}' as corpus,
                  b.{split_col} as eval_split,
                  count(distinct b.speaker) as eval_speakers,
                  count(distinct b.speaker) filter (b.speaker in (select speaker from tr))
                    as speakers_also_in_train,
                  count(*) as eval_clips,
                  round(avg((b.text in (select text from tr))::int), 3) as frac_text_seen_in_train
                from s b where b.{split_col} in ('dev', 'test')
                group by b.{split_col} order by 3
                """
            ).pl()
            hyg.append(r)
    report += ["## 5. Split hygiene\n", md_table(pl.concat(hyg)), ""]

    # 7. Text statistics -----------------------------------------------------------------------
    lab = df.filter(pl.col("text").is_not_null() & (pl.col("text") != ""))
    char_counts = Counter("".join(lab["text"].to_list()).replace(" ", ""))
    chars = pl.DataFrame({"char": list(char_counts), "count": list(char_counts.values())}).sort(
        "count", descending=True
    )
    inventory = " ".join(f"`{c}`:{n}" for c, n in chars.iter_rows())
    report += ["## 6. Text\n", f"Character inventory (all labelled targets): {inventory}\n"]

    def bag(domain: str, col: str) -> Counter[str]:
        texts = df.filter(pl.col("domain") == domain)[col].drop_nulls().to_list()
        return Counter(" ".join(texts).split())

    words = {
        "scripted": bag("scripted", "text"),
        "spont_gold": bag("spont_gold", "text"),
        "spont_silver": bag("spont_silver", "silver"),
    }
    zipf = pl.concat(
        [
            pl.DataFrame({"freq": sorted(c.values(), reverse=True)})
            .with_row_index("rank", offset=1)
            .with_columns(domain=pl.lit(d))
            for d, c in words.items()
        ]
    )
    p = style(
        ggplot(zipf, aes("rank", "freq", color="domain"))
        + geom_line(size=1.2)
        + scale_x_log10()
        + scale_y_log10()
        + scale_color_manual(values=COLORS)
        + labs(title="Word frequency vs rank (Zipf)", x="rank", y="frequency")
        + ggsize(800, 450)
    )
    save(p, "zipf")
    txt_rows = []
    for d, c in words.items():
        n_tok = sum(c.values())
        txt_rows.append(
            {
                "domain": d,
                "tokens": n_tok,
                "types": len(c),
                "hapax_frac_types": round(sum(1 for v in c.values() if v == 1) / len(c), 3),
                "type_token_ratio": round(len(c) / n_tok, 3),
                "mean_word_len": round(sum(len(w) * v for w, v in c.items()) / n_tok, 2),
            }
        )
    report += [md_table(pl.DataFrame(txt_rows)), ""]
    # OOV of spontaneous gold w.r.t. other vocabularies (proxy for unseen-word rate at test time)
    gold_tokens = list(words["spont_gold"].elements())
    for name, vocab in {
        "scripted": set(words["scripted"]),
        "spont_silver": set(words["spont_silver"]),
        "scripted+silver": set(words["scripted"]) | set(words["spont_silver"]),
    }.items():
        oov = sum(1 for w in gold_tokens if w not in vocab) / len(gold_tokens)
        report.append(f"- Spontaneous-gold token OOV rate vs **{name}** vocabulary: **{oov:.1%}**")
    sent = (
        df.filter(pl.col("corpus") == "scripted")
        .group_by("text")
        .agg(clips=pl.len(), speakers=pl.col("speaker").n_unique())
    )
    report += [
        "",
        f"- Scripted: {sent.height} distinct target sentences; clips per sentence median "
        f"{sent['clips'].median():.0f}, max {sent['clips'].max()}; "
        f"speakers per sentence median {sent['speakers'].median():.0f}.",
        "",
    ]

    # 8. Rate outliers -> likely label problems ------------------------------------------------
    rate = (
        df.filter(pl.col("chars_per_s").is_not_null())
        .group_by("domain")
        .agg(
            q01=pl.col("chars_per_s").quantile(0.01).round(2),
            median=pl.col("chars_per_s").median().round(2),
            q99=pl.col("chars_per_s").quantile(0.99).round(2),
            n_suspicious=((pl.col("chars_per_s") < 3) | (pl.col("chars_per_s") > 25)).sum(),
        )
        .sort("domain")
    )
    report += ["## 7. Speaking rate (chars per voiced second)\n", md_table(rate), ""]
    report += [
        "## Figures\n",
        *(f"![{n}]({n}.png)" for n in ("durations", "acoustics", "age", "speakers", "zipf")),
        "",
    ]
    (OUT / "EDA.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    typer.run(main)
