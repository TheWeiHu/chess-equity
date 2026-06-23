# reports/SUMMARY.md — real-data gate index

One row per committed **real-Lichess** evidence report. Each verdict is **quoted or
parsed from that report's own header / `## Gate verdict` section** — this index reads no
data and computes no new numbers (see the real-data-only policy in `CLAUDE.md`).

**How each was produced:** see [REPRODUCE.md](REPRODUCE.md) — one documented
`chess-equity …` command per report below (dump month, n, model, seed/flags), with the
maia2/torch steps flagged as the attended, human-approved headline path. The
gate's PASS rule, stated verbatim in each validation report, is: a rating-conditioned
predictor **PASSes** when it has strictly lower log-loss **and** Brier than the
rating-blind `baseline` **and** its log-loss 95% bootstrap CI clears zero.

Verdict legend: **PASS** / **FAIL** = the report states a gate pass/fail; **info** = a
measurement report (calibration, disagreement, threshold fitting) that states no
PASS/FAIL gate.

| Report | Dump (month) | n | Verdict |
|---|---|--:|---|
| [validation_real.md](validation_real.md) — headline gate: does equity beat centipawns? | 2013-01 | 12,000 | **PASS** — `wdl-a` beats baseline (logloss −0.3403, CI [-0.3881,-0.2978] clears 0); `maia2` also PASS |
| [validation_real_xdump_refit.md](validation_real_xdump_refit.md) — cross-dump refit held-out (fit 2013-01, eval 2016-05) | fit 2013-01 / eval 2016-05 | 100,000 (OOD) | **PASS** — `wdl-a` still beats baseline OOD (logloss −0.2647, CI [-0.2778,-0.2514] clears 0); equity wins 17/17 powered slices — genuine held-out, no leakage |
| [validation_real_2016-05_high.md](validation_real_2016-05_high.md) — properly-powered high-rating (2000+) gate | 2016-05 | 49,269 high (held-out 10,134 / 148 games) | **PASS** — "the high-rating failure was a small-sample artifact"; `wdl-a` beats baseline overall (logloss 0.580 vs 0.848) and within 2000-2199/2200-2399 |
| [validation_real_2016-05.md](validation_real_2016-05.md) — cross-dump replication on the larger 2016-05 dump | 2016-05 | 100,000 (held-out 20,300 / 325 games) | **PASS (caveat)** — `wdl-a` is **in-distribution** here (`fit_month`=2016-05; leakage guard fires), so read it as a consistency check; the independent value is the model-free `baseline` + far bigger high-rating n |
| [goodmoves_real.md](goodmoves_real.md) — positive half: good moves read as good | 2013-01 | 12,000 (11,829 moves) | **PASS** — every bar reads engine-approved moves above blunders (Δgood > Δblunder) ✅ |
| [recalibration_maia2_real.md](recalibration_maia2_real.md) — post-hoc Platt recalibration of maia2 | 2013-01 | 1,398 held-out | **PASS** — ECE 0.1593 → 0.1021 (−36%); gate PASS preserved (caveat: sample all <2000) |
| [wdl_net_real.md](wdl_net_real.md) — Approach D: end-to-end board→WDL net | train 2016-05 / eval 2013-01 | 12,000 | **FAIL** — `wdl-net` log-loss 1.19 vs baseline 0.90; "not worth the complexity at this scale" |
| [calibration_real.md](calibration_real.md) — calibration by rating band | 2013-01 | 12,000 | **info** — ECE-by-band measurement of the rating-blind baseline; no gate |
| [failure_modes_real.md](failure_modes_real.md) — failure modes on binned real outcomes | 2013-01 | 12,000 | **info** — measured cp×rating cell comparison (baseline vs wdl-a vs measured); no gate |
| [divergence_real.md](divergence_real.md) — equity vs Stockfish disagreement | 2013-01 | 12,000 | **info** — product-visible disagreement only; reads no outcomes, so no gate |
| [drama_thresholds_real.md](drama_thresholds_real.md) — drama trigger thresholds | 2016-05 | 295,140 transitions / 4,860 games | **info** — calibrates Δequity thresholds on the real swing distribution; no gate |

## Reading the table

- **The thesis gate passes on real data.** The headline `validation_real.md` PASSes for
  both rating-conditioned predictors on a real 2013-01 held-out outcome set, and the
  positive-half `goodmoves_real.md` confirms good moves read as good.
- **It replicates across dumps and at master level.** The cross-dump rows
  (`validation_real_xdump_refit`, `validation_real_2016-05`, `validation_real_2016-05_high`)
  re-run the same gate on the larger, more-recent 2016-05 dump — `xdump_refit` is the
  clean OOD held-out (wins 17/17 powered slices), and the high-rating row shows the old
  2000+ "failure" was a small-sample artifact at the 2013-01 n=415. Read the plain
  `validation_real_2016-05` row with its leakage caveat (wdl-a in-distribution there).
- **One deliberate FAIL.** `wdl_net_real.md` is a *negative result kept on purpose*: the
  end-to-end board→WDL net loses to the centipawn baseline, so Approach A (`wdl-a`,
  regression on Stockfish `cp_eval`) stays the production path.
- **info rows are not gate evidence** — they characterise calibration, disagreement, and
  drama thresholds, but state no PASS/FAIL.

## Scope

This index covers the committed `reports/*_real.md` artifacts. The `*_sample.md` reports
(`validation_sample`, `calibration_sample`, `baseline_calibration_sample`) are
**illustrative offline smoke artifacts, not evidence**, and are intentionally excluded.
When new real-data reports land, regenerate this table from their headers.
