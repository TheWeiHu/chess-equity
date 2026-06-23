# Post-hoc Platt recalibration of maia2 — real Lichess dump (task 0166)

**Dump:** `lichess_db_standard_rated_2013-01`, 8000 positions built `--with-fen`; scored on a
game-level held-out test of **n=1398 rows / 22 games (seed 0)**; the Platt scaler is fit on
the disjoint **6602-row train split** (game-disjoint from the eval rows — no leakage). maia2
predictions came from the cached value head (`~/.cache/chess-equity/maia2.pkl`).

`maia2` is the gate's worst-calibrated rating-conditioned model. The recalibrator
(`validate/recalibration.py`) fits a two-parameter logistic on the *logit* of maia2's
prediction, `q = sigmoid(a·logit(p) + b)`, by Newton/IRLS minimising log-loss on the
calibration split, then applies it at eval. It is strictly monotonic, so it never re-orders
predictions — only their confidence. The CLI knob is `validate --recalibrate-maia2` (needs
`--holdout`); it is **off by default**, so committed default runs are byte-identical.

## Result — maia2 on the same 1398-row held-out test

| maia2 | log-loss | Brier | ECE |
|---|--:|--:|--:|
| **off** (raw)            | 0.6935 | 0.2332 | 0.1593 |
| **on** (Platt-recalibrated) | 0.6360 | 0.2183 | **0.1021** |

- **ECE 0.1593 → 0.1021 (−36%)** — the acceptance target: recalibration lowers ECE.
- log-loss and Brier also improve (0.6935 → 0.6360; 0.2332 → 0.2183).
- **Gate PASS is preserved** — maia2 still beats the rating-blind baseline on both metrics
  (log-loss −0.3985, Brier −0.0224 vs baseline; the off run was −0.3409 / −0.0076), so the
  recalibration *widens* the win rather than trading calibration for it.

## Caveat — this sample is all `<2000`

The task references maia2's high-rating (2000+) ECE blowup (~0.30). This 8000-row 2013-01
sample contains no 2000+ positions (the `By high_rating` slice is entirely `<2000`), so the
specific high-rating bin can't be exercised here. What this run shows is that the
recalibrator lowers ECE on real held-out data *and* keeps the gate PASS; demonstrating the
2000+ repair specifically needs a master-level dump (held tasks 0154/0153). Platt was chosen
over isotonic for robustness on the sparse high-rating bin (two parameters, not one knot per
bin); isotonic is left as a follow-up.

## Reproduce

```
chess-equity data build --month 2013-01 --with-fen --sample 8000 --out data/recal_2013-01
chess-equity validate --data data/recal_2013-01/dataset.csv \
  --models baseline,wdl-a,maia2 --holdout 0.2 --bootstrap 0 --recalibrate-maia2
```
(Drop `--recalibrate-maia2` for the "off" baseline. Needs `pip install maia2` + torch for the
real maia2 value head.)
