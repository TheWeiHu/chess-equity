"""Seed-stability of the gate verdict (task 0156): does PASS survive re-sampling?

The committed headline gate (``reports/validation_real.md``) runs at **one seed only**
(its header reads ``(seed 0)``). The seed governs two random draws: the game-level
held-out split (:func:`~chess_equity.validate.split.game_level_split`) and the paired
bootstrap that puts a CI on each model-vs-baseline delta. A skeptic can therefore call a
single-seed PASS a cherry-picked sample — maybe seed 0 happened to draw a favourable
test set.

This module answers that directly: re-draw under K seeds, re-run the **same** gate each
time, and report *verdict stability* — the fraction of seeds that PASS, plus the spread
of the headline log-loss delta and its CI across seeds. A PASS that holds across all (or
most) seeds is not a lucky draw; one that flickers is the skeptic's point made for them.

Pure functions over rows + :data:`~chess_equity.validate.harness.Predictor` callables —
the per-seed loop just calls the existing :func:`~chess_equity.validate.harness.evaluate`
/ :func:`~chess_equity.validate.harness.compare_to_baseline` /
:func:`~chess_equity.validate.harness.gate_verdicts` pipeline, so the multi-seed verdict
is the single-seed gate run K times, not a reimplementation. No I/O; runs torch-free on
``baseline``/``wdl-a`` predictors.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from chess_equity.data.schema import PositionRow
from chess_equity.validate.harness import (
    BASELINE_NAME,
    HEADLINE_METRIC,
    MIN_GATE_N,
    Predictor,
    compare_to_baseline,
    evaluate,
    gate_verdicts,
)
from chess_equity.validate.split import game_level_split


@dataclass(frozen=True)
class SeedVerdict:
    """One predictor's gate result under a single seed (task 0156).

    ``log_loss_delta`` is ``model - baseline`` on the held-out overall log-loss (negative
    = the model wins). ``ci_lo``/``ci_hi`` are the headline-metric paired-bootstrap CI
    bounds for that seed (``None`` when the run had no bootstrap). ``passed`` mirrors the
    gate's PASS for the seed, and ``underpowered`` records whether the held-out n fell
    below the floor so the verdict was INCONCLUSIVE rather than PASS/FAIL.
    """

    seed: int
    name: str
    passed: bool
    log_loss_delta: float
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    underpowered: bool
    held_out_n: int


@dataclass(frozen=True)
class ModelStability:
    """A predictor's verdict aggregated across all seeds (task 0156)."""

    name: str
    per_seed: List[SeedVerdict]

    @property
    def n_seeds(self) -> int:
        return len(self.per_seed)

    @property
    def n_pass(self) -> int:
        return sum(1 for v in self.per_seed if v.passed)

    @property
    def pass_fraction(self) -> float:
        return self.n_pass / self.n_seeds if self.per_seed else 0.0

    @property
    def all_pass(self) -> bool:
        return bool(self.per_seed) and self.n_pass == self.n_seeds

    @property
    def any_underpowered(self) -> bool:
        return any(v.underpowered for v in self.per_seed)

    @property
    def deltas(self) -> List[float]:
        return [v.log_loss_delta for v in self.per_seed]

    @property
    def delta_min(self) -> float:
        return min(self.deltas)

    @property
    def delta_max(self) -> float:
        return max(self.deltas)

    @property
    def delta_mean(self) -> float:
        return statistics.fmean(self.deltas)

    @property
    def ci_lo_min(self) -> Optional[float]:
        los = [v.ci_lo for v in self.per_seed if v.ci_lo is not None]
        return min(los) if los else None

    @property
    def ci_hi_max(self) -> Optional[float]:
        his = [v.ci_hi for v in self.per_seed if v.ci_hi is not None]
        return max(his) if his else None


@dataclass(frozen=True)
class SeedStability:
    """The full multi-seed stability summary for a gate run (task 0156)."""

    seeds: List[int]
    holdout: Optional[float]
    baseline_name: str
    models: List[ModelStability]


def reseed_stability(
    rows: Sequence[PositionRow],
    predictors: Dict[str, Predictor],
    *,
    seeds: Sequence[int],
    holdout: Optional[float] = None,
    baseline_name: str = BASELINE_NAME,
    headline_metric: str = HEADLINE_METRIC,
    n_resamples: int = 2000,
    ece_bins: int = 10,
    min_n: int = MIN_GATE_N,
) -> SeedStability:
    """Re-run the gate under each seed and aggregate the verdicts per predictor.

    For every seed: re-draw the held-out test split (when ``holdout`` is given) with that
    seed, score the predictors (:func:`evaluate`), put a paired-bootstrap CI on each delta
    seeded the same way (:func:`compare_to_baseline`, skipped when ``n_resamples`` is 0 or
    there is no challenger), and run :func:`gate_verdicts`. The result records each
    predictor's per-seed PASS, headline-metric delta + CI, and underpowered flag, so the
    caller can report how stable the verdict is across the draws.

    Pure computation. ``holdout`` of ``None`` means score the full row set every seed — the
    split no longer varies, but the bootstrap CI still re-draws, so the section still
    measures CI stability. Raises ``ValueError`` if ``seeds`` is empty.
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("reseed_stability needs at least one seed")

    has_bootstrap = (
        n_resamples > 0 and baseline_name in predictors and len(predictors) > 1
    )
    # name -> list of per-seed verdicts, registry order preserved via first-seen.
    collected: Dict[str, List[SeedVerdict]] = {}
    order: List[str] = []

    for seed in seeds:
        if holdout is not None:
            test = game_level_split(rows, test_fraction=holdout, seed=seed)[1]
        else:
            test = list(rows)
        reports = evaluate(test, predictors, bins=ece_bins)
        comparisons = (
            compare_to_baseline(
                test,
                predictors,
                baseline=baseline_name,
                n_resamples=n_resamples,
                seed=seed,
            )
            if has_bootstrap
            else None
        )
        verdicts = gate_verdicts(
            reports,
            baseline_name=baseline_name,
            comparisons=comparisons,
            headline_metric=headline_metric,
            min_n=min_n,
        )
        for v in verdicts:
            ci = v.headline_ci
            sv = SeedVerdict(
                seed=seed,
                name=v.name,
                passed=v.passed,
                log_loss_delta=v.log_loss_delta,
                ci_lo=ci.lo if ci is not None else None,
                ci_hi=ci.hi if ci is not None else None,
                underpowered=v.underpowered,
                held_out_n=v.held_out_n if v.held_out_n is not None else len(test),
            )
            if v.name not in collected:
                collected[v.name] = []
                order.append(v.name)
            collected[v.name].append(sv)

    models = [ModelStability(name=name, per_seed=collected[name]) for name in order]
    return SeedStability(
        seeds=seeds,
        holdout=holdout,
        baseline_name=baseline_name,
        models=models,
    )


def _stability_word(m: ModelStability) -> str:
    """A one-word read on how stable a predictor's verdict is across seeds."""
    if m.any_underpowered:
        return "inconclusive"
    if m.all_pass:
        return "stable"
    if m.n_pass == 0:
        return "fails"
    return "flickers"


def format_seed_stability(
    stability: SeedStability, *, title: str = "Seed stability"
) -> str:
    """Render the multi-seed verdict as a Markdown section (task 0156).

    A summary table — one row per predictor: how many seeds PASS, the log-loss delta range
    and mean across seeds, and the widest CI bounds — followed by a per-seed detail table
    and a plain-language line per predictor. Returns ``""`` when there is nothing to report
    (no non-baseline predictors).
    """
    if not stability.models:
        return ""
    k = len(stability.seeds)
    seed_list = ", ".join(str(s) for s in stability.seeds)
    holdout = (
        f"holdout={stability.holdout:g}"
        if stability.holdout is not None
        else "no holdout (full set each seed)"
    )
    out: List[str] = [f"## {title}", ""]
    out.append(
        f"Re-draws the held-out split and the bootstrap under K={k} seeds and re-runs the "
        "gate each time. A PASS that holds across seeds is not a cherry-picked sample; one "
        "that flickers is. **Negative log-loss delta = the model beats the baseline.**"
    )
    out.append("")
    out.append(f"Seeds: {seed_list} ({holdout}).")
    out.append("")
    out.append(
        "| model | seeds PASS | log-loss Δ (min … max) | mean Δ | log-loss 95% CI envelope | verdict |"
    )
    out.append("|---|:--:|--:|--:|:--:|:--:|")
    for m in stability.models:
        if m.ci_lo_min is not None and m.ci_hi_max is not None:
            ci = f"[{m.ci_lo_min:+.4f}, {m.ci_hi_max:+.4f}]"
        else:
            ci = "—"
        out.append(
            f"| {m.name} | {m.n_pass}/{m.n_seeds} | "
            f"{m.delta_min:+.4f} … {m.delta_max:+.4f} | {m.delta_mean:+.4f} | "
            f"{ci} | {_stability_word(m)} |"
        )
    out.append("")
    out.append("Per seed:")
    out.append("")
    out.append("| model | seed | log-loss Δ | 95% CI | PASS |")
    out.append("|---|--:|--:|:--:|:--:|")
    for m in stability.models:
        for v in m.per_seed:
            if v.ci_lo is not None and v.ci_hi is not None:
                ci = f"[{v.ci_lo:+.4f}, {v.ci_hi:+.4f}]"
            else:
                ci = "—"
            if v.underpowered:
                mark = "INCONCLUSIVE"
            else:
                mark = "yes" if v.passed else "no"
            out.append(f"| {m.name} | {v.seed} | {v.log_loss_delta:+.4f} | {ci} | {mark} |")
    out.append("")
    for m in stability.models:
        word = _stability_word(m)
        if word == "stable":
            out.append(
                f"- **{m.name}**: PASS in all {m.n_seeds}/{m.n_seeds} seeds — the verdict "
                "is stable, not a single-seed artifact."
            )
        elif word == "inconclusive":
            out.append(
                f"- **{m.name}**: at least one seed is underpowered (held-out n below the "
                "floor) — re-run with more data before reading the stability."
            )
        elif word == "fails":
            out.append(
                f"- **{m.name}**: PASS in 0/{m.n_seeds} seeds — does not beat the baseline."
            )
        else:
            out.append(
                f"- **{m.name}**: PASS in only {m.n_pass}/{m.n_seeds} seeds — the verdict "
                "flickers across re-samples, so a single-seed PASS would be cherry-picked."
            )
    out.append("")
    return "\n".join(out)
