"""Tests for ``chess-equity doctor`` — the optional-engine health check (task 0073).

The reporting logic is pure and the engine probes are injectable, so these tests run
with fakes: no Stockfish binary, no torch, no Maia-2 checkpoint, no network.
"""

import io

import pytest

from chess_equity.broadcast import FeedError, LocalPgnFeed, feed_from_spec
from chess_equity.doctor import Check, check, doctor, probe_broadcast, run_doctor

GAME_PGN = """[Event "Test Broadcast"]
[Site "https://lichess.org/abcd1234"]
[White "Carlsen"]
[Black "Nakamura"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *
"""

HEADER_ONLY_PGN = """[Event "Round not started"]
[White "Carlsen"]
[Black "Nakamura"]
[Result "*"]

*
"""


def test_check_maps_success_to_passing_check():
    c = check("stockfish", lambda: "cp=36")
    assert c == Check("stockfish", True, "cp=36")


def test_check_maps_exception_to_failing_check_with_its_message():
    def missing():
        raise RuntimeError("install with brew install stockfish")

    c = check("stockfish", missing)
    assert c.ok is False
    assert "brew install stockfish" in c.detail


def test_check_falls_back_to_class_name_for_blank_messages():
    class Boom(Exception):
        pass

    c = check("maia2", lambda: (_ for _ in ()).throw(Boom()))
    assert c.ok is False
    assert c.detail == "Boom"


def test_run_doctor_zero_exit_when_all_pass():
    out = io.StringIO()
    rc = run_doctor([Check("stockfish", True, "ok"), Check("maia2", True, "ok")], out)
    assert rc == 0
    assert "2/2 engines OK" in out.getvalue()


def test_run_doctor_nonzero_exit_and_marks_each_failure():
    out = io.StringIO()
    rc = run_doctor([Check("stockfish", True, "ok"), Check("maia2", False, "missing")], out)
    assert rc == 1
    text = out.getvalue()
    assert "[PASS] stockfish" in text
    assert "[FAIL] maia2: missing" in text
    assert "1/2 engines OK" in text


def test_doctor_uses_injected_probes_and_reports_both():
    out = io.StringIO()
    rc = doctor(
        out=out,
        probes={"stockfish": lambda: "works", "maia2": lambda: "works"},
    )
    assert rc == 0
    assert out.getvalue().count("[PASS]") == 2


def test_doctor_nonzero_when_one_injected_probe_raises():
    def broken():
        raise RuntimeError("not installed")

    rc = doctor(out=io.StringIO(), probes={"stockfish": lambda: "ok", "maia2": broken})
    assert rc == 1


def test_doctor_engines_filter_checks_only_the_named_engine():
    # A binary-only CI runner has Stockfish but no torch/Maia-2: restricting to
    # stockfish must skip the (would-fail) maia2 probe and exit 0.
    def maia_missing():
        raise RuntimeError("pip install maia2")

    out = io.StringIO()
    rc = doctor(
        out=out,
        probes={"stockfish": lambda: "ok", "maia2": maia_missing},
        engines=["stockfish"],
    )
    assert rc == 0
    text = out.getvalue()
    assert "[PASS] stockfish" in text
    assert "maia2" not in text
    assert "1/1 engines OK" in text


def test_doctor_engines_none_checks_all():
    out = io.StringIO()
    rc = doctor(
        out=out,
        probes={"stockfish": lambda: "ok", "maia2": lambda: "ok"},
        engines=None,
    )
    assert rc == 0
    assert out.getvalue().count("[PASS]") == 2


# --------------------------------------------------------------------------- #
# broadcast go-live preflight (task 0183)
# --------------------------------------------------------------------------- #


def test_probe_broadcast_passes_on_a_feed_emitting_moves():
    # A replayed PGN stands in for a live feed — no network. moves_per_poll high enough
    # that the single poll reveals the whole (4-half-move) game.
    detail = probe_broadcast(LocalPgnFeed(GAME_PGN, moves_per_poll=10))
    assert "game(s)" in detail
    assert "Nc6" in detail  # last parsed move surfaced for the streamer


class _NotStartedFeed:
    """A reachable feed that has only emitted game headers (round not started)."""

    def poll(self):
        return HEADER_ONLY_PGN


class _SilentFeed:
    """A reachable feed that has emitted nothing yet."""

    def poll(self):
        return None


class _DeadFeed:
    """An unreachable feed — poll raises like LichessRoundFeed/UrlPgnFeed do."""

    def poll(self):
        raise FeedError("lichess round abcd1234: connection refused")


def test_probe_broadcast_raises_when_round_has_no_moves_yet():
    with pytest.raises(RuntimeError, match="no moves yet"):
        probe_broadcast(_NotStartedFeed())


def test_probe_broadcast_raises_when_feed_silent():
    with pytest.raises(RuntimeError, match="no PGN yet"):
        probe_broadcast(_SilentFeed())


def test_probe_broadcast_propagates_unreachable_feed_error():
    with pytest.raises(FeedError, match="connection refused"):
        probe_broadcast(_DeadFeed())


def test_doctor_appends_broadcast_check_and_passes():
    out = io.StringIO()
    rc = doctor(
        out=out,
        probes={"stockfish": lambda: "ok"},
        broadcast_probe=lambda: probe_broadcast(LocalPgnFeed(GAME_PGN, moves_per_poll=10)),
    )
    assert rc == 0
    text = out.getvalue()
    assert "[PASS] stockfish" in text
    assert "[PASS] broadcast" in text


def test_doctor_nonzero_when_broadcast_check_fails():
    out = io.StringIO()
    rc = doctor(
        out=out,
        probes={"stockfish": lambda: "ok"},
        broadcast_probe=lambda: probe_broadcast(_DeadFeed()),
    )
    assert rc == 1
    assert "[FAIL] broadcast" in out.getvalue()


# --------------------------------------------------------------------------- #
# overlay bundle go-live preflight (task 0192)
# --------------------------------------------------------------------------- #

import json

from chess_equity.doctor import overlay_dir, probe_overlay, validate_overlay_event


def test_validate_overlay_event_accepts_a_well_formed_position():
    validate_overlay_event(
        {"type": "position", "ply": 44, "equity": 0.88, "cp": 60,
         "clock": {"white": 13.2, "black": 1.6},
         "drama": {"kind": "scramble", "magnitude": 0.55, "headline": "x"}}
    )


def test_validate_overlay_event_rejects_equity_out_of_range():
    with pytest.raises(ValueError, match=r"equity.*\[0,1\]"):
        validate_overlay_event({"type": "position", "ply": 1, "equity": 1.4})


def test_validate_overlay_event_rejects_missing_equity():
    with pytest.raises(ValueError, match="missing required 'equity'"):
        validate_overlay_event({"type": "position", "ply": 1})


def test_validate_overlay_event_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown/missing 'type'"):
        validate_overlay_event({"type": "wat", "equity": 0.5})


def test_validate_overlay_event_rejects_malformed_drama():
    with pytest.raises(ValueError, match="drama.magnitude"):
        validate_overlay_event(
            {"type": "position", "equity": 0.5, "drama": {"kind": "scramble", "headline": "x"}}
        )


def test_validate_overlay_event_tolerates_replay_extras():
    # a replay file's delayMs (and the file's _comment) are unknown extras → ignored
    validate_overlay_event({"type": "position", "equity": 0.5, "delayMs": 800})


def test_probe_overlay_passes_on_the_real_bundle():
    if overlay_dir() is None:
        pytest.skip("overlay/ bundle not present (installed wheel)")
    detail = probe_overlay()
    assert "bundle OK" in detail
    assert "to_overlay_event valid" in detail


def _write_bundle(directory, events):
    """Drop a minimal-but-valid overlay bundle into ``directory`` with the given events."""
    (directory / "index.html").write_text("<html><body></body></html>", encoding="utf-8")
    (directory / "config.html").write_text("<html><body></body></html>", encoding="utf-8")
    (directory / "overlay.js").write_text("// overlay\n", encoding="utf-8")
    (directory / "mock-game.json").write_text(json.dumps({"events": events}), encoding="utf-8")


def test_probe_overlay_fails_on_a_deliberately_malformed_event(tmp_path):
    # acceptance: a malformed bundled overlay event must fail the preflight.
    _write_bundle(tmp_path, [{"type": "position", "ply": 1, "equity": 9.9}])
    with pytest.raises(ValueError, match=r"equity.*\[0,1\]"):
        probe_overlay(tmp_path)


def test_probe_overlay_fails_on_missing_asset(tmp_path):
    _write_bundle(tmp_path, [{"type": "position", "equity": 0.5}])
    (tmp_path / "overlay.js").unlink()
    with pytest.raises(ValueError, match="overlay.js"):
        probe_overlay(tmp_path)


def test_doctor_appends_overlay_check_and_passes_on_real_bundle():
    if overlay_dir() is None:
        pytest.skip("overlay/ bundle not present (installed wheel)")
    out = io.StringIO()
    rc = doctor(out=out, probes={"stockfish": lambda: "ok"}, overlay_probe=probe_overlay)
    assert rc == 0
    text = out.getvalue()
    assert "[PASS] stockfish" in text
    assert "[PASS] overlay" in text


def test_doctor_nonzero_when_overlay_check_fails(tmp_path):
    _write_bundle(tmp_path, [{"type": "position", "equity": 5.0}])
    out = io.StringIO()
    rc = doctor(
        out=out,
        probes={"stockfish": lambda: "ok"},
        overlay_probe=lambda: probe_overlay(tmp_path),
    )
    assert rc == 1
    assert "[FAIL] overlay" in out.getvalue()


# --------------------------------------------------------------------------- #
# live SSE wiring go-live preflight (task 0209)
# --------------------------------------------------------------------------- #

from chess_equity.doctor import probe_serve_sse, sample_pgn_path  # noqa: E402


def test_probe_serve_sse_passes_on_the_committed_sample_pgn():
    # acceptance: binds the real serve-sse server on an ephemeral port over the committed
    # offline sample game and confirms /sse emits >=1 overlay position event.
    if sample_pgn_path() is None:
        pytest.skip("data/sample PGN not present (installed wheel)")
    detail = probe_serve_sse()
    assert "/sse bound on 127.0.0.1:" in detail
    assert "first position ply" in detail


def test_probe_serve_sse_passes_on_an_explicit_pgn(tmp_path):
    pgn = tmp_path / "game.pgn"
    pgn.write_text(GAME_PGN, encoding="utf-8")
    detail = probe_serve_sse(pgn)
    assert "overlay frame(s)" in detail


def test_probe_serve_sse_raises_on_missing_pgn(tmp_path):
    with pytest.raises(ValueError, match="sample PGN not found"):
        probe_serve_sse(tmp_path / "does-not-exist.pgn")


def test_doctor_appends_serve_sse_check_and_passes(tmp_path):
    pgn = tmp_path / "game.pgn"
    pgn.write_text(GAME_PGN, encoding="utf-8")
    out = io.StringIO()
    rc = doctor(
        out=out,
        probes={"stockfish": lambda: "ok"},
        serve_sse_probe=lambda: probe_serve_sse(pgn),
    )
    assert rc == 0
    text = out.getvalue()
    assert "[PASS] stockfish" in text
    assert "[PASS] serve-sse" in text


def test_doctor_nonzero_when_serve_sse_check_fails(tmp_path):
    out = io.StringIO()
    rc = doctor(
        out=out,
        probes={"stockfish": lambda: "ok"},
        serve_sse_probe=lambda: probe_serve_sse(tmp_path / "nope.pgn"),
    )
    assert rc == 1
    assert "[FAIL] serve-sse" in out.getvalue()


# --------------------------------------------------------------------------- #
# evidence gate preflight (task 0195)
# --------------------------------------------------------------------------- #

from chess_equity.doctor import (  # noqa: E402
    EVIDENCE_FAIL_ALLOWLIST,
    _parse_summary_rows,
    probe_evidence,
    reports_dir,
)

# A minimal SUMMARY.md table that mirrors the real shape (link + 3 columns, verdict last).
_SUMMARY_HEADER = (
    "# reports/SUMMARY.md — real-data gate index\n\n"
    "| Report | Dump | n | Verdict |\n|---|---|--:|---|\n"
)


def _write_summary(directory, rows):
    """Write SUMMARY.md with ``rows`` of (filename, desc, verdict_cell); create each report.

    Each report's body is set from ``verdict_cell`` so it corroborates by default — override
    by writing the report yourself after calling this.
    """
    lines = [_SUMMARY_HEADER]
    for filename, desc, verdict_cell in rows:
        lines.append(f"| [{desc}]({filename}) — {desc} | 2013-01 | 12,000 | {verdict_cell} |\n")
    (directory / "SUMMARY.md").write_text("".join(lines), encoding="utf-8")
    for filename, _desc, verdict_cell in rows:
        # default report body states a matching pass/fail token so corroboration holds
        cell = verdict_cell.upper()
        body = "PASS" if "PASS" in cell else ("FAIL" if "FAIL" in cell else "info")
        (directory / filename).write_text(f"# {filename}\n\nverdict: {body}\n", encoding="utf-8")


def test_parse_summary_rows_normalises_verdicts_and_skips_prose():
    rows = _parse_summary_rows(
        _SUMMARY_HEADER
        + "| [a.md](a.md) — x | m | 1 | **PASS** — wins |\n"
        + "| [b.md](b.md) — y | m | 2 | **PASS (caveat)** — in-dist |\n"
        + "| [c.md](c.md) — z | m | 3 | **FAIL** — loses |\n"
        + "| [d.md](d.md) — w | m | 4 | **info** — measurement |\n"
        + "\nSome prose with a [link](not_a_row.md) that is not a table row.\n"
    )
    assert rows == [
        ("a.md", "PASS"),
        ("b.md", "PASS"),
        ("c.md", "FAIL"),
        ("d.md", "info"),
    ]


def test_parse_summary_rows_raises_on_empty_table():
    with pytest.raises(ValueError, match="no parseable report rows"):
        _parse_summary_rows("# SUMMARY\n\njust prose, no table.\n")


def test_probe_evidence_passes_on_the_real_reports():
    if reports_dir() is None:
        pytest.skip("reports/ dir not present (installed wheel)")
    detail = probe_evidence()
    assert "gate index OK" in detail
    assert "deliberate FAIL" in detail


def test_probe_evidence_passes_with_goodmoves_prose_checkmark(tmp_path):
    # acceptance: a PASS report may state its pass in prose (✅) rather than the word PASS.
    (tmp_path / "SUMMARY.md").write_text(
        _SUMMARY_HEADER
        + "| [goodmoves_real.md](goodmoves_real.md) — gm | m | 1 | **PASS** — ok |\n",
        encoding="utf-8",
    )
    (tmp_path / "goodmoves_real.md").write_text(
        "# good moves\n\ngood reads as good ✅\n", encoding="utf-8"
    )
    assert "1 PASS" in probe_evidence(tmp_path)


def test_probe_evidence_fails_on_missing_report(tmp_path):
    _write_summary(tmp_path, [("validation_real.md", "headline", "**PASS** — wins")])
    (tmp_path / "validation_real.md").unlink()
    with pytest.raises(ValueError, match="missing on disk: validation_real.md"):
        probe_evidence(tmp_path)


def test_probe_evidence_fails_when_pass_report_states_no_pass(tmp_path):
    # acceptance: a regressed proof — SUMMARY still says PASS but the report no longer does.
    _write_summary(tmp_path, [("validation_real.md", "headline", "**PASS** — wins")])
    (tmp_path / "validation_real.md").write_text(
        "# headline\n\nresults were inconclusive\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="states no pass"):
        probe_evidence(tmp_path)


def test_probe_evidence_fails_on_unallowlisted_fail(tmp_path):
    # acceptance: any FAIL other than the deliberate wdl_net_real is a regression.
    _write_summary(tmp_path, [("validation_real.md", "headline", "**FAIL** — lost")])
    with pytest.raises(ValueError, match="regressed to FAIL"):
        probe_evidence(tmp_path)


def test_probe_evidence_allows_the_deliberate_wdl_net_fail(tmp_path):
    assert "wdl_net_real.md" in EVIDENCE_FAIL_ALLOWLIST
    _write_summary(tmp_path, [("wdl_net_real.md", "approach D", "**FAIL** — not worth it")])
    assert "1 deliberate FAIL" in probe_evidence(tmp_path)


def test_probe_evidence_missing_summary_raises(tmp_path):
    with pytest.raises(ValueError, match="missing reports/SUMMARY.md"):
        probe_evidence(tmp_path)


def test_doctor_appends_evidence_check_and_passes_on_real_reports():
    if reports_dir() is None:
        pytest.skip("reports/ dir not present (installed wheel)")
    out = io.StringIO()
    rc = doctor(out=out, probes={"stockfish": lambda: "ok"}, evidence_probe=probe_evidence)
    assert rc == 0
    assert "[PASS] evidence" in out.getvalue()


def test_doctor_nonzero_when_evidence_check_fails(tmp_path):
    _write_summary(tmp_path, [("validation_real.md", "headline", "**FAIL** — lost")])
    out = io.StringIO()
    rc = doctor(
        out=out,
        probes={"stockfish": lambda: "ok"},
        evidence_probe=lambda: probe_evidence(tmp_path),
    )
    assert rc == 1
    assert "[FAIL] evidence" in out.getvalue()


# --------------------------------------------------------------------------- #
# active equity-model preflight (task 0199)
# --------------------------------------------------------------------------- #

from chess_equity.doctor import (  # noqa: E402
    DoctorWarning,
    _wdl_a_provenance_warnings,
    probe_model,
)
from chess_equity.types import WDL, Equity  # noqa: E402


class _FakeModel:
    """A stand-in EquityModel returning a fixed White-POV bar (0..100)."""

    def __init__(self, bar):
        self._bar = bar

    def evaluate(self, fen, white_elo, black_elo):
        return Equity(wdl=WDL(0.5, 0.0, 0.5), equity_white=self._bar, source="fake")


def test_probe_model_passes_on_a_healthy_baseline():
    # acceptance: a healthy model asserts PASS — loads + finite in-range bar.
    detail = probe_model("baseline", build=lambda name: _FakeModel(63.0))
    assert "loads" in detail
    assert "win-equity 0.63" in detail


def test_probe_model_fails_on_a_non_finite_bar():
    with pytest.raises(ValueError, match="non-finite bar"):
        probe_model("baseline", build=lambda name: _FakeModel(float("nan")))


def test_probe_model_fails_on_an_out_of_range_bar():
    with pytest.raises(ValueError, match=r"outside \[0,100\]"):
        probe_model("baseline", build=lambda name: _FakeModel(150.0))


def test_probe_model_fails_on_unknown_model():
    # build raising (unknown name / failed load) is a hard FAIL via check().
    def boom(name):
        raise ValueError("unknown model 'nope'")

    with pytest.raises(ValueError, match="unknown model"):
        probe_model("nope", build=boom)


def test_probe_model_real_baseline_loads_and_passes():
    # the real baseline is torch-free; it must construct and produce a sane bar.
    detail = probe_model("baseline")
    assert "--model baseline loads" in detail


def test_probe_model_wdl_a_real_artifact_passes():
    # acceptance: the committed wdl-a artifact is healthy (n_train=50000, fit_month set).
    detail = probe_model("wdl-a")
    assert "n_train=50000" in detail
    assert "fit_month=2016-05" in detail


def test_probe_model_wdl_a_missing_artifact_fails(tmp_path):
    # acceptance: a missing artifact asserts FAIL before the model is even built.
    missing = tmp_path / "gone.json"
    with pytest.raises(ValueError, match="artifact missing on disk"):
        probe_model("wdl-a", build=lambda name: _FakeModel(50.0), artifact_path=missing)


def test_probe_model_wdl_a_garbled_artifact_fails(tmp_path):
    # acceptance: a garbled artifact asserts FAIL (parse/shape failure surfaced).
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact unreadable"):
        probe_model("wdl-a", build=lambda name: _FakeModel(50.0), artifact_path=bad)


def test_probe_model_wdl_a_missing_fit_month_warns(tmp_path):
    # acceptance: a model that works but lacks fit_month is a WARN, not a FAIL.
    import json

    art = tmp_path / "wdl_a.json"
    art.write_text(
        json.dumps(
            {
                "feature_version": 1,
                "weights": [[0.0] * 10, [0.0] * 10, [0.0] * 10],
                "meta": {"n_train": 50000},  # n_train fine, but no fit_month
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DoctorWarning, match="no fit_month"):
        probe_model("wdl-a", build=lambda name: _FakeModel(50.0), artifact_path=art)


def test_probe_model_wdl_a_seed_n_train_warns(tmp_path):
    import json

    art = tmp_path / "wdl_a.json"
    art.write_text(
        json.dumps(
            {
                "feature_version": 1,
                "weights": [[0.0] * 10, [0.0] * 10, [0.0] * 10],
                "meta": {"n_train": 50, "fit_month": "2016-05"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DoctorWarning, match="overfit seed"):
        probe_model("wdl-a", build=lambda name: _FakeModel(50.0), artifact_path=art)


def test_wdl_a_provenance_warnings_clean_on_real_meta():
    assert _wdl_a_provenance_warnings({"n_train": 50000, "fit_month": "2016-05"}) == []


def test_doctor_appends_model_check_and_marks_warn():
    # WARN is a passing state: exit 0 but the line reads [WARN], not [PASS].
    def warns():
        raise DoctorWarning("--model wdl-a loads — WARN: artifact has no fit_month")

    out = io.StringIO()
    rc = doctor(out=out, probes={"stockfish": lambda: "ok"}, model_probe=warns)
    assert rc == 0
    text = out.getvalue()
    assert "[WARN] model" in text
    assert "[FAIL]" not in text


def test_doctor_nonzero_when_model_check_fails():
    out = io.StringIO()
    rc = doctor(
        out=out,
        probes={"stockfish": lambda: "ok"},
        model_probe=lambda: probe_model("baseline", build=lambda n: _FakeModel(float("nan"))),
    )
    assert rc == 1
    assert "[FAIL] model" in out.getvalue()


def test_doctor_appends_model_check_and_passes_real_baseline():
    out = io.StringIO()
    rc = doctor(out=out, probes={"stockfish": lambda: "ok"}, model_probe=lambda: probe_model("baseline"))
    assert rc == 0
    assert "[PASS] model" in out.getvalue()


def test_feed_from_spec_dispatches_on_the_source_shape(tmp_path):
    from chess_equity.broadcast import LichessRoundFeed, UrlPgnFeed

    pgn_file = tmp_path / "game.pgn"
    pgn_file.write_text(GAME_PGN, encoding="utf-8")
    assert isinstance(feed_from_spec(str(pgn_file)), LocalPgnFeed)
    assert isinstance(feed_from_spec("https://example.com/round.pgn"), UrlPgnFeed)
    assert isinstance(feed_from_spec("abcd1234"), LichessRoundFeed)
