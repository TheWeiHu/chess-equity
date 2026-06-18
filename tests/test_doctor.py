"""Tests for ``chess-equity doctor`` — the optional-engine health check (task 0073).

The reporting logic is pure and the engine probes are injectable, so these tests run
with fakes: no Stockfish binary, no torch, no Maia-2 checkpoint, no network.
"""

import io

from chess_equity.doctor import Check, check, doctor, run_doctor


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
