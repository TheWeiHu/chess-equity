# chess-equity — project instructions

## Data policy: real Lichess data only (no synthetic data)

**Never use synthetic, fabricated, mocked, or hypothesized data to demonstrate or
validate the thesis.** The whole project is a claim about *real* outcomes, so every
number that supports it must come from a real Lichess dump.

- **Evidence/validation runs** (`reports/validation_real.md`, `calibration_real.md`, any
  headline/gate artifact) MUST be built from a real Lichess monthly dump via
  `chess-equity data build --month YYYY-MM` (dumps cached under
  `~/.cache/chess-equity/dumps/`). Real positions, real `cp_eval`, real ratings, real
  results. State the dump and `n` in the report header.
- **No hypothesized/measured-by-hand "practical" numbers** standing in for measured
  outcomes. If you need a practical win-rate, measure it from the real dataset (binned
  by cp/rating, with `n` reported); don't assert it.
- **No constructed/synthetic `PositionRow`s** as evidence. Tiny fixtures for *unit tests*
  of pure functions are fine (and should be labelled as fixtures), but they must never be
  presented as validation evidence or committed as a results artifact.
- **Tiny fixtures** (e.g. `data/sample/`) are for offline smoke tests only and must be
  clearly marked "illustrative, not evidence" (see `reports/validation_sample.md`).

If a task can't be done with real data unattended (needs a dump download), hold it for a
human rather than fabricating a stand-in.
