"""Allow ``python -m chess_equity.cli`` to run the CLI (package entry point)."""

from chess_equity.cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
