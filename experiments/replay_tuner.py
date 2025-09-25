"""Compatibility shim: use datajax.cli.replay_tuner."""

from __future__ import annotations

from datajax.cli.replay_tuner import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
