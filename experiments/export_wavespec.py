"""Compatibility shim: use datajax.cli.export_wavespec."""

from __future__ import annotations

from datajax.cli.export_wavespec import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
