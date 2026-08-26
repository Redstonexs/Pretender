"""``python -m pretender`` entry point.

Delegates to the CLI (``pretender.cli.main``), which lands in a later phase.
Until then, running the module reports the version and exits cleanly.
"""

from __future__ import annotations

import sys


def main() -> None:
    try:
        from pretender.cli import main as cli_main
    except ImportError:
        from pretender import __version__

        print(f"pretender {__version__} — CLI lands in a later phase.", file=sys.stderr)
        raise SystemExit(0)
    raise SystemExit(cli_main())


if __name__ == "__main__":
    main()
