#!/usr/bin/env python3
"""Operator entrypoint for offline RL continual learning."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_repo_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def main() -> int:
    _bootstrap_repo_path()
    from chess_manipulator.rl.continual import main as continual_main

    return continual_main()


if __name__ == "__main__":
    raise SystemExit(main())
