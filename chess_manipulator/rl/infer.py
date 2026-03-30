"""CLI for selecting a legal UCI move from a FEN using the RL adapter."""

from __future__ import annotations

import argparse
import json
from typing import Optional

import chess

from .adapter import RLMoveAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select a legal move from a FEN using the RL adapter.")
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--checkpoint", help="Optional checkpoint produced by the trainer.")
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    adapter = RLMoveAdapter(checkpoint_path=args.checkpoint, device=args.device)
    result = adapter.select_move(args.fen)
    print(json.dumps({"uci_move": result.uci_move, "source": result.source}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
