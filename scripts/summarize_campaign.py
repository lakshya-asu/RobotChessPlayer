#!/usr/bin/env python3
"""Aggregate campaign logs into JSON/Markdown metrics summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_game_logs(game_log_dir: Path) -> list[Path]:
    preferred = sorted(game_log_dir.glob("campaign_game_*.json"))
    if preferred:
        return preferred
    return sorted(game_log_dir.glob("*.json"))


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _summarize_games(game_logs: list[dict[str, Any]]) -> dict[str, Any]:
    total_games = len(game_logs)
    result_counts: dict[str, int] = {}
    termination_counts: dict[str, int] = {}
    plies_per_game: list[int] = []
    confidence_values: list[float] = []
    execution_successes = 0
    perception_successes = 0
    total_turns = 0
    milestone_games: list[dict[str, Any]] = []

    for index, log in enumerate(game_logs, start=1):
        result = str(log.get("result", "*"))
        termination = str(log.get("termination_reason", "unknown"))
        turns = list(log.get("turns", []))
        plies = len(turns)
        result_counts[result] = result_counts.get(result, 0) + 1
        termination_counts[termination] = termination_counts.get(termination, 0) + 1
        plies_per_game.append(plies)

        for turn in turns:
            total_turns += 1
            confidence_values.append(float(turn.get("perception_confidence", 0.0)))
            execution_successes += int(bool(turn.get("execution_success", False)))
            perception_successes += int(bool(turn.get("perception_confirmed", False)))

        if index == 1 or index % 50 == 0:
            milestone_games.append(
                {
                    "game_index": index,
                    "result": result,
                    "termination_reason": termination,
                    "plies": plies,
                    "final_fen": log.get("final_fen", ""),
                }
            )

    return {
        "games_completed": total_games,
        "avg_plies": _safe_mean([float(value) for value in plies_per_game]),
        "max_plies": max(plies_per_game) if plies_per_game else 0,
        "min_plies": min(plies_per_game) if plies_per_game else 0,
        "result_counts": result_counts,
        "termination_counts": termination_counts,
        "mean_perception_confidence": _safe_mean(confidence_values),
        "execution_success_rate": execution_successes / max(total_turns, 1),
        "perception_confirmation_rate": perception_successes / max(total_turns, 1),
        "milestone_games": milestone_games,
    }


def _summarize_offline_updates(offline_dir: Path) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    if not offline_dir.exists():
        return updates
    for run_dir in sorted(path for path in offline_dir.iterdir() if path.is_dir()):
        evaluation_path = run_dir / "evaluation.json"
        promotion_path = run_dir / "promotion.json"
        if not evaluation_path.exists():
            continue
        evaluation = _load_json(evaluation_path)
        promotion = _load_json(promotion_path) if promotion_path.exists() else {}
        updates.append(
            {
                "batch_id": run_dir.name,
                "positions": int(evaluation.get("positions", 0)),
                "legal_rate": float(evaluation.get("legal_rate", 0.0)),
                "agreement_rate": float(evaluation.get("agreement_rate", 0.0)),
                "promoted": bool(promotion.get("promoted", False)),
            }
        )
    return updates


def _build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Campaign Summary",
        "",
        "## Overview",
        "",
        f"- Campaign root: `{summary['campaign_root']}`",
        f"- Games completed: `{summary['games']['games_completed']}`",
        f"- Average plies per game: `{summary['games']['avg_plies']:.2f}`",
        f"- Mean perception confidence: `{summary['games']['mean_perception_confidence']:.3f}`",
        f"- Execution success rate: `{summary['games']['execution_success_rate']:.3f}`",
        f"- Perception confirmation rate: `{summary['games']['perception_confirmation_rate']:.3f}`",
        "",
        "## Results",
        "",
        "| Result | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(summary["games"]["result_counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Termination Reasons",
            "",
            "| Reason | Count |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(summary["games"]["termination_counts"].items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## Milestones",
            "",
            "| Game | Result | Termination | Plies |",
            "|---:|---|---|---:|",
        ]
    )
    for milestone in summary["games"]["milestone_games"]:
        lines.append(
            f"| {milestone['game_index']} | `{milestone['result']}` | `{milestone['termination_reason']}` | {milestone['plies']} |"
        )

    if summary["offline_updates"]:
        lines.extend(
            [
                "",
                "## Offline Updates",
                "",
                "| Batch | Legal Rate | Agreement Rate | Promoted |",
                "|---|---:|---:|---|",
            ]
        )
        for update in summary["offline_updates"]:
            lines.append(
                f"| `{update['batch_id']}` | {update['legal_rate']:.3f} | {update['agreement_rate']:.3f} | `{update['promoted']}` |"
            )

    lines.append("")
    return "\n".join(lines)


def summarize_campaign(campaign_root: Path) -> dict[str, Any]:
    game_log_dir = campaign_root / "game_logs"
    offline_dir = campaign_root / "offline_learning"
    game_logs = [_load_json(path) for path in _find_game_logs(game_log_dir)]
    return {
        "campaign_root": str(campaign_root),
        "game_logs": str(game_log_dir),
        "offline_learning": str(offline_dir),
        "games": _summarize_games(game_logs),
        "offline_updates": _summarize_offline_updates(offline_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a learned-player campaign.")
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-markdown", default="")
    args = parser.parse_args()

    campaign_root = Path(args.campaign_root)
    summary = summarize_campaign(campaign_root)
    payload = json.dumps(summary, indent=2)
    if args.output_json:
      Path(args.output_json).write_text(payload, encoding="utf-8")
    else:
      print(payload)

    if args.output_markdown:
        Path(args.output_markdown).write_text(_build_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
