#!/usr/bin/env /usr/bin/python3
"""Benchmark chess engines and planner/system metrics for the portfolio demo."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import chess
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_manipulator.chess.board import ChessBoardManager
from chess_manipulator.motion import BoardCalibration, JointLimitViolation, TrajectoryPlanner


@dataclass
class EngineResult:
    name: str
    legal_moves: int
    total_positions: int
    mean_ms: float
    median_ms: float
    reproducible_positions: int


@dataclass
class PlannerResult:
    total_cases: int
    successful_cases: int
    planning_success_rate: float
    invalid_ik_rejections: int
    collision_rejections: int
    mean_stage_count: float
    mean_planning_ms: float


@dataclass
class SequenceResult:
    total_moves: int
    completed_moves: int
    mean_planning_ms: float
    mean_stages_per_move: float
    estimated_execution_latency_ms: float
    estimated_end_to_end_ms: float


class UciEngine:
    """Minimal UCI driver for reproducible benchmarking."""

    def __init__(self, command: Sequence[str]) -> None:
        self._process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send("isready")
        self._wait_for("readyok")

    def close(self) -> None:
        self._send("quit")
        self._process.communicate(timeout=5)

    def bestmove(self, fen: str, movetime_ms: int) -> str:
        self._send(f"position fen {fen}")
        self._send(f"go movetime {movetime_ms}")
        line = self._wait_for("bestmove ")
        return line.split()[1]

    def _send(self, line: str) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()

    def _wait_for(self, prefix: str) -> str:
        assert self._process.stdout is not None
        started = time.perf_counter()
        while True:
            if time.perf_counter() - started > 30:
                raise TimeoutError(f"Timed out waiting for {prefix!r}")
            line = self._process.stdout.readline().strip()
            if line.startswith(prefix):
                return line


def _resolve_engine_command(name: str, config: Dict[str, object], override: str | None) -> List[str]:
    executable = override or str(config.get("executable", "")).strip()
    if not executable:
        raise ValueError(f"No executable provided for engine {name}.")
    if name == "sunfish" and executable.endswith(".py"):
        return ["/usr/bin/python3", executable]
    return [executable]


def benchmark_engine(name: str, command: Sequence[str], fens: Sequence[str], think_time: float) -> EngineResult:
    durations = []
    legal_count = 0
    reproducible = 0
    engine = UciEngine(command)

    try:
        for fen in fens:
            board = chess.Board(fen)
            first_move = engine.bestmove(fen, int(think_time * 1000))
            second_move = engine.bestmove(fen, int(think_time * 1000))
            started = time.perf_counter()
            move = chess.Move.from_uci(first_move)
            durations.append((time.perf_counter() - started) * 1000.0 + (think_time * 1000.0))
            if move in board.legal_moves:
                legal_count += 1
            if first_move == second_move:
                reproducible += 1
    finally:
        engine.close()

    return EngineResult(
        name=name,
        legal_moves=legal_count,
        total_positions=len(fens),
        mean_ms=statistics.mean(durations),
        median_ms=statistics.median(durations),
        reproducible_positions=reproducible,
    )


def benchmark_planner(planner: TrajectoryPlanner, cases: Sequence[Dict[str, str]]) -> PlannerResult:
    stage_counts = []
    durations = []
    invalid_ik = 0
    collision = 0
    success = 0
    for case in cases:
        board = ChessBoardManager(case["fen"])
        started = time.perf_counter()
        try:
            description = board.describe_move(case["move"])
            plan = planner.plan_move(description)
            stage_counts.append(len(plan.stages))
            success += 1
        except JointLimitViolation as exc:
            if "collides" in str(exc) or "safe planning envelope" in str(exc):
                collision += 1
            else:
                invalid_ik += 1
        except RuntimeError:
            invalid_ik += 1
        durations.append((time.perf_counter() - started) * 1000.0)
    return PlannerResult(
        total_cases=len(cases),
        successful_cases=success,
        planning_success_rate=(success / len(cases)) if cases else 0.0,
        invalid_ik_rejections=invalid_ik,
        collision_rejections=collision,
        mean_stage_count=statistics.mean(stage_counts) if stage_counts else 0.0,
        mean_planning_ms=statistics.mean(durations) if durations else 0.0,
    )


def benchmark_sequence(planner: TrajectoryPlanner, sequence: Dict[str, object]) -> SequenceResult:
    board = ChessBoardManager(str(sequence["starting_fen"]))
    stages = []
    planning_times = []
    completed = 0
    for move in sequence["moves"]:
        started = time.perf_counter()
        description = board.describe_move(move)
        plan = planner.plan_move(description)
        planning_ms = (time.perf_counter() - started) * 1000.0
        planning_times.append(planning_ms)
        stages.append(len(plan.stages))
        board.push_uci(move)
        completed += 1
    estimated_execution_ms = statistics.mean(stages) * planner.stage_time_sec * 1000.0 if stages else 0.0
    return SequenceResult(
        total_moves=len(sequence["moves"]),
        completed_moves=completed,
        mean_planning_ms=statistics.mean(planning_times) if planning_times else 0.0,
        mean_stages_per_move=statistics.mean(stages) if stages else 0.0,
        estimated_execution_latency_ms=estimated_execution_ms,
        estimated_end_to_end_ms=estimated_execution_ms + (statistics.mean(planning_times) if planning_times else 0.0),
    )


def write_outputs(results: Dict[str, object], csv_path: Path | None, json_path: Path | None) -> None:
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2))
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["section", "name", "metric", "value"])
            for engine in results["engines"]:
                for key, value in engine.items():
                    if key != "name":
                        writer.writerow(["engine", engine["name"], key, value])
            for key, value in results["planner"].items():
                writer.writerow(["planner", "planner", key, value])
            for key, value in results["sequence"].items():
                writer.writerow(["sequence", "opening_sequence", key, value])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default=str(REPO_ROOT / "config" / "benchmark_suite.yaml"))
    parser.add_argument("--stockfish", default="")
    parser.add_argument("--sunfish", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--csv-out", default="")
    args = parser.parse_args()

    suite = yaml.safe_load(Path(args.suite).read_text())
    planner = TrajectoryPlanner(BoardCalibration())

    engine_results = []
    for engine_name, engine_config in suite["engines"].items():
        override = args.stockfish if engine_name == "stockfish" else args.sunfish if engine_name == "sunfish" else ""
        try:
            command = _resolve_engine_command(engine_name, engine_config, override or None)
            engine_results.append(
                benchmark_engine(
                    name=engine_name,
                    command=command,
                    fens=suite["engine_fens"],
                    think_time=float(engine_config.get("think_time_sec", 0.05)),
                )
            )
        except FileNotFoundError:
            continue
        except ValueError:
            continue

    results = {
        "engines": [asdict(result) for result in engine_results],
        "planner": asdict(benchmark_planner(planner, suite["planner_cases"])),
        "sequence": asdict(benchmark_sequence(planner, suite["opening_sequence"])),
    }
    write_outputs(
        results=results,
        csv_path=Path(args.csv_out) if args.csv_out else None,
        json_path=Path(args.json_out) if args.json_out else None,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
