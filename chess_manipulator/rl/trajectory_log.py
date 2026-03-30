"""Trajectory log schema and serialization helpers for offline RL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .replay import Transition


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_dict(mapping: Any) -> dict[str, Any]:
    return dict(mapping) if mapping else {}


@dataclass(frozen=True)
class LoggedTransition:
    """One logged transition from a chess game trajectory."""

    game_id: str
    step_index: int
    fen_before: str
    fen_after: str
    side_to_move: str
    uci_move: str
    action_index: int
    reward: float
    done: bool
    legal_action_indices: list[int]
    state: list[float]
    next_state: list[float]
    source: str = "self_play"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_replay_transition(self) -> Transition:
        return Transition(
            state=list(self.state),
            action=int(self.action_index),
            reward=float(self.reward),
            next_state=list(self.next_state),
            done=bool(self.done),
            legal_next_actions=list(self.legal_action_indices),
        )


@dataclass(frozen=True)
class GameLog:
    """Per-game trajectory container used for offline batch training."""

    game_id: str
    created_at: str
    source: str
    initial_fen: str
    final_fen: str
    result: str
    transitions: list[LoggedTransition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.transitions)


def build_game_log(
    game_id: str,
    source: str,
    initial_fen: str,
    final_fen: str,
    result: str,
    transitions: Sequence[LoggedTransition],
    metadata: dict[str, Any] | None = None,
) -> GameLog:
    return GameLog(
        game_id=game_id,
        created_at=_utc_now_iso(),
        source=source,
        initial_fen=initial_fen,
        final_fen=final_fen,
        result=result,
        transitions=list(transitions),
        metadata=_as_dict(metadata),
    )


def transition_to_dict(transition: LoggedTransition) -> dict[str, Any]:
    return {
        "game_id": transition.game_id,
        "step_index": transition.step_index,
        "fen_before": transition.fen_before,
        "fen_after": transition.fen_after,
        "side_to_move": transition.side_to_move,
        "uci_move": transition.uci_move,
        "action_index": transition.action_index,
        "reward": transition.reward,
        "done": transition.done,
        "legal_action_indices": list(transition.legal_action_indices),
        "state": list(transition.state),
        "next_state": list(transition.next_state),
        "source": transition.source,
        "metadata": _as_dict(transition.metadata),
    }


def transition_from_dict(payload: dict[str, Any]) -> LoggedTransition:
    return LoggedTransition(
        game_id=str(payload["game_id"]),
        step_index=int(payload.get("step_index", 0)),
        fen_before=str(payload.get("fen_before", "")),
        fen_after=str(payload.get("fen_after", "")),
        side_to_move=str(payload.get("side_to_move", "white")),
        uci_move=str(payload.get("uci_move", "")),
        action_index=int(payload.get("action_index", 0)),
        reward=float(payload.get("reward", 0.0)),
        done=bool(payload.get("done", False)),
        legal_action_indices=[int(index) for index in payload.get("legal_action_indices", [])],
        state=[float(value) for value in payload.get("state", [])],
        next_state=[float(value) for value in payload.get("next_state", [])],
        source=str(payload.get("source", "self_play")),
        metadata=_as_dict(payload.get("metadata")),
    )


def game_log_to_dict(game_log: GameLog) -> dict[str, Any]:
    return {
        "game_id": game_log.game_id,
        "created_at": game_log.created_at,
        "source": game_log.source,
        "initial_fen": game_log.initial_fen,
        "final_fen": game_log.final_fen,
        "result": game_log.result,
        "step_count": game_log.step_count,
        "transitions": [transition_to_dict(transition) for transition in game_log.transitions],
        "metadata": _as_dict(game_log.metadata),
    }


def game_log_from_dict(payload: dict[str, Any]) -> GameLog:
    transitions = [transition_from_dict(item) for item in payload.get("transitions", [])]
    if not transitions and payload.get("steps"):
        transitions = [transition_from_dict(item) for item in payload.get("steps", [])]
    return GameLog(
        game_id=str(payload.get("game_id") or payload.get("id") or _utc_now_iso()),
        created_at=str(payload.get("created_at") or _utc_now_iso()),
        source=str(payload.get("source", "self_play")),
        initial_fen=str(payload.get("initial_fen", "")),
        final_fen=str(payload.get("final_fen", "")),
        result=str(payload.get("result", "*")),
        transitions=transitions,
        metadata=_as_dict(payload.get("metadata")),
    )


def write_game_log(path: Path | str, game_log: GameLog, log_format: str = "json") -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if log_format == "jsonl":
        with destination.open("w", encoding="utf-8") as handle:
            for transition in game_log.transitions:
                payload = transition_to_dict(transition)
                payload.update(
                    {
                        "game_id": game_log.game_id,
                        "created_at": game_log.created_at,
                        "source": game_log.source,
                        "initial_fen": game_log.initial_fen,
                        "final_fen": game_log.final_fen,
                        "result": game_log.result,
                        "step_count": game_log.step_count,
                        "game_metadata": _as_dict(game_log.metadata),
                    }
                )
                handle.write(json.dumps(payload))
                handle.write("\n")
        return destination

    destination.write_text(json.dumps(game_log_to_dict(game_log), indent=2), encoding="utf-8")
    return destination


def _load_json_payload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_game_logs(paths: Sequence[Path | str]) -> list[GameLog]:
    grouped: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                _accumulate_game_payload(grouped, payload)
            continue

        payload = _load_json_payload(path)
        if isinstance(payload, list):
            for item in payload:
                _accumulate_game_payload(grouped, item)
        else:
            _accumulate_game_payload(grouped, payload)

    return [game_log_from_dict(item) for item in grouped.values()]


def _accumulate_game_payload(grouped: dict[str, dict[str, Any]], payload: dict[str, Any]) -> None:
    game_id = str(payload.get("game_id") or payload.get("id") or _utc_now_iso())
    entry = grouped.setdefault(
        game_id,
        {
            "game_id": game_id,
            "created_at": payload.get("created_at") or _utc_now_iso(),
            "source": payload.get("source", "self_play"),
            "initial_fen": payload.get("initial_fen", ""),
            "final_fen": payload.get("final_fen", ""),
            "result": payload.get("result", "*"),
            "transitions": [],
            "metadata": _as_dict(payload.get("metadata") or payload.get("game_metadata")),
        },
    )
    if payload.get("initial_fen"):
        entry["initial_fen"] = payload["initial_fen"]
    if payload.get("final_fen"):
        entry["final_fen"] = payload["final_fen"]
    if payload.get("result"):
        entry["result"] = payload["result"]
    if payload.get("created_at"):
        entry["created_at"] = payload["created_at"]
    if payload.get("source"):
        entry["source"] = payload["source"]
    if payload.get("metadata"):
        entry["metadata"] = _as_dict(payload.get("metadata"))
    if payload.get("game_metadata"):
        entry["metadata"] = _as_dict(payload.get("game_metadata"))

    if "transitions" in payload and isinstance(payload["transitions"], list):
        entry["transitions"].extend(payload["transitions"])
        return
    if "steps" in payload and isinstance(payload["steps"], list):
        entry["transitions"].extend(payload["steps"])
        return
    entry["transitions"].append(payload)


def replay_transitions_from_logs(game_logs: Iterable[GameLog]) -> list[Transition]:
    transitions: list[Transition] = []
    for game_log in game_logs:
        transitions.extend(transition.to_replay_transition() for transition in game_log.transitions)
    return transitions
