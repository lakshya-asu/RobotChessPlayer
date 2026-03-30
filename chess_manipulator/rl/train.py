"""CLI and trainer for bounded DQN training on python-chess environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Iterable, Optional, Sequence

import chess

from .checkpointing import append_checkpoint_event, merge_checkpoint_metadata
from .environment import ChessMoveEnv
from .model import build_model, require_torch, torch
from .replay import ReplayBuffer, Transition
from .trajectory_log import GameLog, LoggedTransition, build_game_log, write_game_log


class DQNTrainer:
    """Minimal DQN trainer intended for bounded experimentation."""

    def __init__(
        self,
        episodes: int,
        checkpoint_path: str,
        initial_checkpoint_path: str | None = None,
        batch_size: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        buffer_capacity: int = 10000,
        target_sync_steps: int = 100,
        warmup_steps: int = 250,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: int = 4000,
        max_halfmoves: int = 80,
        opponent: str = "random",
        agent_color: chess.Color = chess.BLACK,
        seed: int = 7,
        device: str = "cpu",
    ) -> None:
        require_torch()
        random.seed(seed)
        torch.manual_seed(seed)
        self.episodes = episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_sync_steps = target_sync_steps
        self.warmup_steps = warmup_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.env = ChessMoveEnv(
            max_halfmoves=max_halfmoves,
            opponent=opponent,
            seed=seed,
            agent_color=agent_color,
        )
        self.online = build_model().to(device)
        self.target = build_model().to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.global_step = 0
        self.update_step = 0
        if initial_checkpoint_path:
            self._load_checkpoint(initial_checkpoint_path)

    def run_episode(
        self,
        *,
        source: str = "self_play",
        game_id: str | None = None,
        log_path: Path | str | None = None,
        log_format: str = "json",
        training: bool = True,
        metadata: dict | None = None,
    ) -> GameLog:
        """Play one bounded game and optionally write a trajectory log."""
        state = self.env.reset()
        initial_fen = self.env.board.fen()
        transitions: list[LoggedTransition] = []
        reward_total = 0.0
        step_index = 0
        resolved_game_id = game_id or f"episode-{self.global_step + 1}"

        done = False
        while not done:
            board_before = self.env.board.copy(stack=False)
            legal_actions = self.env.legal_action_indices()
            action = self._choose_action(state, legal_actions)
            decoded = self.env.action_space.decode(action, board=board_before)
            result = self.env.step(action)
            next_legal = [] if result.done else self.env.legal_action_indices()
            fen_after = str(result.info.get("fen", self.env.board.fen()))
            transition = LoggedTransition(
                game_id=resolved_game_id,
                step_index=step_index,
                fen_before=board_before.fen(),
                fen_after=fen_after,
                side_to_move="white" if board_before.turn == chess.WHITE else "black",
                uci_move=decoded.move.uci() if decoded.is_legal else "",
                action_index=action,
                reward=float(result.reward),
                done=bool(result.done),
                legal_action_indices=list(next_legal),
                state=list(state),
                next_state=list(result.state),
                source=source,
                metadata={
                    "epsilon": self._epsilon(),
                    "opponent": self.env.opponent,
                    "training": training,
                },
            )
            transitions.append(transition)
            if training:
                self.buffer.add(transition.to_replay_transition())
                if len(self.buffer) >= max(self.batch_size, self.warmup_steps):
                    self._optimize()

            reward_total += float(result.reward)
            state = result.state
            done = result.done
            self.global_step += 1
            step_index += 1

        result_text = self.env.board.result(claim_draw=True) if self.env.board.is_game_over() else "*"
        game_log = build_game_log(
            game_id=resolved_game_id,
            source=source,
            initial_fen=initial_fen,
            final_fen=self.env.board.fen(),
            result=result_text,
            transitions=transitions,
            metadata={
                "episode_reward": reward_total,
                "steps": step_index,
                "opponent": self.env.opponent,
                **(metadata or {}),
            },
        )
        if log_path is not None:
            write_game_log(Path(log_path), game_log, log_format=log_format)
        return game_log

    def seed_buffer_from_game_logs(self, game_logs: Sequence[GameLog]) -> int:
        """Load replay transitions from prior game logs into the training buffer."""
        loaded = 0
        for game_log in game_logs:
            for transition in game_log.transitions:
                self.buffer.add(transition.to_replay_transition())
                loaded += 1
        return loaded

    def seed_buffer_from_transitions(self, transitions: Iterable[Transition]) -> int:
        """Load raw replay transitions into the training buffer."""
        loaded = 0
        for transition in transitions:
            self.buffer.add(transition)
            loaded += 1
        return loaded

    def fit_replay_buffer(self, optimization_steps: int) -> int:
        """Run a bounded number of optimization updates from the current replay buffer."""
        completed = 0
        for _ in range(optimization_steps):
            if len(self.buffer) < self.batch_size:
                break
            self._optimize()
            completed += 1
        return completed

    def save_checkpoint(self, history: dict) -> None:
        """Persist the current model and accompanying metadata."""
        self._save_checkpoint(history)

    def train(self) -> dict:
        history = {
            "episodes": self.episodes,
            "mean_reward": 0.0,
            "steps": 0,
            "update_steps": 0,
        }
        episode_rewards = []

        for episode_index in range(self.episodes):
            game_log = self.run_episode(
                source="self_play",
                game_id=f"train-{episode_index + 1}",
                training=True,
            )
            reward_total = float(game_log.metadata.get("episode_reward", 0.0))
            episode_rewards.append(reward_total)

        history["mean_reward"] = sum(episode_rewards) / max(len(episode_rewards), 1)
        history["steps"] = self.global_step
        history["update_steps"] = self.update_step
        self._save_checkpoint(history)
        return history

    def _choose_action(self, state: list[float], legal_actions: list[int]) -> int:
        if not legal_actions:
            raise RuntimeError("No legal actions available.")
        epsilon = self._epsilon()
        if random.random() < epsilon:
            return random.choice(legal_actions)
        with torch.inference_mode():
            inputs = torch.tensor([state], dtype=torch.float32, device=self.device)
            q_values = self.online(inputs)[0]
            legal_scores = [(float(q_values[index].item()), index) for index in legal_actions]
        return max(legal_scores, key=lambda item: (item[0], -item[1]))[1]

    def _optimize(self) -> None:
        batch = self.buffer.sample(self.batch_size)
        states = torch.tensor([item.state for item in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([item.action for item in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([item.next_state for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        current_q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q_values = self.target(next_states)
            next_q = []
            for row, transition in enumerate(batch):
                if transition.done or not transition.legal_next_actions:
                    next_q.append(0.0)
                    continue
                legal_max = max(float(target_q_values[row, index].item()) for index in transition.legal_next_actions)
                next_q.append(legal_max)
            next_q_tensor = torch.tensor(next_q, dtype=torch.float32, device=self.device)
            expected_q = rewards + ((1.0 - dones) * self.gamma * next_q_tensor)

        loss = self.loss_fn(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.update_step += 1
        if self.update_step % self.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

    def _epsilon(self) -> float:
        progress = min(self.global_step / max(self.epsilon_decay_steps, 1), 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def _save_checkpoint(self, history: dict) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.online.state_dict(),
                "history": history,
            },
            self.checkpoint_path,
        )
        metadata = {
            "checkpoint": str(self.checkpoint_path),
            "training": {
                "history": history,
                "episodes": self.episodes,
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "buffer_size": len(self.buffer),
                "target_sync_steps": self.target_sync_steps,
                "warmup_steps": self.warmup_steps,
            },
        }
        save_metadata = merge_checkpoint_metadata(self.checkpoint_path, metadata)
        append_checkpoint_event(self.checkpoint_path, "trained", save_metadata.get("training", {}))

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location=self.device)
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        self.online.load_state_dict(state_dict)
        self.target.load_state_dict(self.online.state_dict())
        self.online.to(self.device)
        self.target.to(self.device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a bounded DQN scaffold for chess move selection.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--checkpoint", required=True, help="Path to write the model checkpoint.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-capacity", type=int, default=10000)
    parser.add_argument("--target-sync-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay-steps", type=int, default=4000)
    parser.add_argument("--max-halfmoves", type=int, default=80)
    parser.add_argument("--opponent", choices=("random", "heuristic"), default="random")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    trainer = DQNTrainer(
        episodes=args.episodes,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        buffer_capacity=args.buffer_capacity,
        target_sync_steps=args.target_sync_steps,
        warmup_steps=args.warmup_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        max_halfmoves=args.max_halfmoves,
        opponent=args.opponent,
        seed=args.seed,
        device=args.device,
    )
    history = trainer.train()
    print(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
