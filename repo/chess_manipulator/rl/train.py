"""CLI for bounded DQN training on python-chess environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Optional

from .environment import ChessMoveEnv
from .model import build_model, require_torch, torch
from .replay import ReplayBuffer, Transition


class DQNTrainer:
    """Minimal DQN trainer intended for bounded experimentation."""

    def __init__(
        self,
        episodes: int,
        checkpoint_path: str,
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
        self.env = ChessMoveEnv(max_halfmoves=max_halfmoves, opponent=opponent, seed=seed)
        self.online = build_model().to(device)
        self.target = build_model().to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.global_step = 0

    def train(self) -> dict:
        history = {"episodes": self.episodes, "mean_reward": 0.0, "steps": 0}
        episode_rewards = []

        for _ in range(self.episodes):
            state = self.env.reset()
            done = False
            reward_total = 0.0

            while not done:
                legal_actions = self.env.legal_action_indices()
                action = self._choose_action(state, legal_actions)
                result = self.env.step(action)
                next_legal = [] if result.done else self.env.legal_action_indices()
                self.buffer.add(
                    Transition(
                        state=state,
                        action=action,
                        reward=result.reward,
                        next_state=result.state,
                        done=result.done,
                        legal_next_actions=next_legal,
                    )
                )
                reward_total += result.reward
                state = result.state
                done = result.done
                self.global_step += 1

                if len(self.buffer) >= max(self.batch_size, self.warmup_steps):
                    self._optimize()

                if self.global_step % self.target_sync_steps == 0:
                    self.target.load_state_dict(self.online.state_dict())

            episode_rewards.append(reward_total)

        history["mean_reward"] = sum(episode_rewards) / max(len(episode_rewards), 1)
        history["steps"] = self.global_step
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
        with torch.inference_mode():
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
        metadata_path = self.checkpoint_path.with_suffix(".json")
        metadata_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


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
