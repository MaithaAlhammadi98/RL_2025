"""
Tabular ε-greedy RL agent with on-disk persistence.
- select(state)        -> action
- update(state, action, reward, next_state=None)

Q-table is stored under src/data/q_table.json.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import random
import threading

from .state import state_key

# Keep in sync with retrieval_policies.ACTIONS
DEFAULT_ACTIONS: List[str] = [
    "broad",           # no extra filter
    "legal_only",      # restrict to legal docs
    "financial_only",  # restrict to finance docs
    "company_only",    # prefer company-specific docs
]

class RLAgent:
    def __init__(
        self,
        actions: List[str] | None = None,
        epsilon: float = 0.2,
        alpha: float = 0.3,
        gamma: float = 0.9,
        q_path: Path | None = None,
    ) -> None:
        self.actions = actions or DEFAULT_ACTIONS
        self.epsilon = float(epsilon)
        self.alpha   = float(alpha)
        self.gamma   = float(gamma)

        # src/backend -> parents[1] == src
        src_dir = Path(__file__).resolve().parents[1]
        data_dir = src_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.q_path = q_path or (data_dir / "q_table.json")

        self._lock = threading.Lock()
        self._q: Dict[str, Dict[str, float]] = self._load()

    # ---------- persistence ----------
    def _load(self) -> Dict[str, Dict[str, float]]:
        if self.q_path.exists():
            try:
                with self.q_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self) -> None:
        with self.q_path.open("w", encoding="utf-8") as f:
            json.dump(self._q, f, indent=2, ensure_ascii=False)

    # ---------- helpers ----------
    def _ensure_state(self, skey: str) -> None:
        if skey not in self._q:
            self._q[skey] = {a: 0.0 for a in self.actions}

    def _best_action(self, skey: str) -> Tuple[str, float]:
        items = self._q[skey].items()
        return max(items, key=lambda kv: kv[1])

    # ---------- API ----------
    def select(self, state: Dict[str, Any]) -> str:
        """ε-greedy selection."""
        skey = state_key(state)
        with self._lock:
            self._ensure_state(skey)
            if random.random() < self.epsilon:
                action = random.choice(self.actions)
            else:
                action, _ = self._best_action(skey)
        return action

    def update(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, Any] | None = None,
    ) -> None:
        """One-step TD(0). If next_state omitted, do a supervised-like update."""
        skey = state_key(state)
        with self._lock:
            self._ensure_state(skey)
            old = self._q[skey].get(action, 0.0)
            if next_state is None:
                target = reward
            else:
                nskey = state_key(next_state)
                self._ensure_state(nskey)
                _, best_next = self._best_action(nskey)
                target = reward + self.gamma * best_next
            new = old + self.alpha * (target - old)
            self._q[skey][action] = float(new)
            self._save()
