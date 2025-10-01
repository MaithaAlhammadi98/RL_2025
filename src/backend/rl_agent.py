from __future__ import annotations
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json, random, threading
from .state import state_key

DEFAULT_ACTIONS: List[str] = ["broad", "legal_only", "financial_only", "company_only"]

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

        src_dir  = Path(__file__).resolve().parents[1]
        data_dir = src_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.q_path = q_path or (data_dir / "q_table.json")

        self._lock = threading.Lock()
        self._Q: Dict[str, Dict[str, float]] = self._load()

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
        tmp = self.q_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._Q, f, indent=2, ensure_ascii=False)
        tmp.replace(self.q_path)

    # ---------- policy ----------
    def _ensure_state(self, s: str) -> Dict[str, float]:
        if s not in self._Q:
            self._Q[s] = {a: 0.0 for a in self.actions}
        return self._Q[s]

    def _epsilon_greedy(self, s: str) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q = self._ensure_state(s)
        return max(q.items(), key=lambda kv: kv[1])[0]

    # ---------- public API ----------
    def select(self, state: Dict[str, Any]) -> str:
        s = state_key(state)
        with self._lock:
            return self._epsilon_greedy(s)

    def update(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, Any] | None = None,
    ) -> None:
        s = state_key(state)
        with self._lock:
            q = self._ensure_state(s)
            old = q.get(action, 0.0)
            best_next = 0.0
            if next_state is not None:
                ns = state_key(next_state)
                best_next = max(self._ensure_state(ns).values(), default=0.0)
            target = reward + self.gamma * best_next
            q[action] = old + self.alpha * (target - old)
            self._save()
