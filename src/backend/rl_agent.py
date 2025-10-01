# src/backend/rl_agent.py
import json, random, os
from collections import defaultdict
from .retrieval_policies import ACTIONS

class RlAgent:
    def __init__(self, eps=0.1, alpha=0.2, gamma=0.0, path="data/q_table.json"):
        self.eps, self.alpha, self.gamma = eps, alpha, gamma
        self.path = path
        self.Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
        if os.path.exists(self.path):
            try:
                self.Q.update(json.load(open(self.path)))
            except Exception:
                pass

    def select(self, state: str) -> str:
        if random.random() < self.eps:
            return random.choice(ACTIONS)
        values = self.Q[state]
        return max(values, key=values.get)

    def update(self, state: str, action: str, reward: float):
        # one-step Q-learning (γ≈0); reduces to incremental bandit update
        q_sa = self.Q[state][action]
        new_q = q_sa + self.alpha * (reward - q_sa)
        self.Q[state][action] = new_q
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        json.dump(self.Q, open(self.path,"w"))
