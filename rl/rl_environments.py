"""
RL Environments for LLMOps
===========================
Custom Gymnasium environments for optimizing RAG pipeline decisions.

Environments:
  1. RAGQualityEnv      — RL environment where the agent selects retrieval
                          parameters (k, reranking, chunking) and receives
                          RAGAS-style quality scores as rewards
  2. RetrievalBandit    — Multi-armed bandit for retrieval strategy selection
                          (dense / sparse / hybrid / reranked)
  3. ChunkingEnv        — Optimize chunk size and overlap for a document corpus
  4. PromptSelectionBandit — Bandit for selecting the best prompt template

All environments follow the Gymnasium API and run locally without external
services (they use mock/simulated reward functions by default, with hooks
to plug in real RAGAS evaluation).

Requirements: gymnasium, numpy

Usage:
    env = RAGQualityEnv()
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    # Or use with stable-baselines3:
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Gymnasium import with graceful fallback
# ──────────────────────────────────────────────────────────────────────────────

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        # Minimal stubs so the module is importable without gymnasium
        class _FakeSpaces:
            class Box:
                def __init__(self, low, high, shape=None, dtype=None):
                    self.low = np.array(low)
                    self.high = np.array(high)
                    self.shape = shape or (len(self.low),)
                    self.dtype = dtype or np.float32
                def sample(self):
                    return np.random.uniform(self.low, self.high).astype(self.dtype)
            class Discrete:
                def __init__(self, n):
                    self.n = n
                def sample(self):
                    return random.randint(0, self.n - 1)
        spaces = _FakeSpaces()

        class gym:
            class Env:
                metadata = {}
                def reset(self, **kw): ...
                def step(self, action): ...
                def render(self): ...
                def close(self): ...

        logger.warning("gymnasium not installed. Environments work but cannot be used with SB3.")


# ──────────────────────────────────────────────────────────────────────────────
# Simulated reward functions (replace with real RAGAS calls in production)
# ──────────────────────────────────────────────────────────────────────────────

def _simulated_ragas_score(
    k: int,
    rerank: bool,
    chunk_size: int,
    query_complexity: float,
    noise: float = 0.05,
) -> Dict[str, float]:
    """
    Simulate RAGAS metrics as a function of retrieval parameters.

    Ground truth relationships (approximate, for simulation):
      - Faithfulness: improves with reranking, degrades with very large k
      - Answer relevancy: improves with moderate k, degrades with too small/large
      - Context precision: improves with small k and reranking
      - Context recall: improves with large k

    Returns scores in [0, 1] for each metric.
    """
    rng = np.random.default_rng()

    # Base scores
    faithfulness = 0.5 + 0.3 * (1 / (1 + math.exp(-0.5 * (rerank * 3 - 1))))
    faithfulness -= 0.01 * max(0, k - 10)  # penalty for very large k

    answer_relevancy = 0.6 + 0.2 * math.exp(-0.5 * ((k - 5) / 3) ** 2)  # peaks at k=5
    answer_relevancy += 0.1 * rerank

    context_precision = 0.7 - 0.02 * k + 0.15 * rerank
    context_recall = 0.4 + 0.04 * k - 0.001 * k ** 2  # peaks around k=20

    # Chunk size effect: smaller chunks → better precision, worse recall
    chunk_effect = math.exp(-0.5 * ((chunk_size - 256) / 200) ** 2)
    context_precision += 0.1 * chunk_effect
    context_recall -= 0.05 * chunk_effect

    # Query complexity: harder queries benefit more from reranking
    if query_complexity > 0.7:
        faithfulness += 0.1 * rerank
        answer_relevancy += 0.05 * rerank

    # Add noise and clip
    def noisy_clip(x):
        return float(np.clip(x + rng.normal(0, noise), 0.0, 1.0))

    return {
        "faithfulness": noisy_clip(faithfulness),
        "answer_relevancy": noisy_clip(answer_relevancy),
        "context_precision": noisy_clip(context_precision),
        "context_recall": noisy_clip(context_recall),
    }


def _ragas_composite(scores: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """Weighted average of RAGAS metrics."""
    w = weights or {
        "faithfulness": 0.35,
        "answer_relevancy": 0.35,
        "context_precision": 0.15,
        "context_recall": 0.15,
    }
    return sum(scores[k] * w[k] for k in w if k in scores)


# ──────────────────────────────────────────────────────────────────────────────
# 1. RAG Quality Environment
# ──────────────────────────────────────────────────────────────────────────────

class RAGQualityEnv(gym.Env):
    """
    Gymnasium environment for optimizing RAG retrieval parameters.

    The agent selects:
      - k: number of retrieved documents (1–20)
      - use_reranking: whether to apply a reranker (0 or 1)
      - chunk_size: document chunk size in tokens (64–512)

    Observation:
      - query_complexity: float [0, 1] — how complex the current query is
      - recent_faithfulness: float [0, 1] — rolling average faithfulness
      - recent_relevancy: float [0, 1] — rolling average answer relevancy
      - step_in_episode: float [0, 1] — normalised step count

    Reward:
      - Composite RAGAS score (weighted average of 4 metrics)
      - Minus latency penalty (reranking + large k = slower)

    Episode:
      - 50 steps, each with a new random query
      - Terminated after 50 steps or if score drops below 0.2 for 5 steps
    """

    metadata = {"render_modes": ["human"]}

    # Action space: [k (1-20), rerank (0/1), chunk_size_idx (0-7)]
    K_OPTIONS = [1, 2, 3, 5, 7, 10, 15, 20]
    CHUNK_OPTIONS = [64, 128, 192, 256, 320, 384, 448, 512]
    N_ACTIONS = len(K_OPTIONS) * 2 * len(CHUNK_OPTIONS)  # 128

    def __init__(
        self,
        max_steps: int = 50,
        reward_fn: Optional[Callable] = None,
        latency_penalty: float = 0.01,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.reward_fn = reward_fn or _simulated_ragas_score
        self.latency_penalty = latency_penalty

        # Observation: [query_complexity, recent_faithfulness, recent_relevancy, step_frac]
        self.observation_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )
        # Discrete action: encodes (k_idx, rerank, chunk_idx)
        self.action_space = spaces.Discrete(self.N_ACTIONS)

        self._step = 0
        self._recent_scores: List[float] = []
        self._query_complexity = 0.5
        self._history: List[Dict] = []

    def _decode_action(self, action: int) -> Tuple[int, bool, int]:
        """Decode flat action index → (k, rerank, chunk_size)."""
        n_chunks = len(self.CHUNK_OPTIONS)
        n_rerank = 2
        chunk_idx = action % n_chunks
        action //= n_chunks
        rerank = bool(action % n_rerank)
        k_idx = action // n_rerank
        k_idx = min(k_idx, len(self.K_OPTIONS) - 1)
        return self.K_OPTIONS[k_idx], rerank, self.CHUNK_OPTIONS[chunk_idx]

    def _get_obs(self) -> np.ndarray:
        recent_faith = np.mean([s.get("faithfulness", 0.5) for s in self._history[-5:]]) if self._history else 0.5
        recent_rel = np.mean([s.get("answer_relevancy", 0.5) for s in self._history[-5:]]) if self._history else 0.5
        return np.array([
            self._query_complexity,
            recent_faith,
            recent_rel,
            self._step / self.max_steps,
        ], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
        self._step = 0
        self._recent_scores = []
        self._history = []
        self._query_complexity = float(np.random.uniform(0.1, 0.9))
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        k, rerank, chunk_size = self._decode_action(int(action))

        # Evaluate retrieval quality
        scores = self.reward_fn(k, rerank, chunk_size, self._query_complexity)
        composite = _ragas_composite(scores)

        # Latency penalty: reranking adds ~50ms, each extra doc ~5ms
        latency_cost = self.latency_penalty * (k * 0.5 + (10 if rerank else 0))
        reward = float(composite - latency_cost)

        self._history.append(scores)
        self._recent_scores.append(composite)
        self._step += 1

        # New query complexity for next step
        self._query_complexity = float(np.random.uniform(0.1, 0.9))

        terminated = self._step >= self.max_steps
        # Early termination if consistently bad
        if len(self._recent_scores) >= 5 and np.mean(self._recent_scores[-5:]) < 0.2:
            terminated = True

        info = {
            "k": k, "rerank": rerank, "chunk_size": chunk_size,
            "ragas_scores": scores, "composite": composite,
            "latency_cost": latency_cost,
        }
        return self._get_obs(), reward, terminated, False, info

    def render(self) -> Optional[str]:
        if not self._history:
            return "No steps taken yet."
        last = self._history[-1]
        return (
            f"Step {self._step}/{self.max_steps}  "
            f"F={last['faithfulness']:.3f}  "
            f"AR={last['answer_relevancy']:.3f}  "
            f"CP={last['context_precision']:.3f}  "
            f"CR={last['context_recall']:.3f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Retrieval Strategy Bandit
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalBandit:
    """
    Multi-armed bandit for retrieval strategy selection.

    Arms (strategies):
      0: Dense retrieval only (FAISS cosine)
      1: Sparse retrieval only (BM25)
      2: Hybrid (dense + sparse, RRF fusion)
      3: Dense + cross-encoder reranker
      4: Hybrid + cross-encoder reranker

    Algorithm: Upper Confidence Bound (UCB1) — balances exploration/exploitation.
    Also implements Thompson Sampling (Beta-Bernoulli) as an alternative.

    The bandit adapts to query distribution shifts — if one strategy starts
    performing worse, UCB will naturally explore alternatives.
    """

    STRATEGIES = [
        "dense_only",
        "sparse_only",
        "hybrid",
        "dense_reranked",
        "hybrid_reranked",
    ]

    # Simulated true mean rewards per strategy (unknown to the bandit)
    _TRUE_MEANS = [0.72, 0.65, 0.78, 0.82, 0.85]

    def __init__(
        self,
        n_arms: int = 5,
        algorithm: str = "ucb1",   # ucb1 | thompson | epsilon_greedy
        epsilon: float = 0.1,
        reward_fn: Optional[Callable[[int, Dict], float]] = None,
    ):
        self.n_arms = n_arms
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.reward_fn = reward_fn or self._simulated_reward

        self._counts = np.zeros(n_arms)
        self._values = np.zeros(n_arms)
        self._alpha = np.ones(n_arms)   # Thompson: Beta distribution alpha
        self._beta = np.ones(n_arms)    # Thompson: Beta distribution beta
        self._total_steps = 0
        self._history: List[Dict] = []

    def _simulated_reward(self, arm: int, context: Dict) -> float:
        """Simulate a noisy reward for an arm."""
        true_mean = self._TRUE_MEANS[arm % len(self._TRUE_MEANS)]
        # Context-dependent: complex queries benefit more from reranking
        complexity = context.get("query_complexity", 0.5)
        if arm >= 3:  # reranked strategies
            true_mean += 0.05 * complexity
        return float(np.clip(np.random.normal(true_mean, 0.05), 0.0, 1.0))

    def select_arm(self, context: Optional[Dict] = None) -> int:
        """Select an arm using the configured algorithm."""
        context = context or {}

        if self.algorithm == "ucb1":
            return self._ucb1_select()
        elif self.algorithm == "thompson":
            return self._thompson_select()
        else:  # epsilon-greedy
            if random.random() < self.epsilon:
                return random.randint(0, self.n_arms - 1)
            return int(np.argmax(self._values))

    def _ucb1_select(self) -> int:
        """UCB1: select arm with highest upper confidence bound."""
        if self._total_steps < self.n_arms:
            return self._total_steps  # pull each arm once first

        ucb_scores = self._values + np.sqrt(
            2 * np.log(self._total_steps) / (self._counts + 1e-8)
        )
        return int(np.argmax(ucb_scores))

    def _thompson_select(self) -> int:
        """Thompson Sampling: sample from Beta posteriors."""
        samples = np.random.beta(self._alpha, self._beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        """Update arm statistics with observed reward."""
        self._counts[arm] += 1
        self._total_steps += 1

        # Running mean
        n = self._counts[arm]
        self._values[arm] += (reward - self._values[arm]) / n

        # Thompson: treat reward as Bernoulli (threshold at 0.7)
        if reward >= 0.7:
            self._alpha[arm] += 1
        else:
            self._beta[arm] += 1

        self._history.append({
            "step": self._total_steps,
            "arm": arm,
            "strategy": self.STRATEGIES[arm] if arm < len(self.STRATEGIES) else f"arm_{arm}",
            "reward": round(reward, 4),
        })

    def pull(self, context: Optional[Dict] = None) -> Tuple[int, float]:
        """Select arm, observe reward, update. Returns (arm, reward)."""
        arm = self.select_arm(context)
        reward = self.reward_fn(arm, context or {})
        self.update(arm, reward)
        return arm, reward

    def run(self, n_steps: int = 1000, context_fn: Optional[Callable] = None) -> Dict:
        """Run the bandit for n_steps and return statistics."""
        total_reward = 0.0
        for step in range(n_steps):
            context = context_fn(step) if context_fn else {"query_complexity": random.random()}
            _, reward = self.pull(context)
            total_reward += reward

        return self.stats()

    def stats(self) -> Dict:
        """Return current bandit statistics."""
        best_arm = int(np.argmax(self._values))
        return {
            "total_steps": self._total_steps,
            "best_arm": best_arm,
            "best_strategy": self.STRATEGIES[best_arm] if best_arm < len(self.STRATEGIES) else f"arm_{best_arm}",
            "arm_counts": self._counts.tolist(),
            "arm_values": [round(v, 4) for v in self._values.tolist()],
            "cumulative_reward": round(sum(h["reward"] for h in self._history), 3),
            "mean_reward": round(np.mean([h["reward"] for h in self._history]) if self._history else 0.0, 4),
        }

    def ascii_summary(self) -> str:
        """Print arm statistics as ASCII bar chart."""
        lines = [f"Retrieval Bandit ({self.algorithm.upper()}) — {self._total_steps} steps"]
        lines.append("")
        for i, strategy in enumerate(self.STRATEGIES[:self.n_arms]):
            val = self._values[i]
            count = int(self._counts[i])
            bar_len = int(val * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            marker = " ← best" if i == int(np.argmax(self._values)) else ""
            lines.append(f"  {strategy:<20} {bar} {val:.3f} (n={count}){marker}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Chunking Optimization Environment
# ──────────────────────────────────────────────────────────────────────────────

class ChunkingEnv(gym.Env):
    """
    RL environment for optimizing document chunking strategy.

    The agent selects:
      - chunk_size: tokens per chunk (64–1024)
      - overlap: overlap between consecutive chunks (0–50% of chunk_size)
      - split_strategy: sentence | paragraph | fixed | semantic

    Reward:
      - Retrieval precision (simulated: smaller chunks → better precision)
      - Retrieval recall (simulated: larger chunks → better recall)
      - Minus storage cost (more chunks = more storage)

    Observation:
      - avg_doc_length: normalised average document length
      - query_avg_length: normalised average query length
      - current_precision: rolling average precision
      - current_recall: rolling average recall
    """

    metadata = {"render_modes": ["human"]}

    CHUNK_SIZES = [64, 128, 192, 256, 320, 384, 512, 768, 1024]
    OVERLAPS = [0, 0.1, 0.2, 0.3, 0.5]  # fraction of chunk_size
    STRATEGIES = ["fixed", "sentence", "paragraph", "semantic"]

    def __init__(self, max_steps: int = 100):
        super().__init__()
        self.max_steps = max_steps

        n_actions = len(self.CHUNK_SIZES) * len(self.OVERLAPS) * len(self.STRATEGIES)
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )
        self._step = 0
        self._history: List[Dict] = []
        self._doc_length = 0.5
        self._query_length = 0.3

    def _decode_action(self, action: int) -> Tuple[int, float, str]:
        n_strat = len(self.STRATEGIES)
        n_overlap = len(self.OVERLAPS)
        strat_idx = action % n_strat
        action //= n_strat
        overlap_idx = action % n_overlap
        chunk_idx = min(action // n_overlap, len(self.CHUNK_SIZES) - 1)
        return (
            self.CHUNK_SIZES[chunk_idx],
            self.OVERLAPS[overlap_idx],
            self.STRATEGIES[strat_idx],
        )

    def _simulate_reward(self, chunk_size: int, overlap: float, strategy: str) -> Dict:
        rng = np.random.default_rng()

        # Precision: smaller chunks → better (less noise per chunk)
        precision = 0.9 - 0.0005 * chunk_size + 0.05 * (strategy == "semantic")
        precision += 0.03 * overlap  # overlap helps precision slightly

        # Recall: larger chunks → better (less chance of splitting relevant content)
        recall = 0.5 + 0.0004 * chunk_size - 0.02 * overlap

        # Storage cost: inversely proportional to chunk size
        n_chunks_factor = 1000 / chunk_size  # relative number of chunks
        storage_cost = 0.01 * n_chunks_factor / 10

        noise = rng.normal(0, 0.03)
        precision = float(np.clip(precision + noise, 0, 1))
        recall = float(np.clip(recall + noise, 0, 1))
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "precision": precision, "recall": recall,
            "f1": f1, "storage_cost": storage_cost,
        }

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
        self._step = 0
        self._history = []
        self._doc_length = float(np.random.uniform(0.2, 0.9))
        self._query_length = float(np.random.uniform(0.1, 0.5))
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        prec = np.mean([h["precision"] for h in self._history[-5:]]) if self._history else 0.5
        rec = np.mean([h["recall"] for h in self._history[-5:]]) if self._history else 0.5
        return np.array([self._doc_length, self._query_length, prec, rec], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        chunk_size, overlap, strategy = self._decode_action(int(action))
        metrics = self._simulate_reward(chunk_size, overlap, strategy)
        reward = metrics["f1"] - metrics["storage_cost"]

        self._history.append(metrics)
        self._step += 1
        self._doc_length = float(np.random.uniform(0.2, 0.9))

        terminated = self._step >= self.max_steps
        info = {"chunk_size": chunk_size, "overlap": overlap, "strategy": strategy, **metrics}
        return self._get_obs(), float(reward), terminated, False, info


# ──────────────────────────────────────────────────────────────────────────────
# 4. Prompt Selection Bandit
# ──────────────────────────────────────────────────────────────────────────────

class PromptSelectionBandit(RetrievalBandit):
    """
    Multi-armed bandit for selecting the best prompt template.

    Extends RetrievalBandit with prompt-specific arms and context features.
    Useful for A/B testing prompt variants in production.
    """

    def __init__(
        self,
        prompt_templates: List[str],
        algorithm: str = "thompson",
        reward_fn: Optional[Callable] = None,
    ):
        super().__init__(
            n_arms=len(prompt_templates),
            algorithm=algorithm,
            reward_fn=reward_fn or self._prompt_reward,
        )
        self.prompt_templates = prompt_templates
        self.STRATEGIES = [f"template_{i}" for i in range(len(prompt_templates))]
        # Simulated true means for each template
        self._TRUE_MEANS = [
            0.6 + 0.1 * (i / max(len(prompt_templates) - 1, 1))
            for i in range(len(prompt_templates))
        ]

    def _prompt_reward(self, arm: int, context: Dict) -> float:
        true_mean = self._TRUE_MEANS[arm % len(self._TRUE_MEANS)]
        return float(np.clip(np.random.normal(true_mean, 0.08), 0.0, 1.0))

    def best_template(self) -> str:
        best_arm = int(np.argmax(self._values))
        return self.prompt_templates[best_arm]


# ──────────────────────────────────────────────────────────────────────────────
# Training loop utilities
# ──────────────────────────────────────────────────────────────────────────────

def random_agent_baseline(env: gym.Env, n_episodes: int = 10) -> Dict:
    """Evaluate a random agent as a baseline."""
    total_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)

    return {
        "mean_episode_reward": round(float(np.mean(total_rewards)), 4),
        "std_episode_reward": round(float(np.std(total_rewards)), 4),
        "min_episode_reward": round(float(np.min(total_rewards)), 4),
        "max_episode_reward": round(float(np.max(total_rewards)), 4),
        "n_episodes": n_episodes,
    }


def greedy_agent_baseline(env: RAGQualityEnv, n_episodes: int = 10) -> Dict:
    """
    Greedy agent: always selects k=5, rerank=True, chunk_size=256
    (a reasonable hand-tuned default).
    """
    # k=5 → K_OPTIONS index 3; rerank=True → 1; chunk_size=256 → CHUNK_OPTIONS index 3
    k_idx = RAGQualityEnv.K_OPTIONS.index(5)
    chunk_idx = RAGQualityEnv.CHUNK_OPTIONS.index(256)
    n_chunks = len(RAGQualityEnv.CHUNK_OPTIONS)
    n_rerank = 2
    # Encode: action = (k_idx * n_rerank + rerank) * n_chunks + chunk_idx
    action = (k_idx * n_rerank + 1) * n_chunks + chunk_idx

    total_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)

    return {
        "mean_episode_reward": round(float(np.mean(total_rewards)), 4),
        "std_episode_reward": round(float(np.std(total_rewards)), 4),
        "n_episodes": n_episodes,
        "agent": "greedy_k5_rerank_chunk256",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="RL environments for LLMOps")
    sub = parser.add_subparsers(dest="cmd")

    rag_p = sub.add_parser("rag-env", help="Run RAGQualityEnv with random + greedy baselines")
    rag_p.add_argument("--episodes", type=int, default=20)

    bandit_p = sub.add_parser("bandit", help="Run retrieval strategy bandit")
    bandit_p.add_argument("--steps", type=int, default=1000)
    bandit_p.add_argument("--algo", default="ucb1", choices=["ucb1", "thompson", "epsilon_greedy"])

    chunk_p = sub.add_parser("chunking", help="Run ChunkingEnv baseline")
    chunk_p.add_argument("--episodes", type=int, default=10)

    prompt_p = sub.add_parser("prompt-bandit", help="Prompt template selection bandit")
    prompt_p.add_argument("--steps", type=int, default=500)

    args = parser.parse_args()

    if args.cmd == "rag-env":
        print("RAGQualityEnv — random vs greedy baseline")
        env = RAGQualityEnv()

        random_stats = random_agent_baseline(env, n_episodes=args.episodes)
        greedy_stats = greedy_agent_baseline(env, n_episodes=args.episodes)

        print(f"\nRandom agent:  mean_reward={random_stats['mean_episode_reward']:.3f} "
              f"± {random_stats['std_episode_reward']:.3f}")
        print(f"Greedy agent:  mean_reward={greedy_stats['mean_episode_reward']:.3f} "
              f"± {greedy_stats['std_episode_reward']:.3f}")
        print("\nTip: train with PPO/DQN from stable-baselines3:")
        print("  from stable_baselines3 import PPO")
        print("  model = PPO('MlpPolicy', RAGQualityEnv(), verbose=1)")
        print("  model.learn(total_timesteps=50_000)")

    elif args.cmd == "bandit":
        print(f"Retrieval Strategy Bandit ({args.algo.upper()}) — {args.steps} steps")
        bandit = RetrievalBandit(algorithm=args.algo)
        stats = bandit.run(n_steps=args.steps)
        print(bandit.ascii_summary())
        print(f"\nFinal stats: {json.dumps({k: v for k, v in stats.items() if k != 'arm_counts'}, indent=2)}")

    elif args.cmd == "chunking":
        print("ChunkingEnv — random baseline")
        env = ChunkingEnv()
        stats = random_agent_baseline(env, n_episodes=args.episodes)
        print(json.dumps(stats, indent=2))

    elif args.cmd == "prompt-bandit":
        templates = [
            "Answer the question based on the context: {context}\nQuestion: {question}",
            "Context: {context}\n\nBased on the above, answer: {question}",
            "You are a helpful assistant. Use the following context to answer.\nContext: {context}\nQ: {question}\nA:",
            "Given this information: {context}\nPlease answer: {question}",
        ]
        print(f"Prompt Selection Bandit (Thompson) — {args.steps} steps")
        bandit = PromptSelectionBandit(templates, algorithm="thompson")
        bandit.run(n_steps=args.steps)
        print(bandit.ascii_summary())
        print(f"\nBest template: {bandit.best_template()[:80]}...")

    else:
        parser.print_help()
