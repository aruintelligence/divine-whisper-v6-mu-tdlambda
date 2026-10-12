"""Reproducible toy μ-field simulation and a genuine linear TD(λ) experiment.

The μ-field, "clarity", and recommendation policy are synthetic proxies. Only
the delayed-reward benchmark has a known target value function.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class MuField:
    """Stateful vector with an inverse-entropy concentration proxy."""

    vector: np.ndarray
    coherence: float = 0.0
    entropy: float = 0.0

    @classmethod
    def new(cls, dim: int = 256) -> MuField:
        if dim < 2:
            raise ValueError("dim must be at least 2")
        return cls(np.zeros(dim, dtype=np.float64), entropy=math.log(dim))

    def update(self, delta: np.ndarray, alpha: float) -> None:
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape != self.vector.shape or not np.all(np.isfinite(delta)):
            raise ValueError("delta must be finite and match the field shape")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        self.vector += alpha * delta
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector /= norm
        probabilities = np.abs(self.vector) + 1e-12
        probabilities /= probabilities.sum()
        self.entropy = float(-np.sum(probabilities * np.log(probabilities)))
        self.coherence = clamp(1.0 - self.entropy / math.log(self.vector.size))


@dataclass
class TDLambdaValue:
    """Online linear TD(λ) with accumulating eligibility traces.

    δ = r + γ V(next) - V(current), with V(terminal) = 0.
    e = γ λ e + features(current); weights += α δ e.
    Traces reset at episode boundaries.
    """

    n_features: int
    alpha: float = 0.1
    gamma: float = 1.0
    lambda_: float = 0.9
    weights: np.ndarray = field(init=False)
    eligibility: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.n_features < 1 or self.alpha <= 0 or not 0 <= self.gamma <= 1 or not 0 <= self.lambda_ <= 1:
            raise ValueError("invalid TD(λ) parameters")
        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.eligibility = np.zeros(self.n_features, dtype=np.float64)

    def features(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.n_features,) or not np.all(np.isfinite(values)):
            raise ValueError("features must be a finite vector of n_features")
        return values

    def predict(self, values: np.ndarray) -> float:
        return float(self.weights @ self.features(values))

    def update(
        self,
        current: np.ndarray,
        reward: float,
        next_state: np.ndarray | None,
    ) -> dict[str, float]:
        x = self.features(current)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        predicted = self.predict(x)
        next_value = 0.0 if next_state is None else self.predict(next_state)
        error = reward + self.gamma * next_value - predicted
        self.eligibility = self.gamma * self.lambda_ * self.eligibility + x
        trace_norm = float(np.linalg.norm(self.eligibility))
        self.weights += self.alpha * error * self.eligibility
        if next_state is None:
            self.eligibility.fill(0)
        return {"prediction_before": predicted, "td_error": error, "trace_norm": trace_norm}


def clarity(field: MuField, distortion: float, friction: float) -> float:
    raw = field.coherence / ((distortion + 1e-6) ** 2 * (friction + 1e-6))
    if distortion < 0.14:
        raw *= 1.0 + 1.8 * ((0.14 - distortion) / 0.14) ** 2
    return clamp(raw)


class Remiel:
    """A seeded mock recommendation rule; its score is not a safety evaluation."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def run(self, field: MuField, clarity_score: float, safety_proxy: float) -> tuple[str, float]:
        score = clamp(clarity_score * safety_proxy)
        if score > 0.7:
            return "continue", score
        if score > 0.4:
            recommendation, scale = "stabilize", 0.005
        else:
            recommendation, scale = "pause_and_review", 0.01
        correction = self.rng.normal(0, scale, field.vector.shape)
        field.update(correction, alpha=0.005)
        return recommendation, score


@dataclass
class EarlyExitEvaluator:
    """Stop after three low-clarity steps or an exhausted synthetic budget."""

    low_steps: int = 0

    def should_halt(self, clarity_score: float, budget: float) -> bool:
        self.low_steps = self.low_steps + 1 if clarity_score < 0.1 else 0
        return self.low_steps >= 3 or budget <= 0.05


def run_demo(seed: int, max_steps: int, lambda_: float, output_dir: Path) -> dict:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    field = MuField.new()
    remiel = Remiel(rng)
    exit_rule = EarlyExitEvaluator()
    distortion, friction = 0.45, 0.35
    steps: list[dict] = []
    stopped_early = False

    for index in range(max_steps):
        text_delta = rng.normal(0, 0.008, field.vector.shape)
        sensory_delta = rng.normal(0, 0.005, field.vector.shape)
        field.update(0.6 * text_delta + 0.4 * sensory_delta, alpha=0.008)

        # A scripted sparse pulse; no external evidence is retrieved or verified.
        anchored = index > 0 and index % 4 == 0
        if anchored:
            pulse = np.zeros_like(field.vector)
            pulse[:8] = 0.2
            field.update(pulse, alpha=0.3)

        clarity_score = clarity(field, distortion, friction)
        safety_proxy = clamp(1.0 - 0.5 * distortion - 0.25 * field.entropy / math.log(field.vector.size))
        budget = clamp(1.0 - 0.04 * index - 0.5 * distortion)
        recommendation, score = remiel.run(field, clarity_score, safety_proxy)
        steps.append({
            "step": index,
            "coherence": field.coherence,
            "entropy": field.entropy,
            "clarity_proxy": clarity_score,
            "safety_proxy": safety_proxy,
            "budget_proxy": budget,
            "distortion": distortion,
            "friction": friction,
            "anchored": anchored,
            "recommendation": recommendation,
            "recommendation_score": score,
        })
        if exit_rule.should_halt(clarity_score, budget):
            stopped_early = True
            break
        distortion *= 0.96
        friction *= 0.98

    # This is a single scripted episode with a terminal reward of 1, regardless
    # of the recommendation or coherence. It demonstrates delayed credit only.
    learner = TDLambdaValue(n_features=4, alpha=0.1, lambda_=lambda_)
    features = [
        np.array([1.0, row["coherence"], row["distortion"], row["friction"]])
        for row in steps
    ]
    for index, row in enumerate(steps):
        terminal = index == len(steps) - 1
        row["reward"] = 1.0 if terminal else 0.0
        row["td"] = learner.update(
            features[index], row["reward"], None if terminal else features[index + 1]
        )

    report = {
        "scope": "synthetic vectors, proxy metrics, scripted terminal reward",
        "seed": seed,
        "lambda": lambda_,
        "requested_steps": max_steps,
        "stopped_early": stopped_early,
        "final_coherence": field.coherence,
        "value_weights": learner.weights.tolist(),
        "coherence_history": [row["coherence"] for row in steps],
        "steps": steps,
    }
    (output_dir / "coherence_trajectory.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    # Overwrite the run's ledger on rerun, so identical seeds produce identical files.
    (output_dir / "heaven_ledger.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in steps),
        encoding="utf-8",
    )
    return report


def random_walk_episode(rng: np.random.Generator, n_states: int = 5) -> list[tuple[int, int | None, float]]:
    """Unbiased walk from the center; left terminal reward 0, right reward 1."""
    state = (n_states + 1) // 2
    transitions = []
    while True:
        next_position = state + int(rng.choice((-1, 1)))
        terminal = next_position in (0, n_states + 1)
        reward = float(next_position == n_states + 1)
        transitions.append((state, None if terminal else next_position, reward))
        if terminal:
            return transitions
        state = next_position


def benchmark(seed: int, episodes: int = 200, replicates: int = 8, lambda_: float = 0.9) -> dict:
    """Paired λ=0 and λ=lambda_ evaluations on identical sampled walks."""
    if episodes < 1 or replicates < 1 or not 0 <= lambda_ <= 1:
        raise ValueError("episodes and replicates must be positive; lambda in [0, 1]")
    n_states = 5
    targets = np.arange(1, n_states + 1, dtype=np.float64) / (n_states + 1)
    basis = np.eye(n_states)
    methods = {"td0": 0.0, "td_lambda": lambda_}
    curves = {name: [] for name in methods}

    for replicate in range(replicates):
        rng = np.random.default_rng(seed + replicate)
        # Materialize each episode once so both methods see the exact same data.
        walks = [random_walk_episode(rng, n_states) for _ in range(episodes)]
        for name, trace_decay in methods.items():
            learner = TDLambdaValue(n_features=n_states, alpha=0.1, lambda_=trace_decay)
            rmse = [float(np.sqrt(np.mean((learner.weights - targets) ** 2)))]
            for walk in walks:
                for state, next_state, reward in walk:
                    learner.update(basis[state - 1], reward, None if next_state is None else basis[next_state - 1])
                rmse.append(float(np.sqrt(np.mean((learner.weights - targets) ** 2))))
            curves[name].append(rmse)

    return {
        "task": "five-state unbiased random walk with absorbing endpoints",
        "true_values": targets.tolist(),
        "reward": "1 at right terminal; 0 otherwise",
        "alpha": 0.1,
        "gamma": 1.0,
        "seed": seed,
        "episodes_per_replicate": episodes,
        "replicates": replicates,
        "sample_at_episode": sorted(set([0, episodes] + list(range(10, episodes, 10)))),
        "methods": {
            name: {
                "lambda": methods[name],
                "final_rmse_by_replicate": [run[-1] for run in curves[name]],
                "mean_rmse_at_sample": [
                    float(np.mean([run[i] for run in curves[name]]))
                    for i in sorted(set([0, episodes] + list(range(10, episodes, 10))))
                ],
            }
            for name in methods
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.9)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if not 0 <= args.lambda_ <= 1:
        parser.error("--lambda must be in [0, 1]")
    if args.steps < 1 or args.episodes < 1 or args.replicates < 1:
        parser.error("--steps, --episodes, and --replicates must be positive")
    demo = run_demo(args.seed, args.steps, args.lambda_, args.out_dir)
    comparison = benchmark(args.seed, args.episodes, args.replicates, args.lambda_)
    (args.out_dir / "random_walk_comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "steps_completed": len(demo["steps"]),
        "final_coherence_proxy": demo["final_coherence"],
        "mean_final_rmse": {
            name: float(np.mean(method["final_rmse_by_replicate"]))
            for name, method in comparison["methods"].items()
        },
        "output_dir": str(args.out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
