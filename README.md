> [!NOTE]
> **Divine Whisper lineage repository.** This is an experimental stage. See the [ecosystem index](https://github.com/aruintelligence/divine-whisper-ecosystem) for the version map and research scope.

# Divine Whisper v6: μ-field simulation and TD(λ)

This repository contains two small, reproducible experiments:

1. A **synthetic** text/sensory vector stream, a stateful μ vector, inverse-entropy concentration (called `coherence` here), scripted sparse pulses, and a mock recommendation rule. A linear TD(λ) learner receives a reward of 1 at the end of this single scripted episode. Its reward is independent of the recommendation and the concentration proxy.
2. A **five-state random walk** with known values, used to compare online TD(0) and TD(λ) on the same sampled episodes. The unbiased walk pays 0 on the left boundary and 1 on the right; the true value for state `i` is `i/6`. Lower root mean squared error (RMSE) means closer predictions on this specific toy task.

The original v6 script referenced undefined components and labeled a vector addition `update_td_lambda` even though it ignored λ. This version uses actual accumulating eligibility traces for a linear value function:

```text
δ = reward + γ V(next) - V(current), with V(terminal) = 0
e = γ λ e + features(current)
weights = weights + α δ e
```

The trace resets at the end of each episode. At λ=0, only the current feature receives each update. The walk benchmark uses the same seed, episodes, features, learning rate, and discount for the two methods. It reports every replicate's final error and mean learning curves; it does not choose the better run after the fact.

## Run

```bash
python -m pip install numpy
python divine_whisper_v6_multimodal_mu_tdlambda.py --seed 42 --steps 12 --lambda 0.9 --episodes 200 --replicates 8 --out-dir artifacts
python -m unittest discover -s tests -v
```

The command writes:

| File | Contents |
| --- | --- |
| `artifacts/coherence_trajectory.json` | Every synthetic step, proxies, delayed reward, TD error, eligibility norm, and final weights |
| `artifacts/heaven_ledger.jsonl` | One JSON record per synthetic step |
| `artifacts/random_walk_comparison.json` | Paired TD(0) and TD(λ) RMSE curves, per-replicate outcomes, true values, parameters |

Running the same command with the same seed and parameters overwrites these files with the same content. To inspect sensitivity, vary `--lambda`, `--seed`, `--episodes`, and `--replicates`. An isolated λ result can fluctuate; compare across seeds and learning rates before inferring anything general.

## Interpretation

The benchmark tests temporal credit assignment on a known Markov chain. The vector simulation does **not** measure consciousness, safety, real multimodal understanding, phase transitions, or a general-purpose agent. Its `clarity`, `safety_proxy`, `budget_proxy`, and `coherence` fields are formulas over generated numbers. The sparse "anchor" is a scripted pulse, not verified evidence. An improvement in the random-walk RMSE is evidence of value prediction on that walk, not a breakthrough in the broader claims.

Original concept and authors: Daniel Jacob Read IV and Shane Travis Horman. MIT licensed.
