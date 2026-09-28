# Recorded experiment results

Generated from saved JSON measurements by `python experiments/report.py`. No manually entered scores.

All runs use one optimization seed (42), CPU PyTorch, shared map seed streams, and greedy evaluation. Confidence intervals quantify map sampling only. Showcase and procedural models are trained separately.

## Showcase

Controlled cue retention: 800 PPO updates, 32 episodes/update, detour curriculum 2/4/8/15.

| Agent | Split | Maps | Success | 95% Wilson CI | Mean reward | Steps / success |
| --- | --- | --- | --- | --- | --- | --- |
| no_memory | validation | 100 | 53% | 43.3%–62.5% | 7.12 | 34.0 |
| no_memory | test | 100 | 57% | 47.2%–66.3% | 7.60 | 34.0 |
| history | validation | 100 | 53% | 43.3%–62.5% | 7.12 | 34.0 |
| history | test | 100 | 57% | 47.2%–66.3% | 7.60 | 34.0 |
| gru | validation | 100 | 100% | 96.3%–100.0% | 12.76 | 34.0 |
| gru | test | 100 | 100% | 96.3%–100.0% | 12.76 | 34.0 |

## Comparison

Quick procedural exploration: 100 PPO updates, 16 episodes/update, 5×5 / 8 rooms / 1 key. Undertrained.

| Agent | Split | Maps | Success | 95% Wilson CI | Mean reward | Steps / success |
| --- | --- | --- | --- | --- | --- | --- |
| no_memory | validation | 50 | 2% | 0.4%–10.5% | -1.95 | 7.0 |
| no_memory | test | 50 | 0% | 0.0%–7.1% | -2.06 | — |
| history | validation | 50 | 10% | 4.3%–21.4% | -0.57 | 7.2 |
| history | test | 50 | 8% | 3.2%–18.8% | -0.78 | 6.0 |
| gru | validation | 50 | 0% | 0.0%–7.1% | -1.99 | — |
| gru | test | 50 | 2% | 0.4%–10.5% | -1.77 | 7.0 |

## Paired interventions

Reset timing is a fraction of the control trajectory length. Noise is per-dimension Gaussian standard deviation, applied at 50%. Positive control: zero noise must preserve behavior.

| Intervention | Strength / fraction | Control | Treatment | Success change | Paired 95% bootstrap CI |
| --- | --- | --- | --- | --- | --- |
| reset | 0.25 | 100% | 43% | -57% | -66.0% to -48.0% |
| reset | 0.5 | 100% | 43% | -57% | -66.0% to -48.0% |
| reset | 0.75 | 100% | 43% | -57% | -66.0% to -48.0% |
| noise | 0.0 | 100% | 100% | +0% | +0.0% to +0.0% |
| noise | 0.1 | 100% | 100% | +0% | +0.0% to +0.0% |
| noise | 0.5 | 100% | 98% | -2% | -5.0% to +0.0% |
| noise | 1.0 | 100% | 86% | -14% | -21.0% to -8.0% |
| noise | 2.0 | 100% | 67% | -33% | -42.0% to -24.0% |

Verified replay: seed **200001**, reset after **8** decisions, first action divergence at **34**. Control success **True**; treatment success **False**. This example was selected after inspecting the test pairs to illustrate an effect; the aggregate uses every tested seed.

## Cue counterfactual audit

On **30/30** additional held-out pairs, the GRU solved both versions of the same dungeon when only the original cue and required seal were flipped. It changed its final choice in **30/30** pairs. Navigation, room descriptions and clues were held fixed. These pairs use test seeds starting at 201000, separate from the main 100-map evaluation.

## Procedural generalization

These weak quick-run results do not support a generalization-performance claim.

| Agent | unseen_same_size | 6x6 | 8x8 | more_keys | unreliable_clues |
| --- | --- | --- | --- | --- | --- |
| no_memory | 0% | 0% | 0% | 0% | 0% |
| history | 8% | 2% | 0% | 0% | 8% |
| gru | 2% | 0% | 0% | 0% | 2% |

## Cue-to-decision dependency distance

Longer detours use the same controlled corridor family, not new free-navigation topologies. Every tested delay exceeds the history-4 window, so this sweep does not locate its performance boundary.

| Agent | 7 decisions | 11 decisions | 19 decisions | 33 decisions | 53 decisions | 63 decisions |
| --- | --- | --- | --- | --- | --- | --- |
| no_memory | 57% | 57% | 57% | 57% | 57% | 57% |
| history | 57% | 57% | 57% | 57% | 57% | 57% |
| gru | 100% | 100% | 100% | 100% | 100% | 100% |

## Local checkpoint provenance

Weights are excluded from Git. These hashes identify the locally evaluated checkpoints; retraining under a different software/hardware stack may not reproduce identical bytes.

| Checkpoint | SHA-256 | Bytes |
| --- | --- | --- |
| results/comparison/gru/checkpoint.pt | f4a73c3ea066477908d77c95196d78f503290303482d59aae9d541fe5d65fd15 | 143797 |
| results/comparison/history/checkpoint.pt | dd0f05db29cb59ec33c8248d6018a650986d7e0bd84035eff234695e0eaaed86 | 97399 |
| results/comparison/no_memory/checkpoint.pt | 5ee5d7c1c91af08e50112ede75380b7e205b75b5fac9d1c50f90999e40760406 | 39543 |
| results/showcase/gru/checkpoint.pt | 3587672deb4f95302a7615de6a9e16d112c4ca7ad200cd80d961c11f691cd769 | 143861 |
| results/showcase/history/checkpoint.pt | c0624d767eeacb6c5e93358a92f42f88fa5c1cbfef5d9cf4d7cd4a47bb08938b | 97463 |
| results/showcase/no_memory/checkpoint.pt | c8af3f2167b6c6fae92171482e24934a4d0781d659cd4b9150a1e2745dd819bd | 39607 |
