# Validation record

Validated locally on 2026-09-28 with Python 3.13, CPU PyTorch 2.10.0 and the exact dependencies in `requirements-tested.txt`.

- 24 tests passed: seeded solvability, key/door rules, observation isolation, cue disappearance, seed split boundaries, terminal/truncation behavior, online-vs-sequence equivalence for four architectures, recurrent temporal gradients, a real PPO update, GAE numerics, memory interventions, clone isolation, replay of the measured causal example, human CLI escape, and dashboard interaction.
- Procedural oracle validation covered 120 generated dungeons across three sizes/key counts, plus controlled detours from 2 to 30 edges.
- Trained all three core architectures for 100 procedural PPO updates and 800 showcase PPO updates. LSTM received a short training smoke check, not a performance study.
- Evaluated procedural validation/test seeds, five generalization scenarios, six showcase delay lengths, eight paired intervention settings, and 30 cue-only counterfactual pairs. See `RESULTS.md` for actual outcomes.
- Verified the live Streamlit page in the browser, including a successful run and the recorded control-success/treatment-failure replay. Automated dashboard tests exercised step, reset, new seed, and fork/compare controls.
- All CLI entry points accepted `--help`; every experiment entry point was executed. Dependency installation was checked with pip's dry-run resolver against the installed environment.
- `compileall` and `git diff --check` passed. Model weights, caches, secrets and temporary files are excluded from Git.

The 4,000-update full configuration is provided but was not run. CUDA and operating systems other than this Windows host were not validated. CI is configured to run tests on Linux, but its outcome is reported separately after a push. Virtual-environment activation instructions are standard setup instructions; no claim is made that every shell/OS activation variant was executed here.
