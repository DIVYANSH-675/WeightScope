# Release validation

Executed locally on 2026-11-17 on Linux CPU.

Environment: Python 3.12.3; pytest 9.2.0; Ruff 0.17.5; PyTorch 2.15.0+cpu; NumPy 2.5.5.

- Editable installation succeeded.
- Ruff lint and format checks passed.
- **4 tests passed**, including a command-line workflow test that creates its own data/artifacts.
- Wheel and source-distribution builds succeeded.
- The experiments below ran successfully; raw evaluation output is preserved in [examples/release-report.json](examples/release-report.json).

## Exploratory run

Generated 20 independent training runs with 20 optimization steps each, seed 11, retaining first/final snapshots. The regressor trained for 50 epochs, selecting by validation MAE. Whole-run partitions contained 24/8/8 snapshots.

| Evaluation | Predictor MAE | Training-mean baseline MAE |
|---|---:|---:|
| Reserved run groups | 0.194925 | 0.106120 |
| New 10-run bank, seed 101 | 0.181428 | 0.150911 |

**The predictor did not beat the constant baseline.** This release validates the invariant architecture and evaluation pipeline, not a useful quality-prediction result. More diverse training data and descriptors that account for the omitted parameters are needed before stronger claims are justified.

## Reproduction and scope

Use the README commands, matching the seeds and sizes above. AstroBridge's final evaluation used seed 202. Exact floating-point values can differ across dependency versions and hardware. Generated training checkpoints remain local under ignored `runs/` directories; they are not bundled in the repository or Python distribution.

The CI workflow validates Python 3.10 and 3.12 independently; its current result is visible in [GitHub Actions](https://github.com/DIVYANSH-675/WeightScope/actions). No GPU benchmark, large external dataset, or paper-result reproduction was run. There is no claim that these development experiments establish general research superiority.
