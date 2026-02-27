# WeightScope

Predict Transformer checkpoint quality from attention-weight invariants.

WeightScope creates a bank of actual tiny Transformer checkpoints and trains a neural predictor of held-out classification accuracy directly from attention weights. It uses an invariant descriptor and a shared per-head encoder, with training-run-aware data splits.

## Status and provenance

Version 0.1.0 is an original, AI-assisted implementation developed for **Divyansh Gupta**. This repository was written independently: no upstream source files or Git history were imported. Its scientific inspiration is [Transformer-NFN](https://arxiv.org/abs/2410.04209); the research ideas remain credited to that paper's authors. The implementation choices and limitations below describe this repository's scope. It is an experimental prototype, not a claimed reproduction of published results.

## Setup

Python 3.10 or newer:

```bash
python -m venv .venv
# Linux/macOS; on Windows use .venv\Scripts\activate
source .venv/bin/activate
# Install CPU-only PyTorch first (omit this line to choose another build).
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

## Run

```bash
weightscope bank --count 20 --steps 20 --seed 11 --output runs/bank.npz
weightscope train --data runs/bank.npz --epochs 50 --seed 11 --output runs/train
# Evaluate a separately generated bank; train/test results are already in runs/train/test.json.
weightscope bank --count 10 --steps 20 --seed 101 --output runs/new-bank.npz
weightscope evaluate --data runs/new-bank.npz --checkpoint runs/train/best.pt --output runs/new-evaluation.json
```

Use fresh output paths for new experiments. Training artifacts and generated data are ignored by Git. For independent evaluation, use a different generation seed or a separate dataset; evaluating on the training data is not an unbiased test.

## Implementation

For each attention head it computes `Q K^T` and `V O`. These products are unchanged by invertible basis transformations

```text
Q -> Q A,    K -> K A^(-T)
V -> V B,    O -> B^(-1) O
```

A shared MLP encodes each head, followed by mean pooling across heads and a scalar readout. This makes predictions invariant to head permutations and the two basis changes above. Feature normalization is fitted on training runs only.

The checkpoint bank uses a position-free, single-layer attention model with residuals, normalization, a feed-forward block, and a classifier. It learns whether the mean of six token IDs exceeds 5.5. The bank stores first/final snapshots from multiple initializations and learning rates. Snapshots sharing a `run_id` always stay in the same train, validation, or test partition. Model selection uses validation MAE; the test partition is scored after selection.

## Data and artifacts

Training writes `config.json`, `split.json` with indices and independent run IDs, `metrics.json`, `best.pt`, and `test.json`. Reports include a constant training-mean baseline. The generated NPZ schema is `q,k,v: [checkpoints,heads,width,head_width]`, `o: [checkpoints,heads,head_width,width]`, `accuracy: [checkpoints]` in `[0,1]`, and `run_id: [checkpoints]`. External banks must use compatible tensor dimensions. All commands run on CPU.

## Validation

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

The test suite covers: Attention-output and regressor invariance under nonorthogonal basis changes and head permutations; group-disjoint partitions; real checkpoint generation and regressor optimization; full bank/train/evaluate CLI workflow.

CI runs the suite on Python 3.10 and 3.12. Small synthetic runs validate software behavior; they do not establish a research result. See [VALIDATION.md](VALIDATION.md) for the release checks actually performed.

## Limitations

This is not the paper's full Transformer-NFN architecture or its full symmetry group. The descriptor ignores embedding, normalization, feed-forward, and classifier parameters, so different-performing models can have identical descriptors. The tiny generated bank is not evidence of generalization to pretrained LLMs. Group splitting prevents snapshot leakage, but all generated models share a task and data distribution; broader claims require independent tasks and real checkpoint banks.

## Preparing independent run groups

Use a one-dimensional numeric or string `run_id` array. Every snapshot from the same training run must carry the same ID. At least five distinct IDs are needed for training, validation, and test partitions.

```python
from weightscope.model import group_split

train, validation, test = group_split(
    ["run-a", "run-a", "run-b", "run-c", "run-d", "run-e"], seed=11
)
```

The two snapshots from `run-a` stay together. The returned arrays contain row indices, not IDs. NaN and infinite IDs are rejected because missing identifiers could otherwise leave samples out of every partition. Nested arrays and object-valued IDs are unsupported; encode structured experiment identifiers as stable strings first.

## License

The original code in this repository is MIT licensed, copyright 2026 Divyansh Gupta. See [LICENSE](LICENSE). The cited paper is a separate work; this code's license does not relicense the paper or external datasets.


## Reproducibility and report utilities

The standard-library artifact CLI can inspect saved experiments without loading a model. The following commands use a small illustrative fixture, not benchmark results:

```bash
python -m weightscope.artifact_cli manifest --inputs examples/metrics-sample.json --base . --metadata '{"purpose":"example"}' --output runs/manifest.json
python -m weightscope.artifact_cli verify --input runs/manifest.json --base . --output runs/verification.json
python -m weightscope.artifact_cli summary --input examples/metrics-sample.json --output runs/summary.json
python -m weightscope.artifact_cli csv --input examples/metrics-sample.json --output runs/metrics.csv
python -m weightscope.artifact_cli compare --before examples/metrics-sample.json --after examples/metrics-sample.json --output runs/comparison.json
python -m weightscope.artifact_cli bootstrap --input examples/metrics-sample.json --key value --samples 1000 --seed 7 --output runs/interval.json
```

The installed `weightscope-artifacts` command offers the same interface. Use fresh output filenames: existing artifacts are never overwritten. Manifests store SHA-256 hashes and paths relative to `--base`; keep input files fixed while creating or verifying a manifest. Verification exits with status 1 for changed or missing inputs, and status 2 for invalid arguments or manifests.

Metric summaries accept JSON arrays, single objects, reports containing `rows` or `episodes`, and streamed JSONL. They summarize numeric fields with population standard deviation and ignore boolean/nonnumeric fields. CSV export accepts flat scalar records; nested data must be transformed first. Comparisons report signed changes, without assuming whether larger values are better. Relative change is null when the baseline is zero.

Bootstrap intervals assume independent observations. Correlated training steps or snapshots from the same run are not independent experimental repetitions; aggregate by run before interpreting an interval. These intervals do not calibrate model uncertainty.


## Predictor diagnostics and tuning

Use `--learning-rate` and `--weight-decay` for controlled regressor experiments. Defaults remain 0.002 and 0.01. Keep group splits and the training bank fixed when comparing these settings, and reserve final evaluation data for the final model.

Evaluation includes MAE, RMSE, signed bias (prediction minus target), and Spearman rank correlation using average tied ranks. Spearman is null when either ranking is constant. The `predictions` array records each checkpoint index, run ID, prediction, target, and signed error so failures can be inspected without reconstructing row order. These diagnostics do not change the original release's underperforming benchmark results.


## Experiment analysis

`weightscope-analysis` (or `python -m weightscope.analysis_cli`) provides distribution reports, matched-run comparisons, grouped summaries, and training-curve diagnostics. Reports include input-file SHA-256 fingerprints. See [ANALYSIS.md](ANALYSIS.md) for runnable examples, the Python API, statistical assumptions, and interpretation limits. These tools analyze saved observations; they do not constitute new model benchmark results.
