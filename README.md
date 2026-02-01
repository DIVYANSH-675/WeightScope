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

## License

The original code in this repository is MIT licensed, copyright 2026 Divyansh Gupta. See [LICENSE](LICENSE). The cited paper is a separate work; this code's license does not relicense the paper or external datasets.
