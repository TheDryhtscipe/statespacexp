# statespacexp

statespacexp is a research codebase. It asks which properties of a recurrent model's state transition let it learn state tracking that generalises to sequences longer than those it was trained on. The properties under test are input dependence, eigenvalue range (non-negative, signed, or complex), and diagonal versus non-diagonal structure.

The work is in progress. The code, the experiment protocol, and the results change as the study runs.

## What is in the package

The Python package is `sst`, in `src/sst/`. Each module's docstring describes its conventions in full.

| Module | Contents |
|---|---|
| `sst.groups` | Finite groups (Z_n, S_n, A_n) as explicit Cayley tables |
| `sst.tasks` | State-tracking tasks with a label at every position: `parity`, `z3`, `s3`, `a5`, `s5`, `flipflop`, and `flipflop_sparse` |
| `sst.reference` | Reference predictors for the tasks |
| `sst.evaluation` | Seeded random streams and length-bucketed evaluation (by default, buckets up to 8 times the training length) |
| `sst.models` | One backbone with interchangeable sequence mixers |
| `sst.config` | `RunConfig` and its stable `run_id` |
| `sst.train` | The training runner, with optional CUDA graphs |
| `sst.results` | Run records, one JSON line per run in `runs.jsonl` |
| `sst.queue` | A directory job queue for one shared GPU |
| `sst.plan` | Learning-rate sweep plans and seed plans |
| `sst.analysis` | Learning-rate selection and per-cell outcome categories |
| `sst.overshoot` | Overshoot ratio of trained TTT layers (inner step size times inner-loss curvature) |
| `sst.bench` | Training throughput per mixer |

### Sequence mixers

Every model shares the same embedding, pre-norm residual blocks, MLP, and readout. Only the sequence mixer changes.

| Mixer | Transition |
|---|---|
| `lin_attn` | identity (linear attention) |
| `diag_fixed` | learned fixed diagonal |
| `diag_pos` | input-dependent diagonal gate in (0, 1) |
| `diag_neg` | input-dependent diagonal in (-1, 1) |
| `diag_complex` | input-dependent rotation and scaling per pair of channels |
| `delta` | delta rule, I − β k kᵀ with β in (0, 1) |
| `delta_neg` | delta rule with β in (0, 2) |
| `deltaproduct` | a product of several generalised Householder steps per token |
| `ttt` | test-time training layer, with a linear or MLP inner model |
| `s4d` | S4D, a complex diagonal linear time-invariant model |
| `transformer` | causal softmax attention, with RoPE or without position encoding |
| `lstm` | LSTM, as a nonlinear control |

The models are written in plain PyTorch. They use no optimised kernel libraries.

## Set up

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). Training is practical only on a CUDA GPU.

1. Clone the repository.
2. Install the dependencies:

   ```bash
   uv sync
   ```

3. Run the fast tests:

   ```bash
   uv run pytest -m "not slow"
   ```

4. Run the type check:

   ```bash
   uv run mypy src tests
   ```

## Run experiments

Results go to `$SST_RESULTS_DIR`. If that variable is not set, they go to `results/` in the main checkout.

1. Plan a learning-rate sweep and queue it:

   ```bash
   uv run python -m sst.plan sweep --families delta,lstm --tasks s5,flipflop --depths 1,2 --steps 20000
   ```

   To see the plan without queueing it, add `--dry-run`.

2. Start a worker. It runs queued jobs one at a time on the GPU:

   ```bash
   uv run python -m sst.queue work
   ```

3. When a cell's sweep is complete, queue the remaining seeds at the chosen learning rate:

   ```bash
   uv run python -m sst.plan advance --n-seeds 5
   ```

4. Print the outcome category of each cell:

   ```bash
   uv run python -m sst.analysis
   ```

To see how many jobs are in each state, run `uv run python -m sst.queue status`.

## Licence

[PLACEHOLDER: licence not chosen yet]
