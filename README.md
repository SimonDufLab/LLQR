# lqr-simplified

`lqr-simplified` is the current research repo for LLQR-style geometry-aware optimization experiments in JAX/Flax.
It contains the main training entrypoint, config-driven experiment definitions, the practical relaxed LLQR preconditioner path, and exact or toy reference paths.

## Main entrypoints

- `run.py`: main Hydra-driven experiment runner and the authoritative runtime path
- `run_single_layer_test.py`: toy analytical script for LLQR-style validation; audit before relying on it as a current regression path

## Config structure

Experiments are composed from `configs/`.
The main user-facing group is `configs/experiment/`, which selects a dataset, architecture, and schedule bundle.
Representative experiments currently include ResNet/CIFAR, ResNet/ImageNet, GPT/WikiText-103, and grokking-style transformer runs.

A typical Hydra invocation shape is:

```bash
python run.py experiment=resnet18-cifar10
```

## LLQR large-batch route guidance

For ResNet-50/ImageNet runs that need `precond_batch_size=256` on the validated
A100 surface, prefer the exact mixed-term grouped chunked route when it fits:

```bash
python run.py experiment=resnet50-imagenet \
  llqr_batch_update_mode=chunked_lqr_segment \
  llqr_batch_update_chunk_size=128 \
  llqr_use_fast_paths=true
```

Keep the default `llqr_second_order_mode=batched_exact` and
`llqr_second_order_chunk_size=null` for this route. The opt-in
`llqr_second_order_mode=sample_separable_exact` route is exact, but current A100
evidence makes it a memory-safety fallback for eligible grouped LLQR segments,
not the recommended compute path when grouped chunked `batched_exact` already
fits.

## SAM surface

The current public SAM configuration surface is:
- `sam_mode`: perturbation source selector; current supported values are `null`, `base_sam`, `base_fsam`, and `past_fsam`
- `perturbation_rho`: perturbation magnitude
- `perturb_mode`: perturbation geometry selector

Current runtime semantics:
- `base_sam` perturbs from the current gradient and leaves `gbar` / `g_last` untouched
- `base_fsam` perturbs from `g_current - gbar`
- `past_fsam` preserves the rolling-buffer variant used before the rename

There is also a research-only ablation surface:
- `sam_research_base_vector_source`: `current_gradient | main_optimizer_momentum | random_direction`
- `sam_research_perturb_sign`: `ascent | descent`

Current runtime status for those ablation knobs:
- the neutral defaults are `current_gradient` and `ascent`
- non-default settings are allowed only when `sam_mode` is `base_sam` or `base_fsam`
- `main_optimizer_momentum` is read from the actual Optax main-optimizer state, with current support for Polyak-like `TraceState.trace` and Adam-style `ScaleByAdamState.mu`
- plain `sgd` is intentionally unsupported for `main_optimizer_momentum`
- `base_sam` uses the selected source directly, while `base_fsam` uses `selected_source - gbar`
- `random_direction` uses one dedicated post-center-pass RNG split and samples a Gaussian pytree matching the center-gradient leaves
- `sam_research_perturb_sign` is applied to the final perturbation tree, so `descent` is exactly the negated `ascent` perturbation

For the durable benchmark trail, bounded plain-optimizer comparison surface, the
research-only perturbation-source ablation matrix, and final closure rationale,
use the workspace notes:
- `../tmp/benchmarks/llqr-base-sam-wave3-comparison/README.md`
- `../tmp/benchmarks/llqr-sam-perturbation-ablation-wave3-local/README.md`
- `../docs/plans/completed/llqr-sam-perturbation-source-ablation-exec-plan.md`
- `../docs/reports/llqr-sam-perturbation-source-ablation-final-report-2026-04-16.md`
- `../docs/reports/llqr-base-sam-support-final-report-2026-04-15.md`

In this workspace, keep local training benchmarks on `agent-quick-local-test`.
The `resnet18-cifar10` comparison remains an external-only higher-memory follow-up.

## Code layout

- `lqr_optimizer/_src/preconditioner.py`: relaxed LLQR preconditioner logic
- `lqr_optimizer/_src/exact_methods.py`: exact or benchmark-style second-order helpers
- `lqr_optimizer/_src/utils/build_lqr.py`: LQR object construction from model linearization
- `lqr_optimizer/_src/utils/build_lqr_segments.py`: grouped LLQR segment builders used by full-batch and chunked split execution-stage updates
- `lqr_optimizer/_src/models/`: architecture definitions
- `lqr_optimizer/_src/block_matrices_approx/`: structured inverse-preconditioner parameterizations

## Further documentation

Start with the workspace-level docs index:
- `../docs/README.md`

Then use:
- `../docs/lqr-simplified-repo-map.md`
- `../docs/lqr-simplified-change-impact.md`
- `../docs/lqr-simplified-architecture-stage-contract.md`
- `../docs/lqr-simplified-methodology.md`
- `../docs/lqr-simplified-agent-notes.md`
