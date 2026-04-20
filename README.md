# lqr-simplified

`lqr-simplified` is the current research repo for LLQR-style geometry-aware optimization experiments in JAX/Flax.
It contains the main training entrypoint, config-driven experiment definitions, the practical relaxed LLQR preconditioner path, and exact or toy reference paths.

## Main entrypoints

- `run.py`: main Hydra-driven experiment runner and the authoritative runtime path
- `run_single_layer_test.py`: toy analytical script for LLQR-style validation; audit before relying on it as a current regression path

## Config structure

Experiments are composed from `configs/`.
The main user-facing group is `configs/experiment/`, which selects a dataset, architecture, and schedule bundle.
Representative experiments currently include ResNet/CIFAR, ResNet/ImageNet, GPT/WikiText-103, grokking-style transformer runs, and the CIFAR architecture-support presets `vgg16bn-cifar10`, `vgg16bn-cifar100`, `wide-resnet28x10-cifar10`, and `wide-resnet28x10-cifar100`.

A typical Hydra invocation shape is:

```bash
python run.py experiment=resnet18-cifar10
```

## CIFAR architecture surfaces

The current dedicated CIFAR architecture additions are:

- `vgg16-bn`, implemented in `lqr_optimizer/_src/models/vgg.py`
- `wide-resnet-28-10`, implemented in `lqr_optimizer/_src/models/wide_resnet.py`

The maintained Friendly-SAM-aligned CIFAR presets for those models are:

- `vgg16bn-cifar10`
- `vgg16bn-cifar100`
- `wide-resnet28x10-cifar10`
- `wide-resnet28x10-cifar100`

For the bounded local smoke posture and the exact direct-preset `run.py`
commands used in this workspace, use:

- `../tmp/benchmarks/llqr-vgg16bn-wrn28x10-architecture-smokes/README.md`
- `../docs/plans/completed/llqr-vgg16bn-wrn28x10-architecture-support-exec-plan.md`
- `../docs/reports/llqr-vgg16bn-wrn28x10-architecture-support-final-report-2026-04-18.md`

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
- `sam_mode`: perturbation source selector; current supported values are `null`, `base_sam`, `base_fsam`, `past_fsam`, and `asam`
- `perturbation_rho`: perturbation magnitude
- `asam_eta`: canonical ASAM stability offset on non-bias parameters; default `0.01`
- `perturb_mode`: perturbation geometry selector
- `norm_mode`: perturbation normalization selector; current supported values are `euclidean`, `matrix_norm`, `layer_matrix_norm`, and `layer_euclidean`

Current runtime semantics:
- `base_sam` perturbs from the current gradient and leaves `gbar` / `g_last` untouched
- `base_fsam` perturbs from `g_current - gbar`
- `past_fsam` preserves the rolling-buffer variant used before the rename
- `asam` applies the canonical ASAM perturbation from the current gradient using element-wise non-bias parameter scaling and leaves `gbar` / `g_last` untouched
- canonical `asam` requires the neutral legacy defaults for `perturb_mode`, `norm_mode`, `sam_research_*`, `gbar_beta`, and `gbar_eps`; those knobs remain part of the legacy SAM / Friendly-SAM surface rather than the ASAM contract
- `run.py` now delegates mode-specific train-step orchestration to `lqr_optimizer/_src/utils/sam_mode_handlers.py`, while `lqr_optimizer/_src/utils/utils.py` keeps the generic perturbation, canonical ASAM, and buffer helpers

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
- `lqr_optimizer/_src/utils/sam_mode_handlers.py`: SAM-family train-step dispatcher for `null`, `base_sam`, `base_fsam`, `past_fsam`, and `asam`
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
- `../docs/plans/completed/llqr-vgg16bn-wrn28x10-architecture-support-exec-plan.md`
- `../docs/reports/llqr-vgg16bn-wrn28x10-architecture-support-final-report-2026-04-18.md`
- `../tmp/benchmarks/llqr-vgg16bn-wrn28x10-architecture-smokes/README.md`
