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

## SAM surface

The current public SAM configuration surface is:
- `sam_mode`: perturbation source selector; current supported values are `null`, `base_sam`, `base_fsam`, and `past_fsam`
- `perturbation_rho`: perturbation magnitude
- `perturb_mode`: perturbation geometry selector

Current runtime semantics:
- `base_sam` perturbs from the current gradient and leaves `gbar` / `g_last` untouched
- `base_fsam` perturbs from `g_current - gbar`
- `past_fsam` preserves the rolling-buffer variant used before the rename

For the durable benchmark trail, bounded plain-optimizer comparison surface, and
final closure rationale, use the workspace notes:
- `../tmp/benchmarks/llqr-base-sam-wave3-comparison/README.md`
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
