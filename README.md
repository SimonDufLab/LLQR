# lqr-simplified

`lqr-simplified` is the current research repo for LLQR-style geometry-aware optimization experiments in JAX/Flax.
It contains the main training entrypoint, config-driven experiment definitions, the practical relaxed LLQR preconditioner path, and exact or toy reference paths.

## Main entrypoints

- `run.py`: main Hydra-driven experiment runner and the authoritative runtime path
- `run_single_layer_test.py`: toy analytical script for LLQR-style validation; audit before relying on it as a current regression path

## Config structure

Experiments are composed from `configs/`.
The main user-facing group is `configs/experiment/`, which selects a dataset, architecture, and schedule bundle.
Representative experiments currently include ResNet/CIFAR, ResNet/ImageNet, GPT/WikiText-103, grokking-style transformer runs, and the CIFAR architecture-support presets `vgg16bn-cifar10`, `vgg16bn-cifar100`, `wide-resnet28x10-cifar10`, `wide-resnet28x10-cifar100`, `pyramidnet110-cifar10`, and `pyramidnet110-cifar100`.
The dataset surface now also includes the fairseq-faithful local `iwslt14_de_en` text contract under `configs/dataset/iwslt14_de_en.yaml`; it expects an extracted `iwslt14.tokenized.de-en/` directory, writes a local `.llqr_numeric_cache/`, and maps the current `run.py` held-out eval path to `valid` while keeping `test` cached for later generation work.

A typical Hydra invocation shape is:

```bash
python run.py experiment=resnet18-cifar10
```

## Architecture surfaces

The current dedicated architecture additions are:

- `vgg16-bn`, implemented in `lqr_optimizer/_src/models/vgg.py`
- `wide-resnet-28-10`, implemented in `lqr_optimizer/_src/models/wide_resnet.py`
- `pyramidnet-110`, implemented in `lqr_optimizer/_src/models/pyramidnet.py`
- `transformer-iwslt-de-en`, implemented in `lqr_optimizer/_src/models/transformer_iwslt.py`
- `vit-ti-16`, implemented in `lqr_optimizer/_src/models/vit.py`
- `vit-s-16`, implemented in `lqr_optimizer/_src/models/vit.py`

The translation train-path surface is now intentionally narrow but public:

- routed architecture: `transformer-iwslt-de-en`
- public experiment preset: `transformer-iwslt14-de-en`
- expected loader contract: `dataset.loader=local_seq2seq_text`
- batch contract: `x=(src_tokens, prev_output_tokens)` and `y=target_tokens_flat`
- model defaults: fairseq-style 6 encoder layers, 6 decoder layers, embed dim `512`, FFN dim `1024`, heads `4`, `relu`, sinusoidal positions, and tied decoder input/output embeddings
- public recipe surface: `main_optimizer=adam`, `adam_betas=[0.9, 0.98]`, `lr_scheduler=inverse_sqrt`, `learning_rate=5e-4`, `weight_decay=1e-4`, `label_smoothing=0.1`, `dataset.max_tokens=4096`, and LLQR `full_batch` seq2seq updates
- current boundary: generation, BLEU, and checkpoint-metric wiring still land in later waves, and `llqr_batch_update_mode=chunked_lqr_segment` remains intentionally unsupported for seq2seq translation batches

The maintained Friendly-SAM-aligned CIFAR presets currently exist for:

- `vgg16bn-cifar10`
- `vgg16bn-cifar100`
- `wide-resnet28x10-cifar10`
- `wide-resnet28x10-cifar100`
- `pyramidnet110-cifar10`
- `pyramidnet110-cifar100`

The current public ViT surface is intentionally narrower:

- routed architectures: `vit-ti-16` and `vit-s-16`
- single public preset: `vit-ti16-cifar100-adamw`
- public optimizer addition used by that preset: `main_optimizer=adamw`
- intentional boundary: `vit-s-16` is routed but still has no public preset

For the maintained CIFAR validation posture in this workspace, use:

- `../tmp/benchmarks/llqr-vgg16bn-wrn28x10-architecture-smokes/README.md`
- `../tmp/benchmarks/llqr-pyramidnet110-architecture-smokes/README.md`
- `../tmp/benchmarks/llqr-vit-ti16-vit-s16-architecture-smokes/README.md`
- `../docs/plans/completed/llqr-pyramidnet110-architecture-support-exec-plan.md`
- `../docs/reports/llqr-pyramidnet110-architecture-support-final-report-2026-04-21.md`
- `../docs/plans/completed/llqr-vit-ti16-vit-s16-architecture-support-exec-plan.md`
- `../docs/reports/llqr-vit-ti16-vit-s16-architecture-support-final-report-2026-04-21.md`
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
- `sam_mode`: perturbation source selector; current supported values are `null`, `base_sam`, `base_fsam`, `past_fsam`, `asam`, and `fisher_sam`
- `perturbation_rho`: perturbation magnitude
- `asam_eta`: canonical ASAM stability offset on non-bias parameters; default `0.01`
- `fisher_sam_eta`: canonical Fisher-SAM additive inverse-Fisher diagonal regularizer; default `0.1`
- `sam_use_preconditioner_on_update`: for supported non-null SAM modes, keep the configured LLQR-backed outer update when `true` and force a vanilla outer update when `false`; default `true`
- `perturb_mode`: perturbation geometry selector
- `norm_mode`: perturbation normalization selector; current supported values are `euclidean`, `matrix_norm`, `layer_matrix_norm`, and `layer_euclidean`

Current runtime semantics:
- `sam_mode=null` disables perturbation and treats `sam_use_preconditioner_on_update` as inert
- `base_sam` perturbs from the current gradient and leaves `gbar` / `g_last` untouched
- `base_fsam` perturbs from `g_current - gbar`
- `past_fsam` preserves the rolling-buffer variant used before the rename
- `asam` applies the canonical ASAM perturbation from the current gradient using element-wise non-bias parameter scaling and leaves `gbar` / `g_last` untouched
- `fisher_sam` applies the canonical Fisher-SAM perturbation from the accumulated minibatch gradient using the diagonal Fisher approximation `g^2`, additive inverse-Fisher regularization `fisher_sam_eta`, and a vanilla outer update, and leaves `gbar` / `g_last` untouched
- `base_sam`, `base_fsam`, `past_fsam`, and `asam` follow `sam_use_preconditioner_on_update` for the outer parameter update; a true LLQR perturbation-only ablation still requires a legacy LLQR-backed perturbation mode such as `perturb_mode=ema_precond_grad` or `ema_direction`
- canonical `asam` requires the neutral legacy defaults for `perturb_mode`, `norm_mode`, `sam_research_*`, `gbar_beta`, and `gbar_eps`; those knobs remain part of the legacy SAM / Friendly-SAM surface rather than the ASAM contract
- canonical `fisher_sam` requires the same neutral legacy defaults and intentionally treats `sam_use_preconditioner_on_update` as inert in favor of the vanilla optimizer update
- `run.py` now delegates mode-specific train-step orchestration to `lqr_optimizer/_src/utils/sam_mode_handlers.py`, while `lqr_optimizer/_src/utils/utils.py` keeps the generic perturbation, canonical ASAM, canonical Fisher-SAM, and buffer helpers

The durable benchmark trail for SAM remains intentionally narrower than the full
public mode set. Use the benchmark notes below for timing or comparison claims,
and use the final correctness-only mode matrix note for the exact current smoke
commands across all public modes.

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
- `../tmp/llqr-sam-handler-and-new-modes-2026-04-20/wave_4/README.md`
- `../docs/plans/completed/llqr-sam-handler-and-new-modes-exec-plan.md`
- `../docs/reports/llqr-sam-handler-and-new-modes-final-report-2026-04-20.md`
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
- `lqr_optimizer/_src/utils/sam_mode_handlers.py`: SAM-family train-step dispatcher for `null`, `base_sam`, `base_fsam`, `past_fsam`, `asam`, and `fisher_sam`
- `lqr_optimizer/_src/utils/seq2seq_utils.py`: seq2seq-specific runtime helpers kept separate from generic `utils.py` while the IWSLT14 translation surface is landing
- `lqr_optimizer/_src/utils/dataloaders/iwslt14_de_en.py`: fairseq-faithful local IWSLT14 text loader with separate source and target dictionaries, flat numeric caches, and token-budget training batches
- `lqr_optimizer/_src/models/transformer_iwslt.py`: fairseq-style IWSLT14 German-to-English encoder-decoder Transformer with explicit train/inference variants and LLQR segment metadata
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
- `../docs/plans/completed/llqr-pyramidnet110-architecture-support-exec-plan.md`
- `../docs/reports/llqr-pyramidnet110-architecture-support-final-report-2026-04-21.md`
- `../docs/plans/completed/llqr-vit-ti16-vit-s16-architecture-support-exec-plan.md`
- `../docs/reports/llqr-vit-ti16-vit-s16-architecture-support-final-report-2026-04-21.md`
- `../docs/plans/completed/llqr-vgg16bn-wrn28x10-architecture-support-exec-plan.md`
- `../docs/reports/llqr-vgg16bn-wrn28x10-architecture-support-final-report-2026-04-18.md`
- `../tmp/benchmarks/llqr-pyramidnet110-architecture-smokes/README.md`
- `../tmp/benchmarks/llqr-vgg16bn-wrn28x10-architecture-smokes/README.md`
