# Layerwise LQR (LLQR)

[Layerwise LQR for Geometry-Aware Optimization of Deep Networks](https://arxiv.org/abs/2605.04230)

[Navigating Potholes with Geometry-Aware Sharpness Minimization](https://arxiv.org/abs/2605.16134)

[Reproduction guide](REPRODUCTION.md)

[Configs](configs/)

[Citation](#citation)

Layerwise LQR (LLQR) treats a deep network as a sequence of layerwise state
transitions and uses that structure to build geometry-aware optimization
updates. This repository contains the current JAX/Flax research implementation:
the scalable relaxed LLQR preconditioner path used in training loops, exact or
toy reference paths for smaller checks, and the LLQR + SAM / Friendly-SAM
experiment surfaces used by the paper-facing releases.

## What Is In This Repository?

- A Hydra-driven experiment runner in `run.py`.
- A relaxed LLQR preconditioner implementation in `lqr_optimizer/_src/preconditioner.py`.
- Exact or benchmark-style second-order helpers in `lqr_optimizer/_src/exact_methods.py`.
- Structured inverse-preconditioner families, including diagonal, Kronecker,
  EKFAC-style, separable EKFAC-style, and PSD variants.
- SAM-family training modes: `base_sam`, `base_fsam`, `past_fsam`, `asam`, and
  `fisher_sam`.
- Public experiment presets for CIFAR, ImageNet, WikiText-103, grokking-style
  transformers, IWSLT14 German-to-English translation, and several CIFAR
  architecture families.
- Toy and reduced validation paths for local sanity checks.

Current status:

- Maintained: `run.py`, Hydra configs under `configs/`, the relaxed LLQR
  preconditioner path, SAM-family modes, CIFAR presets, IWSLT14 translation, and
  current architecture surfaces.
- Experimental: low-memory LLQR operator modes, sample-separable second-order
  paths, large-batch ImageNet routing, and research-only SAM ablations.
- Secondary or stale until audited: `run_single_layer_test.py` and any workflow
  that assumes the exact LLQR baseline is the main large-model training path.

## TL;DR Quickstart

Use the quick local profile when you only want to check that the training path,
config composition, model construction, and LLQR preconditioner wiring are
working.

```bash
uv run python run.py experiment=quick-local-test
```

Expected behavior:

- Hydra composes `configs/experiment/quick-local-test.yaml`.
- The run builds a small MNIST-style training path and an `e-kfac` LLQR
  preconditioner.
- CLI output reports training and validation progress.
- Local logging artifacts may appear under `.aim/` and `outputs/`.

This is a smoke test, not a paper reproduction run. Full paper-result commands
are in [REPRODUCTION.md](REPRODUCTION.md).

## Installation

This repository uses Python 3.11+ and `uv`.

```bash
uv sync --python 3.11
```

Then run commands from the repository root:

```bash
uv run python run.py experiment=quick-local-test
```

Notes:

- The current `pyproject.toml` pins JAX with the CUDA 12 local extra. CPU-only or
  cluster-specific environments may need a local JAX install adjustment that
  matches the target machine.
- Aim is used for experiment logging.
- Large CIFAR, ImageNet, IWSLT14, and ViT-class runs are not intended as casual
  laptop CPU smoke tests.

## Minimal Usage

Experiments are selected with Hydra:

```bash
uv run python run.py experiment=resnet18-cifar10
```

Common override shape:

```bash
uv run python run.py experiment=resnet18-cifar10 \
  block_structure=e-kfac \
  precond_steps=50 \
  precond_batch_size=64
```

SAM-family modes are selected with `sam_mode`:

```bash
uv run python run.py experiment=resnet18-cifar10 \
  sam_mode=base_sam \
  perturbation_rho=0.1
```

The current public interface is the config-driven `run.py` workflow. Treat
internal modules under `lqr_optimizer/_src/` as research implementation surfaces
rather than a stable installed optimizer-wrapper API.

## Reproducing Paper Results

Paper-result details belong in [REPRODUCTION.md](REPRODUCTION.md). That guide
contains commands for CIFAR, ImageNet, IWSLT14, and the large-batch ResNet-50
route.

## Repository Map

- `run.py`: main Hydra training and evaluation entrypoint.
- `configs/`: experiment, dataset, architecture, scheduler, preconditioner, and
  SAM configuration groups.
- `lqr_optimizer/_src/preconditioner.py`: relaxed LLQR preconditioner logic.
- `lqr_optimizer/_src/utils/build_lqr.py`: layerwise LQR construction and
  geometry terms.
- `lqr_optimizer/_src/utils/build_lqr_segments.py`: grouped LLQR segment
  builders for split-stage models.
- `lqr_optimizer/_src/utils/sam_mode_handlers.py`: SAM-family train-step
  orchestration.
- `lqr_optimizer/_src/utils/dataloaders/`: dataset loaders, including IWSLT14
  German-to-English text support.
- `lqr_optimizer/_src/models/`: Flax model definitions.
- `lqr_optimizer/_src/block_matrices_approx/`: structured inverse-preconditioner
  parameterizations.

## Experiments

Representative public presets include:

- `quick-local-test`: small local smoke profile.
- `resnet18-cifar10`, `resnet18-cifar100`, `resnet18-cifar100-adamw`.
- `resnet50-imagenet`, `short-resnet50-imagenet`.
- `vgg16bn-cifar10`, `vgg16bn-cifar100`.
- `wide-resnet28x10-cifar10`, `wide-resnet28x10-cifar100`.
- `pyramidnet110-cifar10`, `pyramidnet110-cifar100`.
- `vit-ti16-cifar100-adamw`.
- `gpt2small-wikitext103`, `grokking`.
- `transformer-iwslt14-de-en` and
  `transformer-iwslt14-de-en-ema-cosine0925`.

For ResNet-50/ImageNet runs that need the validated large-batch update route,
the current recommended shape is:

```bash
uv run python run.py experiment=resnet50-imagenet \
  llqr_batch_update_mode=chunked_lqr_segment \
  llqr_batch_update_chunk_size=128 \
  llqr_use_fast_paths=true
```

Use external GPU hardware for full large-model training and benchmark claims.

## Citation

```bibtex
@misc{dufortlabbe2026layerwise,
  title={Layerwise LQR for Geometry-Aware Optimization of Deep Networks},
  author={Simon Dufort-Labb{\'e} and Pierre-Luc Bacon and Razvan Pascanu and Simon Lacoste-Julien and Aristide Baratin},
  year={2026},
  eprint={2605.04230},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{dufortlabbe2026potholes,
  title={Navigating Potholes with Geometry-Aware Sharpness Minimization},
  author={Simon Dufort-Labb{\'e} and Mehrab Hamidi and Razvan Pascanu and Ioannis Mitliagkas and Damien Scieur and Aristide Baratin},
  year={2026},
  eprint={2605.16134},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## License

No repository license file is currently declared. Check with the maintainers
before redistributing or reusing this code outside the intended research-release
context.
