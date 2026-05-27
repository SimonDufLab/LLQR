# Reproducing LLQR Paper Results

This guide collects commands for the paper-facing LLQR runs:

- [Layerwise LQR for Geometry-Aware Optimization of Deep Networks](https://arxiv.org/abs/2605.04230)
- [Navigating Potholes with Geometry-Aware Sharpness Minimization](https://arxiv.org/abs/2605.16134)

Run commands from the repository root. Replace placeholder paths such as
`<aim_repo>`, `<imagenet2012_location>`, and
`<iwslt14_tokenized_de_en_location>` with local cluster paths before launching.

Use the commands below to reproduce the main experiment families and compare
your runs with the corresponding paper figures and tables.

## Conventions

- Use `uv run python run.py ...` from this repository root.
- For your own runs, save the exact command, Git commit, host, accelerator type, seed,
  dataset path, Aim repository, Aim run hash, final metric, best metric, and
  checkpoint path.
- Treat full CIFAR architecture sweeps, ImageNet, and IWSLT14 translation as
  GPU or cluster jobs.
- For multi-seed runs, add `init_key=<seed>` and save the seed list next to the
  resulting table or figure.

## Reproduction Matrix

| Paper surface | Family | Presets | Main sweep | Data requirement |
|---|---|---|---|---|
| LLQR + SAM | CIFAR architecture runs | `resnet18-*`, `vgg16bn-*`, `wide-resnet28x10-*`, `pyramidnet110-*` | `sam_mode` x `(divergence, ema_decay)` x LLQR toggle | CIFAR via configured loader |
| LLQR + SAM | ImageNet ResNet-50 | `resnet50-imagenet`, `short-resnet50-imagenet` | same as CIFAR | ImageNet 2012 directory |
| LLQR + SAM | IWSLT14 De-En Transformer | `transformer-iwslt14-de-en` | `sam_mode`, fixed `ngd/0.925`, LLQR toggle | tokenized fairseq-style IWSLT14 directory |
| LLQR large-batch route | ImageNet ResNet-50 | `resnet50-imagenet` | chunked grouped LLQR update route | ImageNet 2012 directory |

## LLQR Toggle Convention

Use `use_preconditioner` and `sam_use_preconditioner_on_update` together for
`base_sam` and `base_fsam` ablations.

LLQR-disabled:

```bash
use_preconditioner=false sam_use_preconditioner_on_update=false perturb_mode=ema_grad
```

LLQR-enabled:

```bash
use_preconditioner=true sam_use_preconditioner_on_update=true perturb_mode=ema_precond_grad
```

For `fisher_sam`, use `perturbation_rho=0.1 perturb_mode=ema_grad`.
Canonical Fisher-SAM uses the vanilla outer update, so
`sam_use_preconditioner_on_update` is intentionally inert for that mode.

## CIFAR SAM/LLQR Paper Runs

The CIFAR paper sweep uses these experiment presets:

```text
resnet18-cifar100
vgg16bn-cifar100
wide-resnet28x10-cifar100
pyramidnet110-cifar100
resnet18-cifar10
vgg16bn-cifar10
wide-resnet28x10-cifar10
pyramidnet110-cifar10
```

The CIFAR sweep varies:

- `sam_mode`: `base_sam`, `base_fsam`, `fisher_sam`
- paired divergence and EMA settings:
  `divergence=ngd ema_decay=0.95` and
  `divergence=newton ema_decay=0.9`
- the LLQR toggle pair for `base_sam` and `base_fsam`
- `init_key`, when running additional seeds

Launch the main variants:

```bash
experiments=(
  resnet18-cifar100
  vgg16bn-cifar100
  wide-resnet28x10-cifar100
  pyramidnet110-cifar100
  resnet18-cifar10
  vgg16bn-cifar10
  wide-resnet28x10-cifar10
  pyramidnet110-cifar10
)

sam_modes=(base_sam base_fsam fisher_sam)
divergence_pairs=("ngd 0.95" "newton 0.9")

for experiment in "${experiments[@]}"; do
  for sam_mode in "${sam_modes[@]}"; do
    for pair in "${divergence_pairs[@]}"; do
      read -r divergence ema_decay <<< "${pair}"
      fisher_args=()
      if [ "${sam_mode}" = "fisher_sam" ]; then
        fisher_args=(perturbation_rho=0.1 perturb_mode=ema_grad)
      fi

      uv run python run.py \
        experiment="${experiment}" \
        sam_mode="${sam_mode}" \
        divergence="${divergence}" \
        ema_decay="${ema_decay}" \
        "logging.aim_repo=<aim_repo>" \
        "${fisher_args[@]}"
    done
  done
done
```

Add the LLQR-disabled or LLQR-enabled override block from
[LLQR Toggle Convention](#llqr-toggle-convention) to each `base_sam` and
`base_fsam` command as needed.

## ImageNet ResNet-50 SAM/LLQR Paper Runs

The ImageNet reproduction presets are:

- `resnet50-imagenet`: current config uses `total_epochs=100` and
  `precond_batch_size=256`.
- `short-resnet50-imagenet`: current config uses `total_epochs=50` and
  `precond_batch_size=128`.

Provide the ImageNet 2012 data location at launch time with
`dataset.dataset_dir=<imagenet2012_location>`.

The ImageNet sweep uses the same `sam_mode` and divergence pairs as CIFAR. The
LLQR-enabled SAM setting is
`perturbation_rho=0.075 gbar_beta=0.6 perturb_mode=ema_precond_grad`; keep those
as explicit overrides for this experiment family.

Launch the main ImageNet variants:

```bash
imagenet_experiments=(
  resnet50-imagenet
  short-resnet50-imagenet
)

sam_modes=(base_sam base_fsam fisher_sam)
divergence_pairs=("ngd 0.95" "newton 0.9")

for experiment in "${imagenet_experiments[@]}"; do
  for sam_mode in "${sam_modes[@]}"; do
    for pair in "${divergence_pairs[@]}"; do
      read -r divergence ema_decay <<< "${pair}"
      paper_args=(perturbation_rho=0.075 gbar_beta=0.6)
      llqr_args=(
        use_preconditioner=true
        sam_use_preconditioner_on_update=true
        perturb_mode=ema_precond_grad
      )
      if [ "${sam_mode}" = "fisher_sam" ]; then
        paper_args=(perturbation_rho=0.1 perturb_mode=ema_grad)
        llqr_args=()
      fi

      uv run python run.py \
        experiment="${experiment}" \
        "dataset.dataset_dir=<imagenet2012_location>" \
        sam_mode="${sam_mode}" \
        divergence="${divergence}" \
        ema_decay="${ema_decay}" \
        "logging.aim_repo=<aim_repo>" \
        "${paper_args[@]}" \
        "${llqr_args[@]}"
    done
  done
done
```

For `base_sam` and `base_fsam`, replace `llqr_args` with the LLQR-disabled
override block when running the non-LLQR ablation.

## IWSLT14 Transformer SAM/LLQR Paper Runs

The IWSLT14 reproduction preset is `transformer-iwslt14-de-en`. It expects the
fairseq-style tokenized IWSLT14 German-to-English directory. Either set
`IWSLT14_DATA_DIR` or pass
`dataset.dataset_dir=<iwslt14_tokenized_de_en_location>` at launch time.

Prepare the tokenized text dataset once on a login or prep node:

```bash
SCRATCH=/path/to/scratch bash scripts/prepare_iwslt14_de_en_scratch.sh
```

The helper downloads the public IWSLT14 De-En archive, uses Moses and
`subword-nmt`, and writes:

```text
$SCRATCH/iwslt14_de_en_cache/iwslt14.tokenized.de-en/
$SCRATCH/iwslt14_de_en_cache/iwslt14.tokenized.de-en.tar.gz
```

If Moses or `subword-nmt` already exist on the cluster, point the helper at the
shared installs:

```bash
SCRATCH=/path/to/scratch \
MOSES_ROOT=/path/to/mosesdecoder \
SUBWORD_NMT_ROOT=/path/to/subword-nmt \
bash scripts/prepare_iwslt14_de_en_scratch.sh
```

Then stage the packed dataset into node-local storage inside each SLURM job:

```bash
cd "$SLURM_TMPDIR"
tar -xzf "$SCRATCH/iwslt14_de_en_cache/iwslt14.tokenized.de-en.tar.gz"
export IWSLT14_DATA_DIR="$SLURM_TMPDIR/iwslt14.tokenized.de-en"
```

The training loader builds `.llqr_numeric_cache/` inside the extracted tokenized
directory on first use.

The explored IWSLT14 setup uses `divergence=ngd ema_decay=0.925`. Do not use
the CIFAR/ImageNet Newton pair unless a new reproduction note validates it for
translation. The preset keeps `start_sam_after_step=4000`, so active SAM begins
after inverse-sqrt warmup.

Launch the documented IWSLT14 variants:

```bash
iwslt_sam_modes=(base_sam base_fsam fisher_sam)

for sam_mode in "${iwslt_sam_modes[@]}"; do
  fisher_args=()
  if [ "${sam_mode}" = "fisher_sam" ]; then
    fisher_args=(perturbation_rho=0.1 perturb_mode=ema_grad)
  fi

  uv run python run.py \
    experiment=transformer-iwslt14-de-en \
    "dataset.dataset_dir=<iwslt14_tokenized_de_en_location>" \
    sam_mode="${sam_mode}" \
    divergence=ngd \
    ema_decay=0.925 \
    "logging.aim_repo=<aim_repo>" \
    "${fisher_args[@]}"
done
```

For `base_sam` and `base_fsam`, add the LLQR-disabled or LLQR-enabled override
block from [LLQR Toggle Convention](#llqr-toggle-convention) as needed.

## Large-Batch ResNet-50 Route

For ResNet-50/ImageNet runs that need `precond_batch_size=256` on the validated
A100 surface, prefer grouped outer chunking with the default exact mixed-term
second-order route:

```bash
uv run python run.py \
  experiment=resnet50-imagenet \
  "dataset.dataset_dir=<imagenet2012_location>" \
  llqr_batch_update_mode=chunked_lqr_segment \
  llqr_batch_update_chunk_size=128 \
  llqr_use_fast_paths=true \
  "logging.aim_repo=<aim_repo>"
```

Keep `llqr_second_order_mode=batched_exact` and
`llqr_second_order_chunk_size=null` for this route. The opt-in
`llqr_second_order_mode=sample_separable_exact` route is an exact fallback for
eligible grouped LLQR segments, not the recommended compute path when grouped
chunked `batched_exact` already fits.

## Run Log Checklist

For your own reproduction notes, save:

- paper figure or table target
- dataset and model
- exact command and overrides
- seed list
- expected metric and final observed metric
- compute type and count
- wall-clock time
- Aim run hash
- checkpoint path or download URL
- known caveats, failed seeds, or evaluation notes
