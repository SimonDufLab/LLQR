"""Simple dictionary file to match string identifiers to the desired output"""
from jax.tree_util import Partial
from lqr_optimizer._src.models.mlp import create_mlp, create_mlp_legacy
from lqr_optimizer._src.models.resnet import create_resnet18, create_resnet50
from lqr_optimizer._src.models.pyramidnet import create_pyramidnet110
from lqr_optimizer._src.models.vgg import create_vgg16bn
from lqr_optimizer._src.models.wide_resnet import create_wide_resnet28x10
from lqr_optimizer._src.models.grok_model import create_grok_model
from lqr_optimizer._src.models.gpt import create_gpt_model
from lqr_optimizer._src.models.vit import create_vit_s16, create_vit_ti16
from lqr_optimizer._src.utils import divergence

from lqr_optimizer._src.utils.utils import (cosine_annealing_schedule_per_epoch, step_warmup, linear_schedule,
                                            warmup_cosine_annealing_schedule, piecewise_constant_schedule,
                                            warmup_piecewise_decay_schedule)

model_choice = {
  "mlp": create_mlp,
  "mlp-legacy": create_mlp_legacy,
  "resnet-18": create_resnet18,
  "resnet-50": create_resnet50,
  "pyramidnet-110": create_pyramidnet110,
  "vgg16-bn": create_vgg16bn,
  "wide-resnet-28-10": create_wide_resnet28x10,
  "grok-transformer": create_grok_model,
  "gpt2-small": create_gpt_model,
  "vit-ti-16": create_vit_ti16,
  "vit-s-16": create_vit_s16,
}

divergence_choice = {
  "ngd": divergence.ngd_divergence_f,
  "renyi": divergence.renyi_divergence,
  "stable_renyi": divergence.renyi_divergence_stable,
  "renyi_inf": divergence.renyi_inf,
  "renyi_zero": divergence.renyi_zero,
  "renyi_2": divergence.renyi_two,
  "renyi_half": divergence.renyi_half,
  "reverse_kl": divergence.reverse_kl,
  "newton": None, # special case where we linearize the known loss function grad to retrieve the final Q of the LQR
}

lr_schedule_choice = {
  "cosine_annealing": cosine_annealing_schedule_per_epoch,
  "cosine_decay_with_floor": Partial(cosine_annealing_schedule_per_epoch, cycle=False),
  "warmup_cosine_decay": warmup_cosine_annealing_schedule,
  "step_warmup": step_warmup,
  "linear": linear_schedule,
  "piecewise_constant": piecewise_constant_schedule,
  "warmup_piecewise_decay": warmup_piecewise_decay_schedule,
}
