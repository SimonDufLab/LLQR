"""Simple dictionary file to match string identifiers to the desired output"""
from lqr_optimizer._src.models.mlp import create_mlp
from lqr_optimizer._src.models.resnet import create_resnet18
from lqr_optimizer._src.models.grok_model import create_grok_model
from lqr_optimizer._src.utils import divergence

from lqr_optimizer._src.utils.utils import cosine_annealing_schedule_per_epoch, step_warmup, linear_schedule

model_choice = {
  "mlp": create_mlp,
  "resnet-18": create_resnet18,
  "grok-transformer": create_grok_model,
}

divergence_choice = {
  "ngd": divergence.ngd_divergence_f,
  "renyi": divergence.renyi_divergence,
  "newton": None, # special case where we linearize the known loss function grad to retrieve the final Q of the LQR
}

lr_schedule_choice = {
  "cosine_annealing": cosine_annealing_schedule_per_epoch,
  "step_warmup": step_warmup,
  "linear": linear_schedule,
}