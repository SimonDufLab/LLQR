"""Simple dictionary file to match string identifiers to the desired output"""
from lqr_optimizer._src.models.mlp import create_mlp
from lqr_optimizer._src.models.resnet import create_resnet18
from lqr_optimizer._src.utils import divergence

model_choice = {
  "mlp": create_mlp,
  "resnet-18": create_resnet18,
}

divergence_choice = {
  "ngd": divergence.ngd_divergence_f,
  "renyi": divergence.renyi_divergence,
  "newton": None, # special case where we linearize the known loss function grad to retrieve the final Q of the LQR
}