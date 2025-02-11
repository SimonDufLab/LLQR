"""Simple dictionary file to match string identifiers to the desired output"""
from lqr_optimizer._src.models.mlp import create_mlp
from lqr_optimizer._src.models.resnet import create_resnet18

model_choice = {
  "mlp": create_mlp,
  "resnet-18": create_resnet18,
}