"""Preconditioner classes and their update rules."""
import abc

import optax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial
from flax.core.frozen_dict import FrozenDict

from lqr_optimizer._src.utils.utils import normalize_gradient, timed_jit, vmapped_clip_norm, pytree_max_min, pytree_l2_norm, get_per_layer_norm
import lqr_optimizer._src.block_matrices_approx.block_structures as block_structures
from lqr_optimizer._src.utils.build_lqr import (lqr_forward_matrices_and_states, lqr_final_costs_and_adjoints,
                             lqr_backward_matrices_and_adjoints)

BLOCK_STRUCTURE_DICT = {
  'dense': block_structures.DenseBlock,
  'diagonal': block_structures.DiagonalBlock,
  'scalar': block_structures.ScalarBlock,
  'kfac': block_structures.KroneckerBlock
}

class BasePreconditioner(abc.ABC):
  def __init__(self,
               divergence_function,
               loss_fn, # As applied to the NN output
               block_structure,
               block_structure_init,
               model,
               network_params,
               optax_solver,
               trainstate_solver,
               precond_clip_norm,
               preconditioner_update_steps,
               multibatch: bool = False,
               precond_on_update: bool =False,
               normalize_grad_for_lqr = True,
               damping:float = 0.0,
               divergence_args_index = -1):
    self._divergence_function = divergence_function
    self._damping =damping
    self._loss_fn = loss_fn
    self._optax_solver = optax_solver
    self._trainstate_solver = trainstate_solver
    self._clip_norm = precond_clip_norm
    self._layer_names = list(network_params.keys())
    self._block_structure = BLOCK_STRUCTURE_DICT[block_structure](network_params, self._layer_names, block_structure_init)
    self._block_structure_name = block_structure
    # self._block_structure.make_blocks(network_params, model.layer_names)
    self._layer_apply = model.apply_block_from_params
    self._model_apply = model.apply
    self._divergence_args_index = divergence_args_index
    self._preconditioner_update_steps = preconditioner_update_steps
    self._multibatch = multibatch
    self._precond_on_update = precond_on_update
    if normalize_grad_for_lqr:
      self._normalize_grad_for_lqr_fn = normalize_gradient
    else:
      self._normalize_grad_for_lqr_fn = lambda _: _  # Nothing, identity fn
    if precond_clip_norm:
      self._clip_norm_fn = vmapped_clip_norm
    else:
      self._clip_norm_fn = lambda x, _: x # Nothing, identity fn

    self._update_preconditioner_fn = self._get_evaluate_lqr(self._optax_solver, self._preconditioner_update_steps,
                                                            multibatch=self._multibatch,
                                                            precond_on_update=self._precond_on_update)

  def apply(self, update):
    return self._block_structure.matrix_product(self._block_structure.blocks, update)

  def get_stats(self):
    precond_max, precond_min = pytree_max_min(self._block_structure.blocks)
    precond_norm = pytree_l2_norm(self._block_structure.blocks)
    per_layer_norm = get_per_layer_norm(self._block_structure.blocks)
    return precond_max, precond_min, precond_norm, per_layer_norm

  def _get_evaluate_lqr(self, optax_solver=None, steps=1, multibatch=False, precond_on_update=False):
    def compute_loss(_params, _other_model_variables, x, y):
      if type(_other_model_variables) is FrozenDict:
        _other_model_variables = dict(_other_model_variables)
      return self._loss_fn(self._model_apply({'params': _params}|_other_model_variables, x), y)

    # if multibatch:
    #   @jax.jit
    #   def evaluate_lqr(preconditioner, params, other_model_variables, datapoint):
    #     inputs, targets = datapoint
    #     a, b, a_transpose, states =lqr_forward_matrices_and_states(inputs, params, self._layer_apply, self._layer_names,
    #                                                                other_model_variables)
    #     if self._divergence_args_index is not None:
    #       div_arg = states[self._divergence_args_index]
    #     else:
    #       div_arg = None
    #     final_q, final_p, final_lin_cost = lqr_final_costs_and_adjoints(self._loss_fn, states[-1], targets,
    #                                                                     div_f=self._divergence_function,
    #                                                                     div_arg=div_arg)
    #     final_lin_cost = jnp.atleast_1d(final_lin_cost)
    #     q_backward, r_backward, m_backward, m_transpose_backward = lqr_backward_matrices_and_adjoints(params, states,
    #                                                                                                   final_p,
    #                                                                                                   a_transpose,
    #                                                                                                   self._layer_apply,
    #                                                                                                   self._layer_names,
    #                                                                                                   self._damping,
    #                                                                                                   other_model_variables)
    #     gradients = jax.grad(compute_loss, argnums=0)(params, other_model_variables, inputs, targets)
    #     gradients = self._normalize_grad_for_lqr_fn(gradients)
    #     gradients = jax.tree_map(lambda v: -1 * v, gradients)  # Starting update is negative gradient
    #     cost = 0
    #     x = jnp.zeros(states[0].size)
    #     u_dict = self._block_structure.matrix_product(preconditioner, gradients)
    #     for i, layer_name in enumerate(self._layer_names):
    #       u, _ = ravel_pytree(u_dict[layer_name])
    #       cost += (x.T @ q_backward[-i - 1](x) + u.T @ r_backward[-i - 1](u)) / 2 + u.T @ m_backward[-i - 1](x)
    #       x = a[i](x) + b[i](u)
    #
    #     cost += x.T@final_lin_cost + (x.T@final_q(x))/2
    #
    #     return cost
    #   return evaluate_lqr

    # else:
    @jax.jit
    def evaluate_lqr_grad(preconditioner, params, other_model_variables, datapoint, trainstate_opt_state):
      inputs, targets = datapoint
      a, b, a_transpose, states = lqr_forward_matrices_and_states(inputs, params, self._layer_apply,
                                                                  self._layer_names, other_model_variables)
      if self._divergence_args_index is not None:
        div_arg = states[self._divergence_args_index]
      else:
        div_arg = None
      final_q, final_p, final_lin_cost = lqr_final_costs_and_adjoints(self._loss_fn, states[-1], targets,
                                                                      div_f=self._divergence_function,
                                                                      div_arg=div_arg)
      final_lin_cost = jnp.atleast_1d(final_lin_cost)
      q_backward, r_backward, m_backward, m_transpose_backward = lqr_backward_matrices_and_adjoints(params, states,
                                                                                                    final_p,
                                                                                                    a_transpose,
                                                                                                    self._layer_apply,
                                                                                                    self._layer_names,
                                                                                                    self._damping,
                                                                                                    other_model_variables)
      gradients = jax.grad(compute_loss, argnums=0)(params, other_model_variables, inputs, targets)
      if precond_on_update:
        gradients, _ = self._trainstate_solver.update(gradients, trainstate_opt_state)
      gradients = self._normalize_grad_for_lqr_fn(gradients)
      gradients = jax.tree_map(lambda v: -1 * v, gradients)  # Starting update is negative gradient

      def lqr_cost(_preconditioner):
        cost = 0
        x = jnp.zeros(states[0].size)
        u_dict = self._block_structure.matrix_product(_preconditioner, gradients)
        for i, layer_name in enumerate(self._layer_names):
          u, _ = ravel_pytree(u_dict[layer_name])
          cost += (x.T @ q_backward[-i - 1](x) + u.T @ r_backward[-i - 1](u)) / 2 + u.T @ m_backward[-i - 1](x)
          x = a[i](x) + b[i](u)

        cost += x.T @ final_lin_cost + (x.T @ final_q(x)) / 2

        return cost

      # opt_state = optax_solver.init(preconditioner)
      # for _ in range(steps):
      #   precond_grad = jax.grad(lqr_cost, argnums=0)(preconditioner)
      #   _update, opt_state = optax_solver.update(precond_grad, opt_state)
      #   preconditioner = optax.apply_updates(preconditioner, _update)
      # return jax.tree_map(Partial(jnp.nan_to_num, nan=1.0, posinf=1.0, neginf=1.0), preconditioner)

      local_precond_grad = jax.grad(lqr_cost, argnums=0)(preconditioner)
      return jax.tree_map(Partial(jnp.nan_to_num, nan=1.0, posinf=1.0, neginf=1.0), local_precond_grad)

    vmapped_evaluate_lqr_grad = jax.vmap(evaluate_lqr_grad, in_axes=(None, None, None, (0, 0), None))
    def get_precond_grad(preconditioner, params, other_model_variables, datapoint, trainstate_opt_state):
      grads = vmapped_evaluate_lqr_grad(preconditioner, params, other_model_variables, datapoint, trainstate_opt_state)
      grads = self._clip_norm_fn(grads, self._clip_norm)
      return jax.tree_map(Partial(jnp.mean, axis=0), grads)

      # @timed_jit # switch back to jax.jit after debugging
      # # jax.jit
      # def get_update(preconditioner, params, other_model_variables, datapoint):
      #   opt_state = optax_solver.init(preconditioner)
      #   for _ in range(steps):
      #     precond_grad = get_precond_grad(preconditioner, params, other_model_variables, datapoint)
      #     _update, opt_state = optax_solver.update(precond_grad, opt_state)
      #     preconditioner = optax.apply_updates(preconditioner, _update)
      #   return jax.tree_map(Partial(jnp.nan_to_num, nan=1.0, posinf=1.0, neginf=1.0), preconditioner)
      #
      # return get_update

    if multibatch:
      def get_update(preconditioner, params, other_model_variables, dataloader, trainstate_opt_state):
        # Initialize the optimizer state for the preconditioner.
        opt_state = optax_solver.init(preconditioner)

        # Define a single update step to be run in the compiled loop.
        @jax.jit
        def update_step(precond, opt_state, datapoint):
          precond_grad = get_precond_grad(precond, params, other_model_variables, datapoint, trainstate_opt_state)
          updates, opt_state = optax_solver.update(precond_grad, opt_state)
          new_precond = optax.apply_updates(precond, updates)
          return (new_precond, opt_state)

        for _ in range(steps):
          precond, opt_state = update_step(preconditioner, opt_state, next(dataloader))
        # Safeguard against any numerical issues
        return jax.tree_map(Partial(jnp.nan_to_num, nan=1.0, posinf=1.0, neginf=1.0), precond)

    else:
      @jax.jit
      def get_update(preconditioner, params, other_model_variables, datapoint, trainstate_opt_state):
        # Initialize the optimizer state for the preconditioner.
        opt_state = optax_solver.init(preconditioner)

        # Define a single update step to be run in the compiled loop.
        def update_step(i, state):
          precond, opt_state = state
          precond_grad = get_precond_grad(precond, params, other_model_variables, datapoint, trainstate_opt_state)
          updates, opt_state = optax_solver.update(precond_grad, opt_state)
          new_precond = optax.apply_updates(precond, updates)
          return (new_precond, opt_state)

        # Use a compiled loop to perform “steps” iterations.
        init_state = (preconditioner, opt_state)
        final_precond, _ = jax.lax.fori_loop(0, steps, update_step, init_state)
        # Safeguard against any numerical issues
        return jax.tree_map(Partial(jnp.nan_to_num, nan=1.0, posinf=1.0, neginf=1.0), final_precond)

    return get_update

  def update_preconditioner(self, params, dataloader, opt_state, other_model_variables=FrozenDict({})):
    """params is the current weights of the NN"""
    if self._multibatch:
      self._block_structure.update_blocks(
        self._update_preconditioner_fn(self._block_structure.blocks, params, other_model_variables, dataloader,
                                       opt_state))
    else:
      self._block_structure.update_blocks(
        self._update_preconditioner_fn(self._block_structure.blocks, params, other_model_variables, next(dataloader), opt_state))

    if self._block_structure_name in ('scalar', "diagonal"):
      # We clip to (almost) 0 those 2 structures to avoid gradient inversion
      self._block_structure.clip_blocks(min_for_block=1e-8)

    # print(self._block_structure.blocks["layers_2"])
    # print(self._block_structure.blocks)

  def expose_blocks(self):
    return self._block_structure.blocks

  def load_blocks(self, saved_blocks):
    self._block_structure.update_blocks(saved_blocks)
