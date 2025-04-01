"""Preconditioner classes and their update rules."""
import abc

import optax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial
from flax.core.frozen_dict import FrozenDict

from lqr_optimizer._src.utils.utils import normalize_gradient, timed_jit, vmapped_clip_norm, pytree_max_min, pytree_l2_norm
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
               precond_clip_norm,
               preconditioner_update_steps,
               multibatch: bool = False,
               damping:float = 0.0,
               divergence_args_index = -1):
    self._divergence_function = divergence_function
    self._damping =damping
    self._loss_fn = loss_fn
    self._optax_solver = optax_solver
    self._clip_norm = precond_clip_norm
    self._layer_names = list(network_params.keys())
    self._block_structure = BLOCK_STRUCTURE_DICT[block_structure](network_params, self._layer_names, block_structure_init)
    # self._block_structure.make_blocks(network_params, model.layer_names)
    self._layer_apply = model.apply_block_from_params
    self._model_apply = model.apply
    self._divergence_args_index = divergence_args_index
    self._preconditioner_update_steps = preconditioner_update_steps
    self._multibatch = multibatch

    self._update_preconditioner_fn = self._get_evaluate_lqr(self._optax_solver, self._preconditioner_update_steps, multibatch=self._multibatch)

  def apply(self, update):
    return self._block_structure.matrix_product(self._block_structure.blocks, update)

  def get_stats(self):
    precond_max, precond_min = pytree_max_min(self._block_structure.blocks)
    precond_norm = pytree_l2_norm(self._block_structure.blocks)
    return precond_max, precond_min, precond_norm

  def _get_evaluate_lqr(self, optax_solver=None, steps=1, multibatch=False):
    def compute_loss(_params, _other_model_variables, x, y):
      return self._loss_fn(self._model_apply({'params': _params}|_other_model_variables, x), y)

    if multibatch:
      @jax.jit
      def evaluate_lqr(preconditioner, params, other_model_variables, datapoint):
        inputs, targets = datapoint
        a, b, a_transpose, states =lqr_forward_matrices_and_states(inputs, params, self._layer_apply, self._layer_names,
                                                                   other_model_variables)
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
        gradients = normalize_gradient(gradients)
        gradients = jax.tree_map(lambda v: -1 * v, gradients)  # Starting update is negative gradient
        cost = 0
        x = jnp.zeros(states[0].size)
        u_dict = self._block_structure.matrix_product(preconditioner, gradients)
        for i, layer_name in enumerate(self._layer_names):
          u, _ = ravel_pytree(u_dict[layer_name])
          cost += (x.T @ q_backward[-i - 1](x) + u.T @ r_backward[-i - 1](u)) / 2 + u.T @ m_backward[-i - 1](x)
          x = a[i](x) + b[i](u)

        cost += x.T@final_lin_cost + (x.T@final_q(x))/2

        return cost
      return evaluate_lqr

    else:
      # @jax.jit
      def evaluate_lqr_grad(preconditioner, params, other_model_variables, datapoint):
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
        gradients = normalize_gradient(gradients)
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

      vmapped_evaluate_lqr_grad = jax.vmap(evaluate_lqr_grad, in_axes=(None, None, None, (0, 0)))
      def get_precond_grad(preconditioner, params, other_model_variables, datapoint):
        grads = vmapped_evaluate_lqr_grad(preconditioner, params, other_model_variables, datapoint)
        grads = vmapped_clip_norm(grads, self._clip_norm)
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

      @jax.jit
      def get_update(preconditioner, params, other_model_variables, datapoint):
        # Initialize the optimizer state for the preconditioner.
        opt_state = optax_solver.init(preconditioner)

        # Define a single update step to be run in the compiled loop.
        def update_step(i, state):
          precond, opt_state = state
          precond_grad = get_precond_grad(precond, params, other_model_variables, datapoint)
          updates, opt_state = optax_solver.update(precond_grad, opt_state)
          new_precond = optax.apply_updates(precond, updates)
          return (new_precond, opt_state)

        # Use a compiled loop to perform “steps” iterations.
        init_state = (preconditioner, opt_state)
        final_precond, _ = jax.lax.fori_loop(0, steps, update_step, init_state)
        # Safeguard against any numerical issues
        return jax.tree_map(Partial(jnp.nan_to_num, nan=1.0, posinf=1.0, neginf=1.0), final_precond)

      return get_update

  def update_preconditioner(self, params, dataloader, other_model_variables=FrozenDict({})):
    """params is the current weights of the NN"""
    if self._multibatch:
      lqr_loss_fn = lambda x, y : jnp.mean(jax.vmap(self._get_evaluate_lqr(params, multibatch=True, other_model_variables=other_model_variables), in_axes=(None, (0,0)))(x, y))
      # lqr_loss_fn = lambda x: jnp.mean(jax.vmap(self._get_evaluate_lqr, in_axes=(None, (0,0)))(params, next(dataloader))(x))
      # reinitialize the optimizer state every time:
      opt_state = self._optax_solver.init(self._block_structure.blocks)
      # datapoint = next(dataloader)
      for _ in range(self._preconditioner_update_steps):
        _cost, precond_grad = jax.value_and_grad(lqr_loss_fn, argnums=0)(self._block_structure.blocks, next(dataloader))
        # precond_grad = jax.grad(lqr_loss_fn, argnums=0)(self._block_structure.blocks)
        updates, opt_state = self._optax_solver.update(precond_grad, opt_state)
        self._block_structure.update_blocks(optax.apply_updates(self._block_structure.blocks, updates))
        # print(_cost)
    else:
      self._block_structure.update_blocks(
        self._update_preconditioner_fn(self._block_structure.blocks, params, other_model_variables, next(dataloader)))

    # print(self._block_structure.blocks["layers_2"])
    # print(self._block_structure.blocks)

  def expose_blocks(self):
    return self._block_structure.blocks

  def load_blocks(self, saved_blocks):
    self._block_structure.update_blocks(saved_blocks)
