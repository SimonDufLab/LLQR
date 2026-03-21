"""Preconditioner classes and their update rules."""
import abc
from functools import partial

import optax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial
from flax.core.frozen_dict import FrozenDict, freeze

from lqr_optimizer._src.utils.utils import (normalize_gradient, timed_jit, pytree_max_min, pytree_l2_norm,
                                            get_per_layer_norm, _deep_copy_pytree, infer_batch_layout, get_per_layer_skews)
import lqr_optimizer._src.block_matrices_approx.block_structures as block_structures
from lqr_optimizer._src.utils.build_lqr import (lqr_forward_matrices_and_states, lqr_final_costs_and_adjoints,
                             lqr_backward_matrices_and_adjoints)

BLOCK_STRUCTURE_DICT = {
  'dense': block_structures.DenseBlock,
  'diagonal': block_structures.DiagonalBlock,
  'scalar': block_structures.ScalarBlock,
  'kfac': block_structures.KroneckerBlock,
  'e-kfac': block_structures.EKFACBlock,
  'sep-e-kfac': block_structures.SeparableEKFACBlock,
  'psd-e-kfac': block_structures.PSDEKFACBlock,
  'psd-sep-e-kfac': block_structures.PSDSeparableEKFACBlock,
  'rms_norm-e-kfac': Partial(block_structures.EKFACBlockNormalized, norm_mode="rms"),
  'max_norm-e-kfac': Partial(block_structures.EKFACBlockNormalized, norm_mode="maxabs"),
  'scaled-rms_norm-e-kfac': Partial(block_structures.EKFACBlockNormalized, norm_mode="rms", use_alpha=True),
  'scaled-max_norm-e-kfac': Partial(block_structures.EKFACBlockNormalized, norm_mode="maxabs", use_alpha=True),
  'h-kfac': block_structures.HouseholderDiagKroneckerBlock,
  'diag-kfac': block_structures.DiagKroneckerBlock,
  'sym-kfac' : block_structures.MirrorSymKroneckerBlock,
  'psd-sym-kfac' : block_structures.PSDSymKroneckerBlock,
  'sds-sym-kfac' : block_structures.SDSSymKroneckerBlock,
  'sds-sym-kfac_norm-d' : block_structures.NormalizedSDSSymKroneckerBlock,
  'low-rank-memory': block_structures.LowRankBlockMemory,
  'low-rank-memory-asym': block_structures.LowRankBlockMemoryAsym,
  'sym_swm_kfac': block_structures.Sym_SWM_KFAC,
  'asym_swm_kfac': block_structures.Asym_SWM_KFAC,
  'sym_swm_e-kfac': block_structures.Sym_SWM_EKFAC,
  'sym_swm_sep-e-kfac': block_structures.Sym_SWM_SeparableEKFAC,
}


def _recover_loss_gradients_from_transition_transposes(params, layer_names, transition_transposes, final_lin_cost):
  """Recover the exact loss gradient by a reverse sweep over the joint transition transposes."""
  state_cotangent = ravel_pytree(final_lin_cost)[0]
  recovered_by_layer = {}

  for layer_index in range(len(layer_names) - 1, -1, -1):
    layer_name = layer_names[layer_index]
    _, unravel_params_fn = ravel_pytree(params[layer_name])
    param_cotangent, state_cotangent = transition_transposes[layer_index](state_cotangent)
    recovered_by_layer[layer_name] = unravel_params_fn(jnp.ravel(jnp.atleast_1d(param_cotangent)))

  ordered_recovered = {layer_name: recovered_by_layer[layer_name] for layer_name in layer_names}
  if isinstance(params, FrozenDict):
    return freeze(ordered_recovered)
  return ordered_recovered

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
               preconditioner_update_steps,
               precond_rank,
               precond_identity_scaling,
               batch_solve_precond: bool = True,
               multibatch: bool = False,
               precond_on_update: bool =False,
               normalize_grad_for_lqr = True,
               warm_start_precond = True,
               damping: float = 0.0,
               allow_grad_inversion: bool = False,
               divergence_args_index = -1):
    self._divergence_function = divergence_function
    self._damping =damping
    self._loss_fn = loss_fn
    self._optax_solver = optax_solver
    self._trainstate_solver = trainstate_solver
    self._layer_names = list(network_params.keys())
    self._block_structure = BLOCK_STRUCTURE_DICT[block_structure](network_params, self._layer_names,
                                                                  block_structure_init, rank=precond_rank,
                                                                  identity_scale=precond_identity_scaling)
    self._block_structure_name = block_structure
    # self._block_structure.make_blocks(network_params, model.layer_names)
    self._layer_apply = model.apply_block_from_params
    self._model_apply = model.apply
    self._divergence_args_index = divergence_args_index
    self._preconditioner_update_steps = preconditioner_update_steps
    self._batch_solve_precond = batch_solve_precond
    self._multibatch = multibatch
    self._warm_start_precond = warm_start_precond
    self._precond_on_update = precond_on_update
    self._allow_grad_inversion = allow_grad_inversion
    if normalize_grad_for_lqr:
      self._normalize_grad_for_lqr_fn = normalize_gradient
    else:
      self._normalize_grad_for_lqr_fn = lambda _: _  # Nothing, identity fn

    self._update_preconditioner_fn = self._get_evaluate_lqr(self._optax_solver, self._preconditioner_update_steps,
                                                            batch_solve_precond=self._batch_solve_precond,
                                                            multibatch=self._multibatch,
                                                            precond_on_update=self._precond_on_update)
    # self._jit_apply_fn = jax.jit(
    #   lambda blocks, update: self._block_structure.matrix_product(blocks, update)
    # )
  def apply(self, blocks, update):
    # return self._jit_apply_fn(self._block_structure.blocks, update)
    return self._block_structure.matrix_product(blocks, update)

  def get_stats(self):
    if hasattr(self._block_structure, "_memory"):
      precond_max, precond_min = pytree_max_min(self._block_structure.get_memory())
      precond_norm = pytree_l2_norm(self._block_structure.get_memory())
      per_layer_norm = get_per_layer_norm(self._block_structure.get_memory())
    else:
      precond_max, precond_min = pytree_max_min(self._block_structure.blocks)
      precond_norm = pytree_l2_norm(self._block_structure.blocks)
      per_layer_norm = get_per_layer_norm(self._block_structure.blocks)
    return precond_max, precond_min, precond_norm, per_layer_norm

  def get_precond_asymmetry(self):
    if hasattr(self._block_structure, "_memory"):
      skews = get_per_layer_skews(self._block_structure.get_memory())
    else:
      skews = get_per_layer_skews(self._block_structure.blocks)
    return skews


  def _get_evaluate_lqr(self, optax_solver=None, steps=1, batch_solve_precond=True, multibatch=False, precond_on_update=False):
    def compute_loss(_params, _other_model_variables, x, y):
      if type(_other_model_variables) is FrozenDict:
        _other_model_variables = dict(_other_model_variables)
      return self._loss_fn(self._model_apply({'params': _params}|_other_model_variables, x), y)

    if batch_solve_precond:
      def get_operators_and_gradients(params, other_model_variables, datapoint, trainstate_opt_state):
        inputs, targets = datapoint
        transitions, transition_transposes, states = lqr_forward_matrices_and_states(
          inputs, params, self._layer_apply, self._layer_names, other_model_variables)
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
                                                                                                      transition_transposes,
                                                                                                      self._layer_apply,
                                                                                                      self._layer_names,
                                                                                                      self._damping,
                                                                                                      other_model_variables)
        gradients = _recover_loss_gradients_from_transition_transposes(
          params, self._layer_names, transition_transposes, final_lin_cost)
        if precond_on_update:
          gradients, _ = self._trainstate_solver.update(gradients, trainstate_opt_state, params)
        gradients = self._normalize_grad_for_lqr_fn(gradients)
        gradients = jax.tree_map(lambda v: -1 * v, gradients)  # Starting update is negative gradient

        return gradients, (transitions, q_backward, r_backward, m_backward, final_q, final_lin_cost)

      # def lqr_cost(_preconditioner, input_size, gradients, kernel_shapes, operators):
      def lqr_cost(_preconditioner, input_size, gradients, operators):
        transitions, q_backward, r_backward, m_backward, final_q, final_lin_cost = operators
        cost = 0
        x = jnp.zeros(input_size)
        # u_dict = self._block_structure.train_matrix_product_for_scan(_preconditioner, gradients, kernel_shapes)
        u_dict = self._block_structure.train_matrix_product(_preconditioner, gradients)
        for i, layer_name in enumerate(self._layer_names):
          u, _ = ravel_pytree(u_dict[layer_name])
          cost += (x.T @ q_backward[-i - 1](x) + u.T @ r_backward[-i - 1](u)) / 2 + u.T @ m_backward[-i - 1](x)
          x = transitions[i](u, x)

        # cost += x.T @ jnp.squeeze(final_lin_cost) + (x.T @ final_q(x)) / 2 # squeeze is causing problems
        x1 = jnp.ravel(x)  # () -> (1,), (n,1)/(1,n)/(n,) -> (n,)
        c1 = jnp.ravel(final_lin_cost)
        qx1 = jnp.ravel(final_q(x))  # final_q(x) = Qx

        cost += jnp.dot(x1, c1) + 0.5 * jnp.dot(x1, qx1)

        return cost

      # def lqr_grad_fn(_preconditioner, input_size, gradients, kernel_shapes, operators):
      def lqr_grad_fn(_preconditioner, input_size, gradients, operators):
        v, grads = jax.value_and_grad(lqr_cost, argnums=0)(
          # _preconditioner, input_size, gradients, kernel_shapes, operators)
          _preconditioner, input_size, gradients, operators)
        grads = jax.tree_map(Partial(jnp.nan_to_num, nan=0.0, posinf=1.0, neginf=-1.0), grads)
        # grads = jax.tree_map(zero_if_bad, grads)
        # print(jax.tree_map(lambda g: g.shape, grads))
        return v, grads

      @Partial(jax.jit, donate_argnames=("preconditioner",))
      def _get_update(preconditioner, opt_state, params, precond_lr, other_model_variables, datapoint, trainstate_opt_state):
        gradients, operators = get_operators_and_gradients(
          params, other_model_variables, datapoint, trainstate_opt_state)
        # gradients, kernel_shapes = self._block_structure.prepare_vectors(gradients)
        input_size = datapoint[0].size

        # Define a single update step to be run in the compiled loop.
        def update_step(carry, _):
          precond, opt_state = carry
          # precond_grad = lqr_grad_fn(precond, input_size, gradients, kernel_shapes, operators)
          _, precond_grad = lqr_grad_fn(precond, input_size, gradients, operators)
          extra_args = {'value_and_grad_fn': Partial(lqr_grad_fn, input_size=input_size, gradients=gradients, operators=operators)}
          updates, opt_state = optax_solver.update(precond_grad, opt_state, precond, **extra_args)
          updates = jax.tree_map(lambda g: g * precond_lr, updates)
          new_precond = optax.apply_updates(precond, updates)
          return (new_precond, opt_state), None

        # # Use a compiled loop to perform “steps” iterations.
        # init_state = (preconditioner, opt_state)
        # final_precond, _ = jax.lax.fori_loop(0, steps, update_step, init_state)

        (final_precond, _), _ = jax.lax.scan(update_step,
                                             (preconditioner, opt_state),
                                             xs=None, length=steps, unroll=1)
        return jax.tree_map(jnp.nan_to_num, final_precond)


      def get_update(preconditioner, params, precond_lr, other_model_variables, datapoint, trainstate_opt_state):
        # Snapshot of preconditioner to donate arg during jitted fn, but need original for EMA later
        precond_snapshot = _deep_copy_pytree(preconditioner)
        # Initialize the optimizer state for the preconditioner.
        opt_state = optax_solver.init(precond_snapshot)

        return _get_update(precond_snapshot, opt_state, params, precond_lr, other_model_variables, datapoint, trainstate_opt_state)

      return get_update
    else:
      @jax.jit
      def evaluate_lqr_grad(preconditioner, params, other_model_variables, datapoint, trainstate_opt_state):
        inputs, targets = datapoint
        transitions, transition_transposes, states = lqr_forward_matrices_and_states(
          inputs, params, self._layer_apply, self._layer_names, other_model_variables)
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
                                                                                                      transition_transposes,
                                                                                                      self._layer_apply,
                                                                                                      self._layer_names,
                                                                                                      self._damping,
                                                                                                      other_model_variables)
        gradients = jax.grad(compute_loss, argnums=0)(params, other_model_variables, inputs, targets)
        if precond_on_update:
          gradients, _ = self._trainstate_solver.update(gradients, trainstate_opt_state, params)
        gradients = self._normalize_grad_for_lqr_fn(gradients)
        gradients = jax.tree_map(lambda v: -1 * v, gradients)  # Starting update is negative gradient

        def lqr_cost(_preconditioner):
          cost = 0
          x = jnp.zeros(states[0].size)
          u_dict = self._block_structure.train_matrix_product(_preconditioner, gradients)
          for i, layer_name in enumerate(self._layer_names):
            u, _ = ravel_pytree(u_dict[layer_name])
            cost += (x.T @ q_backward[-i - 1](x) + u.T @ r_backward[-i - 1](u)) / 2 + u.T @ m_backward[-i - 1](x)
            x = transitions[i](u, x)

          # cost += x.T @ jnp.squeeze(final_lin_cost) + (x.T @ final_q(x)) / 2 # squeeze is causing problems
          x1 = jnp.ravel(x)  # () -> (1,), (n,1)/(1,n)/(n,) -> (n,)
          c1 = jnp.ravel(final_lin_cost)
          qx1 = jnp.ravel(final_q(x))  # assume final_q(x) ~ Qx

          cost += jnp.dot(x1, c1) + 0.5 * jnp.dot(x1, qx1)

          return cost

        local_precond_grad = jax.grad(lqr_cost, argnums=0)(preconditioner)
        return jax.tree_map(Partial(jnp.nan_to_num, nan=1.0, posinf=1.0, neginf=1.0), local_precond_grad)

      vmapped_evaluate_lqr_grad = jax.vmap(evaluate_lqr_grad, in_axes=(None, None, None, (0, 0), None))

      def get_precond_grad(preconditioner, params, other_model_variables, datapoint, trainstate_opt_state):
        # print(datapoint[0].shape)
        datapoint = tuple(jnp.expand_dims(x, axis=1) for x in datapoint)
        # print(datapoint[0].shape)
        grads = vmapped_evaluate_lqr_grad(preconditioner, params, other_model_variables, datapoint,
                                          trainstate_opt_state)
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
        def get_update(preconditioner, params, precond_lr, other_model_variables, dataloader, trainstate_opt_state):
          # Initialize the optimizer state for the preconditioner.
          opt_state = optax_solver.init(preconditioner)

          # Define a single update step to be run in the compiled loop.
          @jax.jit
          def update_step(precond, opt_state, datapoint):
            precond_grad = get_precond_grad(precond, params, other_model_variables, datapoint, trainstate_opt_state)
            updates, opt_state = optax_solver.update(precond_grad, opt_state)
            updates = jax.tree_map(lambda g: g * precond_lr, updates)
            new_precond = optax.apply_updates(precond, updates)
            return (new_precond, opt_state)

          for _ in range(steps):
            precond, opt_state = update_step(preconditioner, opt_state, next(dataloader))
          # Safeguard against any numerical issues
          return jax.tree_map(Partial(jnp.nan_to_num, nan=1.0, posinf=1.0, neginf=1.0), precond)

      else:
        @jax.jit
        def get_update(preconditioner, params, precond_lr, other_model_variables, datapoint, trainstate_opt_state):
          # Initialize the optimizer state for the preconditioner.
          opt_state = optax_solver.init(preconditioner)

          # Define a single update step to be run in the compiled loop.
          def update_step(i, state):
            precond, opt_state = state
            precond_grad = get_precond_grad(precond, params, other_model_variables, datapoint, trainstate_opt_state)
            updates, opt_state = optax_solver.update(precond_grad, opt_state)
            updates = jax.tree_map(lambda g: g * precond_lr, updates)
            new_precond = optax.apply_updates(precond, updates)
            return (new_precond, opt_state)

          # Use a compiled loop to perform “steps” iterations.
          init_state = (preconditioner, opt_state)
          final_precond, _ = jax.lax.fori_loop(0, steps, update_step, init_state)
          # Safeguard against any numerical issues
          return jax.tree_map(Partial(jnp.nan_to_num, nan=1.0, posinf=1.0, neginf=1.0), final_precond)

      return get_update

  def update_preconditioner(self, params, dataloader, precond_lr, opt_state, precond_batch_size, ema_decay=0, other_model_variables=FrozenDict({})):
    """params is the current weights of the NN"""
    if self._multibatch:
      self._block_structure.update_blocks(
        self._update_preconditioner_fn(self._block_structure.blocks, params, precond_lr, other_model_variables, dataloader,
                                       opt_state), ema_decay)
    else:
      # Accumulate batches until we reach (or exceed) the requested preconditioner batch size
      # First batch to infer layout
      b0 = next(dataloader)
      layout = infer_batch_layout(b0)
      mode = layout["mode"]
      batch_axis = layout["batch_axis"]
      T = layout["T"]  # only used in text mode

      # Now start accumulation with the first batch already taken
      batches = [b0]
      # "Batch size" = size along batch_axis of x
      first_x = jax.tree_util.tree_leaves(b0)[0]
      if not isinstance(first_x, jnp.ndarray):
        first_x = jnp.asarray(first_x)
      current_B = first_x.shape[batch_axis]
      acc_size = int(current_B)

      while acc_size < precond_batch_size:
        b = next(dataloader)
        x_leaf = jax.tree_util.tree_leaves(b)[0]
        if not isinstance(x_leaf, jnp.ndarray):
          x_leaf = jnp.asarray(x_leaf)
        B = x_leaf.shape[batch_axis]
        batches.append(b)
        acc_size += int(B)

      # Concatenate according to layout
      def concat_fn(*xs):
        x0 = xs[0]
        if not isinstance(x0, jnp.ndarray):
          x0 = jnp.asarray(x0)

        if mode == "cv":
          # CV: batch axis always 0 for all arrays
          if x0.ndim >= 1:
            return jnp.concatenate(xs, axis=0)
          else:
            return x0

        else:  # mode == "text"
          # For text:
          # - inputs: [T, B] -> concat along axis=1 (batch axis)
          # - targets: [T*B] -> concat along axis=0
          if x0.ndim >= 2:
            # assume it's something like [T, B, ...] and batch_axis is 1
            return jnp.concatenate(xs, axis=batch_axis)
          elif x0.ndim == 1:
            # flattened targets
            return jnp.concatenate(xs, axis=0)
          else:
            return x0

      acc_batches = jax.tree_util.tree_map(concat_fn, *batches)

      # Clip to exactly precond_batch_size along batch dimension
      def clip_fn(x):
        if not isinstance(x, jnp.ndarray):
          x = jnp.asarray(x)

        if mode == "cv":
          # x: [B, ...] or [B]
          if x.ndim >= 1:
            return x[:precond_batch_size]
          else:
            return x

        else:  # mode == "text"
          if x.ndim >= 2:
            # inputs: [T, B_total] -> [T, precond_batch_size]
            # batch_axis is 1 here
            idx = [slice(None)] * x.ndim
            idx[batch_axis] = slice(0, precond_batch_size)
            return x[tuple(idx)]
          elif x.ndim == 1 and T is not None:
            # targets: [T*B_total] -> [T * precond_batch_size]
            return x[: T * precond_batch_size]
          else:
            return x

      acc_batches = jax.tree_util.tree_map(clip_fn, acc_batches)
      if self._warm_start_precond and not hasattr(self._block_structure, "_memory"):
        _blocks = self._block_structure.blocks
      else:
        _blocks = self._block_structure.reinit_blocks()
      self._block_structure.update_blocks(
        self._update_preconditioner_fn(
          _blocks,
          params,
          precond_lr,
          other_model_variables,
          acc_batches,
          opt_state,
        ),
        ema_decay,
      )

    if not self._allow_grad_inversion and self._block_structure_name in ('scalar', "diagonal"):
      # We clip to (almost) 0 those 2 structures to avoid gradient inversion
      self._block_structure.clip_blocks(min_for_block=1e-8)

    # print(self._block_structure.blocks["layers_2"])
    # print(self._block_structure.blocks)

  def compile_precond_updater(self, params, dataloader, precond_lr, opt_state, precond_batch_size, other_model_variables=FrozenDict({})):
    """For when we want to trigger jax compilation of _update_preconditioner_fn without applying the update"""
    if self._multibatch:
        blocks = self._update_preconditioner_fn(self._block_structure.blocks, params, precond_lr, other_model_variables, dataloader,
                                       opt_state)
    else:
      # Accumulate batches until we reach (or exceed) the requested preconditioner batch size
      # First batch to infer layout
      b0 = next(dataloader)
      layout = infer_batch_layout(b0)
      mode = layout["mode"]
      batch_axis = layout["batch_axis"]
      T = layout["T"]  # only used in text mode

      # Now start accumulation with the first batch already taken
      batches = [b0]
      # "Batch size" = size along batch_axis of x
      first_x = jax.tree_util.tree_leaves(b0)[0]
      if not isinstance(first_x, jnp.ndarray):
        first_x = jnp.asarray(first_x)
      current_B = first_x.shape[batch_axis]
      acc_size = int(current_B)

      while acc_size < precond_batch_size:
        b = next(dataloader)
        x_leaf = jax.tree_util.tree_leaves(b)[0]
        if not isinstance(x_leaf, jnp.ndarray):
          x_leaf = jnp.asarray(x_leaf)
        B = x_leaf.shape[batch_axis]
        batches.append(b)
        acc_size += int(B)

      # Concatenate according to layout
      def concat_fn(*xs):
        x0 = xs[0]
        if not isinstance(x0, jnp.ndarray):
          x0 = jnp.asarray(x0)

        if mode == "cv":
          # CV: batch axis always 0 for all arrays
          if x0.ndim >= 1:
            return jnp.concatenate(xs, axis=0)
          else:
            return x0

        else:  # mode == "text"
          # For text:
          # - inputs: [T, B] -> concat along axis=1 (batch axis)
          # - targets: [T*B] -> concat along axis=0
          if x0.ndim >= 2:
            # assume it's something like [T, B, ...] and batch_axis is 1
            return jnp.concatenate(xs, axis=batch_axis)
          elif x0.ndim == 1:
            # flattened targets
            return jnp.concatenate(xs, axis=0)
          else:
            return x0

      acc_batches = jax.tree_util.tree_map(concat_fn, *batches)

      # Clip to exactly precond_batch_size along batch dimension
      def clip_fn(x):
        if not isinstance(x, jnp.ndarray):
          x = jnp.asarray(x)

        if mode == "cv":
          # x: [B, ...] or [B]
          if x.ndim >= 1:
            return x[:precond_batch_size]
          else:
            return x

        else:  # mode == "text"
          if x.ndim >= 2:
            # inputs: [T, B_total] -> [T, precond_batch_size]
            # batch_axis is 1 here
            idx = [slice(None)] * x.ndim
            idx[batch_axis] = slice(0, precond_batch_size)
            return x[tuple(idx)]
          elif x.ndim == 1 and T is not None:
            # targets: [T*B_total] -> [T * precond_batch_size]
            return x[: T * precond_batch_size]
          else:
            return x

      acc_batches = jax.tree_util.tree_map(clip_fn, acc_batches)
      blocks = self._update_preconditioner_fn(self._block_structure.blocks, params, precond_lr, other_model_variables,
                                     acc_batches, opt_state)

  def expose_blocks(self):
    return self._block_structure.blocks

  def load_blocks(self, saved_blocks):
    self._block_structure.update_blocks(saved_blocks)
