"""Sample-separable exact second-order LLQR operator actions."""

import numbers
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class SampleSeparableSecondOrderActions(NamedTuple):
  q: Callable
  mt: Callable
  m: Callable
  r: Callable


def _tree_vdot(lhs, rhs):
  if jax.tree_util.tree_structure(lhs) != jax.tree_util.tree_structure(rhs):
    raise ValueError("Expected matching pytree structures for tree_vdot")
  lhs_leaves = jax.tree_util.tree_leaves(lhs)
  rhs_leaves = jax.tree_util.tree_leaves(rhs)
  return sum(jnp.vdot(lhs_leaf, rhs_leaf) for lhs_leaf, rhs_leaf in zip(lhs_leaves, rhs_leaves))


def _validate_second_order_chunk_size(second_order_chunk_size):
  if (
      not isinstance(second_order_chunk_size, numbers.Integral)
      or isinstance(second_order_chunk_size, bool)
      or int(second_order_chunk_size) <= 0
  ):
    raise ValueError("second_order_chunk_size must be a positive integer")
  return int(second_order_chunk_size)


def _zeros_like_tree(tree):
  return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _add_trees(lhs, rhs):
  return jax.tree_util.tree_map(jnp.add, lhs, rhs)


def _add_weighted_tree(lhs, rhs, weight):
  return jax.tree_util.tree_map(lambda x, y: x + weight.astype(y.dtype) * y, lhs, rhs)


def _move_batch_axis_to_front(tree, batch_axis):
  def move_leaf(leaf):
    if leaf.ndim <= batch_axis:
      raise ValueError(
        f"Sample-separable second-order operator expected batch axis {batch_axis} "
        f"to exist on every leaf, got shape {leaf.shape}"
      )
    return jnp.moveaxis(leaf, batch_axis, 0)

  return jax.tree_util.tree_map(move_leaf, tree)


def _move_output_batch_axis_to_front(tree, batch_axis, batch_size):
  def move_leaf(leaf):
    if leaf.ndim > batch_axis and leaf.shape[batch_axis] == batch_size:
      return jnp.moveaxis(leaf, batch_axis, 0)
    if batch_axis == 0 and leaf.ndim >= 1 and leaf.shape[0] % batch_size == 0:
      per_sample = leaf.shape[0] // batch_size
      return leaf.reshape((batch_size, per_sample) + leaf.shape[1:])
    raise ValueError(
      "Sample-separable second-order operator could not align output cotangent "
      f"shape {leaf.shape} with batch size {batch_size} on axis {batch_axis}."
    )

  return jax.tree_util.tree_map(move_leaf, tree)


def _pad_and_chunk_tree(tree, chunk_size):
  leaves = jax.tree_util.tree_leaves(tree)
  if not leaves:
    raise ValueError("Expected a non-empty pytree for sample-separable chunking")
  batch_size = int(leaves[0].shape[0])
  num_chunks = max(1, (batch_size + chunk_size - 1) // chunk_size)
  padded_batch_size = num_chunks * chunk_size
  pad = padded_batch_size - batch_size

  def pad_and_chunk_leaf(leaf):
    if leaf.shape[0] != batch_size:
      raise ValueError("Sample-separable chunking expected a consistent leading batch dimension")
    if pad:
      pad_width = [(0, pad)] + [(0, 0)] * (leaf.ndim - 1)
      leaf = jnp.pad(leaf, pad_width)
    return leaf.reshape((num_chunks, chunk_size) + leaf.shape[1:])

  mask = jnp.concatenate(
    [jnp.ones(batch_size, dtype=jnp.float32), jnp.zeros(pad, dtype=jnp.float32)],
    axis=0,
  ).reshape((num_chunks, chunk_size))
  return jax.tree_util.tree_map(pad_and_chunk_leaf, tree), mask


def _prepare_chunked_state_and_output(state_batch, output_cotangent, batch_axis, chunk_size):
  moved_state = _move_batch_axis_to_front(state_batch, batch_axis)
  batch_size = int(jax.tree_util.tree_leaves(moved_state)[0].shape[0])
  moved_output = _move_output_batch_axis_to_front(output_cotangent, batch_axis, batch_size)
  state_chunks, mask = _pad_and_chunk_tree(moved_state, chunk_size)
  output_chunks, _ = _pad_and_chunk_tree(moved_output, chunk_size)
  return moved_state, state_chunks, output_chunks, mask


def _chunked_tangent(tangent, batch_axis, chunk_size):
  return _pad_and_chunk_tree(_move_batch_axis_to_front(tangent, batch_axis), chunk_size)[0]


def _expand_single_sample_tree(tree, batch_axis):
  return jax.tree_util.tree_map(lambda leaf: jnp.expand_dims(leaf, axis=batch_axis), tree)


def _extract_single_sample_output_tree(tree, batch_axis, reference_sample_output):
  def extract_leaf(leaf, reference):
    if leaf.shape == reference.shape:
      return leaf
    if leaf.ndim > batch_axis and leaf.shape[batch_axis] == 1:
      squeezed = jnp.squeeze(leaf, axis=batch_axis)
      if squeezed.shape == reference.shape:
        return squeezed
    raise ValueError(
      "Sample-separable second-order operator could not align single-sample "
      f"output shape {leaf.shape} with cotangent shape {reference.shape}."
    )

  return jax.tree_util.tree_map(extract_leaf, tree, reference_sample_output)


def _single_sample_apply(apply_batched, batch_axis):
  def apply_single(params, sample_state, sample_output_cotangent):
    batched_state = _expand_single_sample_tree(sample_state, batch_axis)
    batched_output = apply_batched(params, batched_state)
    return _extract_single_sample_output_tree(batched_output, batch_axis, sample_output_cotangent)

  return apply_single


def _single_sample_state_only_apply(apply_state_batched, batch_axis):
  def apply_single(sample_state, sample_output_cotangent):
    batched_state = _expand_single_sample_tree(sample_state, batch_axis)
    batched_output = apply_state_batched(batched_state)
    return _extract_single_sample_output_tree(batched_output, batch_axis, sample_output_cotangent)

  return apply_single


def _apply_chunk_mask(tree, chunk_mask):
  def mask_leaf(leaf):
    broadcast_shape = (chunk_mask.shape[0],) + (1,) * (leaf.ndim - 1)
    return leaf * chunk_mask.reshape(broadcast_shape).astype(leaf.dtype)

  return jax.tree_util.tree_map(mask_leaf, tree)


def _restore_front_batched_tree_from_chunks(chunk_tree, batch_size, batch_axis):
  reshaped = jax.tree_util.tree_map(
    lambda leaf: leaf.reshape((leaf.shape[0] * leaf.shape[1],) + leaf.shape[2:])[:batch_size],
    chunk_tree,
  )
  return jax.tree_util.tree_map(lambda leaf: jnp.moveaxis(leaf, 0, batch_axis), reshaped)


def _accumulate_param_output(initial_accumulator, scan_inputs, sample_action):
  def chunk_body(accumulator, chunk_inputs):
    state_chunk, output_chunk, tangent_chunk, chunk_mask = chunk_inputs

    def sample_body(sample_accumulator, sample_inputs):
      sample_state, sample_output, sample_tangent, sample_mask = sample_inputs
      sample_contribution = sample_action(sample_state, sample_output, sample_tangent)
      return _add_weighted_tree(sample_accumulator, sample_contribution, sample_mask), None

    chunk_accumulator, _ = jax.lax.scan(
      sample_body,
      _zeros_like_tree(initial_accumulator),
      (state_chunk, output_chunk, tangent_chunk, chunk_mask),
    )
    return _add_trees(accumulator, chunk_accumulator), None

  accumulated, _ = jax.lax.scan(chunk_body, initial_accumulator, scan_inputs)
  return accumulated


def build_sample_separable_second_order_actions(apply_batched, params, state_batch, output_cotangent, *,
                                                batch_axis, second_order_chunk_size, damping=0.0):
  """Build exact sample-separable Q, M^T, M, and R actions.

  Parameter-side actions scan over samples and reduce immediately into one
  parameter-shaped pytree, avoiding batch-of-parameter intermediates.
  """
  chunk_size = _validate_second_order_chunk_size(second_order_chunk_size)
  moved_state, state_chunks, output_chunks, mask = _prepare_chunked_state_and_output(
    state_batch, output_cotangent, batch_axis, chunk_size
  )
  batch_size = int(jax.tree_util.tree_leaves(moved_state)[0].shape[0])
  apply_single = _single_sample_apply(apply_batched, batch_axis)

  def sample_hamiltonian(current_params, sample_state, sample_output_cotangent):
    return _tree_vdot(apply_single(current_params, sample_state, sample_output_cotangent),
                      sample_output_cotangent)

  grad_params = jax.grad(sample_hamiltonian, argnums=0)
  grad_state = jax.grad(sample_hamiltonian, argnums=1)

  def q(state_tangent):
    tangent_chunks = _chunked_tangent(state_tangent, batch_axis, chunk_size)

    def sample_q_action(sample_state, sample_output, sample_tangent):
      _, hessian_action = jax.jvp(
        lambda state: grad_state(params, state, sample_output),
        (sample_state,),
        (sample_tangent,),
      )
      return hessian_action

    def chunk_action(state_chunk, output_chunk, tangent_chunk, chunk_mask):
      chunk_outputs = jax.vmap(sample_q_action)(state_chunk, output_chunk, tangent_chunk)
      return _apply_chunk_mask(chunk_outputs, chunk_mask)

    chunk_outputs = jax.vmap(chunk_action)(state_chunks, output_chunks, tangent_chunks, mask)
    return _restore_front_batched_tree_from_chunks(chunk_outputs, batch_size, batch_axis)

  def mt(param_tangent):
    def sample_mt_action(sample_state, sample_output):
      _, mixed_action = jax.jvp(
        lambda current_params: grad_state(current_params, sample_state, sample_output),
        (params,),
        (param_tangent,),
      )
      return mixed_action

    def chunk_action(state_chunk, output_chunk, chunk_mask):
      chunk_outputs = jax.vmap(sample_mt_action)(state_chunk, output_chunk)
      return _apply_chunk_mask(chunk_outputs, chunk_mask)

    chunk_outputs = jax.vmap(chunk_action)(state_chunks, output_chunks, mask)
    return _restore_front_batched_tree_from_chunks(chunk_outputs, batch_size, batch_axis)

  def m(state_tangent):
    tangent_chunks = _chunked_tangent(state_tangent, batch_axis, chunk_size)

    def sample_m_action(sample_state, sample_output, sample_tangent):
      _, mixed_action = jax.jvp(
        lambda state: grad_params(params, state, sample_output),
        (sample_state,),
        (sample_tangent,),
      )
      return mixed_action

    return _accumulate_param_output(
      _zeros_like_tree(params),
      (state_chunks, output_chunks, tangent_chunks, mask),
      sample_m_action,
    )

  def r(param_tangent):
    def sample_r_action(sample_state, sample_output, _unused_tangent):
      del _unused_tangent
      _, hessian_action = jax.jvp(
        lambda current_params: grad_params(current_params, sample_state, sample_output),
        (params,),
        (param_tangent,),
      )
      return hessian_action

    accumulated = _accumulate_param_output(
      _zeros_like_tree(params),
      (state_chunks, output_chunks, state_chunks, mask),
      sample_r_action,
    )
    return jax.tree_util.tree_map(
      lambda action, tangent: action + damping * tangent,
      accumulated,
      param_tangent,
    )

  return SampleSeparableSecondOrderActions(q=q, mt=mt, m=m, r=r)


def build_sample_separable_state_only_action(apply_state_batched, state_batch, output_cotangent, *,
                                             batch_axis, second_order_chunk_size):
  """Build an exact sample-separable passive-state Hessian action."""
  chunk_size = _validate_second_order_chunk_size(second_order_chunk_size)
  moved_state, state_chunks, output_chunks, mask = _prepare_chunked_state_and_output(
    state_batch, output_cotangent, batch_axis, chunk_size
  )
  batch_size = int(jax.tree_util.tree_leaves(moved_state)[0].shape[0])
  apply_single = _single_sample_state_only_apply(apply_state_batched, batch_axis)

  def sample_hamiltonian(sample_state, sample_output_cotangent):
    return _tree_vdot(apply_single(sample_state, sample_output_cotangent), sample_output_cotangent)

  grad_state = jax.grad(sample_hamiltonian, argnums=0)

  def sample_q_action(sample_state, sample_output, sample_tangent):
    _, hessian_action = jax.jvp(
      lambda state: grad_state(state, sample_output),
      (sample_state,),
      (sample_tangent,),
    )
    return hessian_action

  def q(state_tangent):
    tangent_chunks = _chunked_tangent(state_tangent, batch_axis, chunk_size)

    def chunk_action(state_chunk, output_chunk, tangent_chunk, chunk_mask):
      chunk_outputs = jax.vmap(sample_q_action)(state_chunk, output_chunk, tangent_chunk)
      return _apply_chunk_mask(chunk_outputs, chunk_mask)

    chunk_outputs = jax.vmap(chunk_action)(state_chunks, output_chunks, tangent_chunks, mask)
    return _restore_front_batched_tree_from_chunks(chunk_outputs, batch_size, batch_axis)

  return q
