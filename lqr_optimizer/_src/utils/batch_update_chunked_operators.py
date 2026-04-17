"""Chunked batch-update Hamiltonian operators."""

import jax
import jax.numpy as jnp


def _tree_vdot(lhs, rhs):
  lhs_leaves = jax.tree_util.tree_leaves(lhs)
  rhs_leaves = jax.tree_util.tree_leaves(rhs)
  if len(lhs_leaves) != len(rhs_leaves):
    raise ValueError("Expected matching pytree structures for tree_vdot")
  return sum(jnp.vdot(lhs_leaf, rhs_leaf) for lhs_leaf, rhs_leaf in zip(lhs_leaves, rhs_leaves))


def _move_batch_axis_to_front(tree, batch_axis):
  def move_leaf(leaf):
    if leaf.ndim <= batch_axis:
      raise ValueError(
        f"Chunked batch update operator expected batch axis {batch_axis} to exist on every leaf, got shape {leaf.shape}"
      )
    return jnp.moveaxis(leaf, batch_axis, 0)

  return jax.tree_util.tree_map(move_leaf, tree)


def _pad_and_chunk_tree(tree, chunk_size):
  leaves = jax.tree_util.tree_leaves(tree)
  if not leaves:
    raise ValueError("Expected a non-empty pytree for chunked batch update")
  batch_size = int(leaves[0].shape[0])
  num_chunks = max(1, (batch_size + chunk_size - 1) // chunk_size)
  padded_batch_size = num_chunks * chunk_size
  pad = padded_batch_size - batch_size

  def pad_and_chunk_leaf(leaf):
    if leaf.shape[0] != batch_size:
      raise ValueError("Chunked batch update operator expected a consistent leading batch dimension")
    if pad:
      pad_width = [(0, pad)] + [(0, 0)] * (leaf.ndim - 1)
      leaf = jnp.pad(leaf, pad_width)
    return leaf.reshape((num_chunks, chunk_size) + leaf.shape[1:])

  mask = jnp.concatenate(
    [jnp.ones(batch_size, dtype=jnp.float32), jnp.zeros(pad, dtype=jnp.float32)],
    axis=0,
  ).reshape((num_chunks, chunk_size))
  return jax.tree_util.tree_map(pad_and_chunk_leaf, tree), mask


def _expand_single_sample_tree(tree, batch_axis):
  return jax.tree_util.tree_map(lambda leaf: jnp.expand_dims(leaf, axis=batch_axis), tree)


def _squeeze_single_sample_tree(tree, batch_axis):
  def squeeze_leaf(leaf):
    if leaf.ndim <= batch_axis:
      raise ValueError(
        f"Chunked batch update operator expected output batch axis {batch_axis} to exist on every leaf, got shape {leaf.shape}"
      )
    return jnp.squeeze(leaf, axis=batch_axis)

  return jax.tree_util.tree_map(squeeze_leaf, tree)


def _prepare_chunk_inputs(state_batch, output_cotangent_batch, batch_axis, batch_chunk_size):
  moved_state = _move_batch_axis_to_front(state_batch, batch_axis)
  moved_output = _move_batch_axis_to_front(output_cotangent_batch, batch_axis)
  state_chunks, mask = _pad_and_chunk_tree(moved_state, batch_chunk_size)
  output_chunks, _ = _pad_and_chunk_tree(moved_output, batch_chunk_size)
  return state_chunks, output_chunks, mask


def _apply_chunk_mask(tree, chunk_mask):
  def mask_leaf(leaf):
    broadcast_shape = (chunk_mask.shape[0],) + (1,) * (leaf.ndim - 1)
    return leaf * chunk_mask.reshape(broadcast_shape).astype(leaf.dtype)

  return jax.tree_util.tree_map(mask_leaf, tree)


def _restore_front_batched_tree_from_chunks(chunk_tree, batch_size, batch_axis):
  leaves = jax.tree_util.tree_leaves(chunk_tree)
  if not leaves:
    raise ValueError("Expected a non-empty chunk tree when restoring batch outputs")

  reshaped = jax.tree_util.tree_map(
    lambda leaf: leaf.reshape((leaf.shape[0] * leaf.shape[1],) + leaf.shape[2:])[:batch_size],
    chunk_tree,
  )
  return jax.tree_util.tree_map(lambda leaf: jnp.moveaxis(leaf, 0, batch_axis), reshaped)


def _single_sample_apply(apply_batched, batch_axis):
  def apply_single(parameters, sample_state):
    batched_state = _expand_single_sample_tree(sample_state, batch_axis)
    batched_output = apply_batched(parameters, batched_state)
    return _squeeze_single_sample_tree(batched_output, batch_axis)

  return apply_single


def _single_sample_state_only_apply(apply_batched, batch_axis):
  def apply_single(sample_state):
    batched_state = _expand_single_sample_tree(sample_state, batch_axis)
    batched_output = apply_batched(batched_state)
    return _squeeze_single_sample_tree(batched_output, batch_axis)

  return apply_single


def build_chunked_control_only_hessian_operator(apply_batched, layer_params, fixed_input_state,
                                                output_cotangent, *, batch_axis, batch_chunk_size,
                                                damping):
  """Chunked cached exact operator for a control-only parameter Hessian action."""
  if batch_chunk_size is None:
    raise ValueError("batch_chunk_size must be set for chunked batch update operators")

  state_chunks, output_chunks, mask = _prepare_chunk_inputs(
    fixed_input_state, output_cotangent, batch_axis, batch_chunk_size
  )
  apply_single = _single_sample_apply(apply_batched, batch_axis)

  def sample_hamiltonian(parameters, sample_state, sample_output_cotangent):
    return _tree_vdot(apply_single(parameters, sample_state), sample_output_cotangent)

  grad_params = jax.grad(sample_hamiltonian, argnums=0)

  def sample_r_action(parameters, control_tangent, sample_state, sample_output_cotangent):
    _, hvp = jax.jvp(
      lambda params: grad_params(params, sample_state, sample_output_cotangent),
      (parameters,),
      (control_tangent,),
    )
    return hvp

  def k_u(control_tangent):
    flat_control = jnp.ravel(control_tangent)

    def scan_body(accumulator, scan_inputs):
      state_chunk, output_chunk, chunk_mask = scan_inputs
      chunk_contribs = jax.vmap(
        lambda sample_state, sample_output: sample_r_action(
          layer_params, flat_control, sample_state, sample_output
        )
      )(state_chunk, output_chunk)
      chunk_sum = jnp.tensordot(chunk_mask.astype(chunk_contribs.dtype), chunk_contribs, axes=1)
      return accumulator + chunk_sum, None

    accumulated, _ = jax.lax.scan(
      scan_body,
      jnp.zeros_like(layer_params),
      (state_chunks, output_chunks, mask),
    )
    return accumulated + damping * flat_control

  return k_u


def build_chunked_joint_param_output_operator(apply_batched, layer_params, layer_state,
                                              output_cotangent, *, batch_axis, batch_chunk_size,
                                              damping):
  """Chunked cached exact parameter-output action for R_i u + M_i x."""
  if batch_chunk_size is None:
    raise ValueError("batch_chunk_size must be set for chunked batch update operators")

  state_chunks, output_chunks, mask = _prepare_chunk_inputs(
    layer_state, output_cotangent, batch_axis, batch_chunk_size
  )
  apply_single = _single_sample_apply(apply_batched, batch_axis)

  def sample_hamiltonian(parameters, sample_state, sample_output_cotangent):
    return _tree_vdot(apply_single(parameters, sample_state), sample_output_cotangent)

  grad_params = jax.grad(sample_hamiltonian, argnums=0)

  def sample_r_action(parameters, control_tangent, sample_state, sample_output_cotangent):
    _, hvp = jax.jvp(
      lambda params: grad_params(params, sample_state, sample_output_cotangent),
      (parameters,),
      (control_tangent,),
    )
    return hvp

  def sample_m_action(parameters, sample_state, sample_output_cotangent, sample_state_tangent):
    _, mixed_action = jax.jvp(
      lambda state: grad_params(parameters, state, sample_output_cotangent),
      (sample_state,),
      (sample_state_tangent,),
    )
    return mixed_action

  def k_u(control_tangent, state_tangent):
    flat_control = jnp.ravel(control_tangent)
    tangent_chunks, _ = _pad_and_chunk_tree(
      _move_batch_axis_to_front(state_tangent, batch_axis), batch_chunk_size
    )

    def scan_body(accumulator, scan_inputs):
      state_chunk, output_chunk, tangent_chunk, chunk_mask = scan_inputs
      r_chunk = jax.vmap(
        lambda sample_state, sample_output: sample_r_action(
          layer_params, flat_control, sample_state, sample_output
        )
      )(state_chunk, output_chunk)
      m_chunk = jax.vmap(
        lambda sample_state, sample_output, sample_tangent: sample_m_action(
          layer_params, sample_state, sample_output, sample_tangent
        )
      )(state_chunk, output_chunk, tangent_chunk)
      weighted_mask = chunk_mask.astype(r_chunk.dtype)
      chunk_sum = (
        jnp.tensordot(weighted_mask, r_chunk, axes=1)
        + jnp.tensordot(weighted_mask, m_chunk, axes=1)
      )
      return accumulator + chunk_sum, None

    accumulated, _ = jax.lax.scan(
      scan_body,
      jnp.zeros_like(layer_params),
      (state_chunks, output_chunks, tangent_chunks, mask),
    )
    return accumulated + damping * flat_control

  return k_u


def build_chunked_joint_state_output_operator(apply_batched, layer_params, layer_state,
                                              output_cotangent, *, batch_axis, batch_chunk_size):
  """Chunked cached exact state-output action for M_i^T u + Q_i x."""
  if batch_chunk_size is None:
    raise ValueError("batch_chunk_size must be set for chunked batch update operators")

  batch_size = int(jax.tree_util.tree_leaves(_move_batch_axis_to_front(layer_state, batch_axis))[0].shape[0])
  state_chunks, output_chunks, mask = _prepare_chunk_inputs(
    layer_state, output_cotangent, batch_axis, batch_chunk_size
  )
  apply_single = _single_sample_apply(apply_batched, batch_axis)

  def sample_hamiltonian(parameters, sample_state, sample_output_cotangent):
    return _tree_vdot(apply_single(parameters, sample_state), sample_output_cotangent)

  grad_state = jax.grad(sample_hamiltonian, argnums=1)

  def sample_q_action(sample_state, sample_output_cotangent, sample_state_tangent):
    _, hvp = jax.jvp(
      lambda state: grad_state(layer_params, state, sample_output_cotangent),
      (sample_state,),
      (sample_state_tangent,),
    )
    return hvp

  def sample_m_transpose_action(parameters, sample_state, sample_output_cotangent, control_tangent):
    _, mixed_action = jax.jvp(
      lambda params: grad_state(params, sample_state, sample_output_cotangent),
      (parameters,),
      (control_tangent,),
    )
    return mixed_action

  def k_x(control_tangent, state_tangent):
    flat_control = jnp.ravel(control_tangent)
    tangent_chunks, _ = _pad_and_chunk_tree(
      _move_batch_axis_to_front(state_tangent, batch_axis), batch_chunk_size
    )

    def chunk_action(state_chunk, output_chunk, tangent_chunk, chunk_mask):
      q_chunk = jax.vmap(
        lambda sample_state, sample_output, sample_tangent: sample_q_action(
          sample_state, sample_output, sample_tangent
        )
      )(state_chunk, output_chunk, tangent_chunk)
      mt_chunk = jax.vmap(
        lambda sample_state, sample_output: sample_m_transpose_action(
          layer_params, sample_state, sample_output, flat_control
        )
      )(state_chunk, output_chunk)
      return _apply_chunk_mask(
        jax.tree_util.tree_map(jnp.add, q_chunk, mt_chunk),
        chunk_mask,
      )

    chunk_outputs = jax.vmap(chunk_action)(state_chunks, output_chunks, tangent_chunks, mask)
    return _restore_front_batched_tree_from_chunks(chunk_outputs, batch_size, batch_axis)

  return k_x


def build_chunked_state_only_hessian_operator(apply_batched, layer_state,
                                              output_cotangent, *, batch_axis, batch_chunk_size):
  """Chunked cached exact state-only action for passive Q_i x."""
  if batch_chunk_size is None:
    raise ValueError("batch_chunk_size must be set for chunked batch update operators")

  batch_size = int(jax.tree_util.tree_leaves(_move_batch_axis_to_front(layer_state, batch_axis))[0].shape[0])
  state_chunks, output_chunks, mask = _prepare_chunk_inputs(
    layer_state, output_cotangent, batch_axis, batch_chunk_size
  )
  apply_single = _single_sample_state_only_apply(apply_batched, batch_axis)

  def sample_hamiltonian(sample_state, sample_output_cotangent):
    return _tree_vdot(apply_single(sample_state), sample_output_cotangent)

  grad_state = jax.grad(sample_hamiltonian, argnums=0)

  def sample_q_action(sample_state, sample_output_cotangent, sample_state_tangent):
    _, hvp = jax.jvp(
      lambda state: grad_state(state, sample_output_cotangent),
      (sample_state,),
      (sample_state_tangent,),
    )
    return hvp

  def k_x(state_tangent):
    tangent_chunks, _ = _pad_and_chunk_tree(
      _move_batch_axis_to_front(state_tangent, batch_axis), batch_chunk_size
    )

    def chunk_action(state_chunk, output_chunk, tangent_chunk, chunk_mask):
      q_chunk = jax.vmap(
        lambda sample_state, sample_output, sample_tangent: sample_q_action(
          sample_state, sample_output, sample_tangent
        )
      )(state_chunk, output_chunk, tangent_chunk)
      return _apply_chunk_mask(q_chunk, chunk_mask)

    chunk_outputs = jax.vmap(chunk_action)(state_chunks, output_chunks, tangent_chunks, mask)
    return _restore_front_batched_tree_from_chunks(chunk_outputs, batch_size, batch_axis)

  return k_x
