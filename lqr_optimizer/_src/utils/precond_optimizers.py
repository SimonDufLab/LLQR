"""Optax compatible optimizers to be used for solving the preconditioner over a static batch (deterministic problem)"""
from __future__ import annotations
from typing import Callable, NamedTuple, Any, Optional, Tuple, Literal, Dict
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import lax
import optax

PyTree = Any

def _tree_dot(x: PyTree, y: PyTree) -> jnp.ndarray:
    xs, ys = jtu.tree_leaves(x), jtu.tree_leaves(y)
    return sum(jnp.vdot(a, b) for a, b in zip(xs, ys))

def _tree_add(x: PyTree, y: PyTree) -> PyTree:
    return jtu.tree_map(lambda a, b: a + b, x, y)

def _tree_sub(x: PyTree, y: PyTree) -> PyTree:
    return jtu.tree_map(lambda a, b: a - b, x, y)

def _tree_scale(x: PyTree, s: jnp.ndarray) -> PyTree:
    return jtu.tree_map(lambda a: a * s, x)

class NCGState(NamedTuple):
    step: jnp.ndarray
    prev_grad: Optional[PyTree]
    prev_dir: Optional[PyTree]
    prev_alpha: jnp.ndarray
    # for Optax line searches only; ignored by "internal"
    ls_state: Optional[optax.OptState]

# Implementation derived from JAXopt NonlinearCG solver: https://jaxopt.github.io/stable/_autosummary/jaxopt.NonlinearCG.html#
def nonlinear_cg(
    *,
    method: Literal["fr", "pr+", "hs+", "dy", "hz"] = "pr+",
    # --- internal strong-Wolfe knobs ---
    c1: float = 1e-4,
    c2: float = 0.9,
    max_ls: int = 20,
    alpha_init: float = 1.0,
    alpha_increase: float = 1.2,
    alpha_max: float = 1.0,
    # --- restarts ---
    restart_every: Optional[int] = None,
    enforce_descent: bool = True,
    hz_clip: Tuple[float, float] = (-1e3, 1e3),
    # --- choose the line-search backend ---
    linesearch: Literal["internal", "optax_backtracking", "optax_zoom"] = "internal",
    # kwargs forwarded to the chosen Optax linesearch (if used)
    optax_ls_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Nonlinear CG optimizer as an Optax GradientTransformation with selectable line search.

    Pass extra_args={'value_and_grad_fn': lambda params: (loss, grad)}.
    If linesearch is 'optax_backtracking' or 'optax_zoom', we call the corresponding
    Optax transform to scale the search direction p_k -> alpha * p_k using (value, grad, value_fn).
    See Optax L-BFGS example for the exact calling convention.
    """

    # Prepare optional Optax line-search transform in the closure (Python object, not in state)
    optax_ls = None
    restart_every = 0 if restart_every is None else restart_every
    if linesearch == "optax_backtracking":
        # Implements Armijo backtracking (sufficient decrease). :contentReference[oaicite:1]{index=1}
        optax_ls = optax.scale_by_backtracking_linesearch(
            **(optax_ls_kwargs or {})
        )
    elif linesearch == "optax_zoom":
        # Implements strong-Wolfe (sufficient decrease + curvature) via zoom. :contentReference[oaicite:2]{index=2}
        optax_ls = optax.scale_by_zoom_linesearch(
            **(optax_ls_kwargs or {})
        )

    def init_fn(params: PyTree) -> NCGState:
        ls_state = optax_ls.init(params) if optax_ls is not None else None
        return NCGState(
            step=jnp.array(0, dtype=jnp.int32),
            prev_grad=jax.tree_map(jnp.zeros_like, params),
            prev_dir=jax.tree_map(jnp.zeros_like, params),
            prev_alpha=jnp.asarray(alpha_init, dtype=jnp.result_type(float)),
            ls_state=ls_state,
        )

    # --- compact strong-Wolfe (internal) ---
    def strong_wolfe_line_search(x, fx, gx, p, value_and_grad_fn):
        def phi(alpha):
            x_new = jtu.tree_map(lambda a, d: a + alpha * d, x, p)
            f_new, g_new = value_and_grad_fn(x_new)
            gtp_new = _tree_dot(g_new, p)
            return f_new, g_new, gtp_new, x_new

        gtp0 = _tree_dot(gx, p)
        alpha = jnp.minimum(alpha_increase, alpha_max)
        best = None

        for _ in range(max_ls):
            f1, g1, gtp1, x1 = phi(alpha)
            # Armijo
            if (f1 <= fx + c1 * alpha * gtp0):
                best = (f1, g1, gtp1, x1)
                # Curvature
                if jnp.abs(gtp1) <= -c2 * gtp0:
                    return best
                alpha = jnp.minimum(alpha * alpha_increase, alpha_max)
            else:
                alpha = alpha * 0.5

        if best is not None:
            return best
        return phi(jnp.asarray(1e-6))

    def _beta(method, g, g_prev, p_prev):
        y = _tree_sub(g, g_prev)
        gg = jnp.maximum(_tree_dot(g, g), 1e-30)
        gg_prev = jnp.maximum(_tree_dot(g_prev, g_prev), 1e-30)
        py = jnp.maximum(_tree_dot(p_prev, y), 1e-30)
        if method == "fr":
            b = gg / gg_prev
        elif method == "pr+":
            b = jnp.maximum(_tree_dot(g, y) / gg_prev, 0.0)
        elif method == "hs+":
            b = jnp.maximum(_tree_dot(g, y) / py, 0.0)
        elif method == "dy":
            b = gg / py
        elif method == "hz":
            denom = py
            t = _tree_sub(y, _tree_scale(p_prev, (2.0 * _tree_dot(y, y) / jnp.maximum(denom, 1e-30))))
            b = _tree_dot(t, g) / jnp.maximum(denom, 1e-30)
            b = jnp.clip(b, hz_clip[0], hz_clip[1])
        else:
            raise ValueError(f"Unknown method {method}")
        return b

    def _compute_dir_and_restart(state, g, method):
        def init_branch(_):
            p0 = _tree_scale(g, -1.0)
            return p0, jnp.bool_(False)

        def cg_branch(_):
            beta = _beta(method, g, state.prev_grad, state.prev_dir)
            p = _tree_add(_tree_scale(g, -1.0), _tree_scale(state.prev_dir, beta))

            # Predicate: should we restart? (descent violation)
            dot_ge_0 = (_tree_dot(p, g) >= 0)
            restart_flag = jnp.logical_and(enforce_descent, dot_ge_0)

            # If you want to ALSO reset the direction when restarting, uncomment:
            # p = lax.select(restart_flag, _tree_scale(g, -1.0), p)

            return p, restart_flag

        return lax.cond(state.step == 0, init_branch, cg_branch, operand=None)

    def update_fn(grads: PyTree, state: NCGState, params: Optional[PyTree] = None, *, value_and_grad_fn=None, **extra_args):
        del grads, extra_args
        if params is None:
            raise ValueError("nonlinear_cg.update expects current params.")
        if value_and_grad_fn is None:
            raise ValueError("Pass extra_args['value_and_grad_fn']: params -> (loss, grad).")

        # value_and_grad_fn: Callable[[PyTree], Tuple[jnp.ndarray, PyTree]] = extra_args['value_and_grad_fn']

        # Current value and gradient (use provided grads for consistency with Optax API)
        fx, g = value_and_grad_fn(params)

        # Build conjugate direction
        # restart = False
        # if state.step==0:
        #     p = _tree_scale(g, -1.0)
        # else:
        #     beta = _beta(method, g, state.prev_grad, state.prev_dir)
        #     p = _tree_add(_tree_scale(g, -1.0), _tree_scale(state.prev_dir, beta))
        #     if enforce_descent and (_tree_dot(p, g) >= 0):
        #         restart = True
        p, restart = _compute_dir_and_restart(state, g, method)
        periodic_restart = lax.cond(
          restart_every > 0,
          lambda re: ((state.step + 1) % re) == 0,
          lambda re: jnp.bool_(False),
          restart_every,
        )

        # if restart or (restart_every is not None and ((int(state.step) + 1) % restart_every == 0)):
        #     p = _tree_scale(g, -1.0)
        do_restart = jnp.logical_or(restart, periodic_restart)
        p = lax.cond(
          do_restart,
          lambda args: args[0],  # take restart_dir
          lambda args: args[1],  # keep p
          (_tree_scale(g, -1.0), p),
        )

        if linesearch == "internal":
            # Use embedded strong-Wolfe
            f1, g1, gtp1, x1 = strong_wolfe_line_search(params, fx, g, p, value_and_grad_fn)
            updates = _tree_sub(x1, params)
            new_ls_state = state.ls_state
        else:
            # Use Optax linesearch transform to scale p -> alpha * p
            # It expects: update(grads=direction, state, params, value=fx, grad=g, value_fn=fun)
            value_fn = lambda w: value_and_grad_fn(w)[0]
            scaled_update, new_ls_state = optax_ls.update(
                p, state.ls_state, params, value=fx, grad=g, value_fn=value_fn
            )
            updates = scaled_update  # already α * p

            # For convenience we can also get the post-step grad to keep CG history fresh:
            # (optional; CG doesn't strictly need it if direction uses current g)
            # But for parity with internal branch, compute g1 at new params:
            x1 = jtu.tree_map(lambda a, du: a + du, params, updates)
            f1, g1 = value_and_grad_fn(x1)

        new_state = NCGState(
            step=state.step + 1,
            prev_grad=g1,
            prev_dir=p,
            prev_alpha=state.prev_alpha,  # kept for possible future heuristics
            ls_state=new_ls_state,
        )
        return updates, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
