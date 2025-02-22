from argon.struct import struct

import argon.transforms as agt
import argon.numpy as npx
import argon.scipy as spx
import argon.typing as atyp

@struct(frozen=True)
class RelaxedSinkhornState:
    iteration: atyp.Array
    err: atyp.Array
    log_K: atyp.Array
    log_q_p1_min: atyp.Array
    log_q_p1_max: atyp.Array
    log_q_p2_min: atyp.Array
    log_q_p2_max: atyp.Array
    log_q_total: atyp.Array | None

@struct(frozen=True)
class RelaxedSinkhorn:
    relaxation_factor: float = 0.0
    max_iterations: int = 100,
    epsilon: float = 1e-3
    tolerance: float = 1e-3,
    enforce_total_mass: bool = True

    @agt.jit
    def solve(self, log_p1, log_p2, C):
        # normalize the log_p1 and log_p2 just in case
        log_p1 = log_p1 - spx.special.logsumexp(log_p1)
        log_p2 = log_p2 - spx.special.logsumexp(log_p2)
        log_p1_min, log_p1_max = log_p1 - self.relaxation_factor, log_p1 + self.relaxation_factor
        log_p2_min, log_p2_max = log_p2 - self.relaxation_factor, log_p2 + self.relaxation_factor

        K = -C / self.epsilon
        K = K - spx.special.logsumexp(K, axis=-1, keepdims=True)
        state = RelaxedSinkhornState(
            npx.zeros((), dtype=npx.uint32),
            K, npx.zeros_like(K), npx.zeros_like(K),
            npx.zeros_like(K), npx.zeros_like(K),
            npx.zeros_like(K) if self.enforce_total_mass else None,
        )
        def _step(state : RelaxedSinkhornState) -> RelaxedSinkhornState:
            (log_K, log_q_p1_min, log_q_p1_max, log_q_p2_min, 
                    log_q_p2_max, log_q_total) = (
                state.log_K, state.log_q_p1_min, state.log_q_p1_max, 
                state.log_q_p2_min, state.log_q_p2_max, state.log_q_total
            )
            log_K_prev = log_K
            def project(log_K, log_q, log_p_max, upper: bool = False, axis=0):
                assert axis in (0, 1)
                # elementwise-scaleK = K * q
                log_K_prev = log_K
                log_K = log_K + log_q
                # Project onto constraint sum[K, axis] <= log_q_p_max
                log_u = log_p_max - spx.special.logsumexp(K, axis=0)
                if upper: log_u = npx.where(log_u > 0, 0, log_u)
                else: log_u = npx.where(log_u < 0, 0, log_u)
                log_K = log_K + (log_u[None, :] if axis == 0 else log_u[:,None])
                log_q = log_q + log_K_prev - log_K
                return log_K, log_q
            log_K, log_q_p1_min = project(log_K, log_q_p1_min, log_p1_min, False, 0)
            log_K, log_q_p1_max = project(log_K, log_q_p1_max, log_p1_max, True, 0)
            log_K, log_q_p2_min = project(log_K, log_q_p2_min, log_p2_min, False, 0)
            log_K, log_q_p2_max = project(log_K, log_q_p2_max, log_p2_max, True, 0)
            # lastly, project onto the total mass constraint
            if self.enforce_total_mass:
                log_K_pre_total = log_K
                log_K = log_K + log_q_total
                log_K = log_K - spx.special.logsumexp(log_K)
                log_q_total = log_q_total + log_K_pre_total - log_K
            err = npx.sum(npx.square(log_K - log_K_prev))
            return RelaxedSinkhornState(
                state.iteration + 1, err,
                log_K, log_q_p1_min, log_q_p1_max, 
                log_q_p2_min, log_q_p2_max,
                log_q_total
            )
        final = agt.while_loop(
            lambda s: npx.logical_or(s.iteration > self.max_iterations, s.err < self.tolerance),
            _step, state
        )
        final_cost = npx.sum(npx.exp(final.log_K) * C)
        return final_cost, final.log_K