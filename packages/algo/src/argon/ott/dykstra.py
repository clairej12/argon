from argon.struct import struct

import argon.transforms as agt
import argon.numpy as npx
import argon.scipy as spx
import argon.typing as atyp
import jax

@struct(frozen=True)
class RelaxedDykstraState:
    iteration: atyp.Array
    err: atyp.Array
    K: atyp.Array
    q_p1_min: atyp.Array
    q_p1_max: atyp.Array
    q_p2_min: atyp.Array
    q_p2_max: atyp.Array
    q_total: atyp.Array | None

@struct(frozen=True)
class RelaxedDykstraLogState:
    iteration: atyp.Array
    err: atyp.Array
    log_K: atyp.Array
    log_q_p1_min: atyp.Array
    log_q_p1_max: atyp.Array
    log_q_p2_min: atyp.Array
    log_q_p2_max: atyp.Array
    log_q_total: atyp.Array | None

@struct(frozen=True)
class RelaxedDykstra:
    relaxation_factor: float = 1.0
    max_iterations: int = 100
    epsilon: float = 1e-3
    tolerance: float = 1e-3
    enforce_total_mass: bool = True
    use_log: bool = True

    @agt.jit
    def _solve_log(self, C, log_p1=None, log_p2=None):
        if log_p1 is None:
            log_p1 = npx.zeros(C.shape[0])
        if log_p2 is None:
            log_p2 = npx.zeros(C.shape[1])
        
        # normalize the log_p1 and log_p2 just in case
        log_p1 = log_p1 - spx.special.logsumexp(log_p1)
        log_p2 = log_p2 - spx.special.logsumexp(log_p2)
        log_rf = npx.log(self.relaxation_factor)
        log_p1_min, log_p1_max = log_p1 - log_rf, log_p1 + log_rf
        log_p2_min, log_p2_max = log_p2 - log_rf, log_p2 + log_rf

        K = -C / self.epsilon
        if self.enforce_total_mass:
            K = K - spx.special.logsumexp(K)
        state = RelaxedDykstraLogState(
            npx.zeros((), dtype=npx.uint32),
            npx.array(npx.inf),
            K, npx.zeros_like(K), npx.zeros_like(K),
            npx.zeros_like(K), npx.zeros_like(K),
            npx.zeros_like(K) if self.enforce_total_mass else None,
        )
        def _step(state : RelaxedDykstraLogState) -> RelaxedDykstraLogState:
            (log_K, log_q_p1_min, log_q_p1_max, log_q_p2_min, 
                    log_q_p2_max, log_q_total) = (
                state.log_K, state.log_q_p1_min, state.log_q_p1_max, 
                state.log_q_p2_min, state.log_q_p2_max, state.log_q_total
            )
            log_K_prev = log_K
            def project(log_K, log_q, log_p_max, upper: bool = False, axis=0):
                assert axis in (0, 1)
                # elementwise-scale K = K * q
                log_K_prev = log_K
                log_K = log_K + log_q
                # Project onto constraint sum[K, axis] <= log_q_p_max
                log_u = log_p_max - spx.special.logsumexp(K, axis=axis)
                if upper: log_u = npx.where(log_u > 0, 0, log_u)
                else: log_u = npx.where(log_u < 0, 0, log_u)
                log_K = log_K + (log_u[None, :] if axis == 0 else log_u[:,None])
                log_q = log_q + log_K_prev - log_K
                return log_K, log_q
            log_K, log_q_p1_max = project(log_K, log_q_p1_max, log_p1_max, True, 1)
            log_K, log_q_p1_min = project(log_K, log_q_p1_min, log_p1_min, False, 1)
            log_K, log_q_p2_max = project(log_K, log_q_p2_max, log_p2_max, True, 0)
            log_K, log_q_p2_min = project(log_K, log_q_p2_min, log_p2_min, False, 0)
            # lastly, project onto the total mass constraint
            if self.enforce_total_mass:
                log_K_pre_total = log_K
                log_K = log_K + log_q_total
                log_K = log_K - spx.special.logsumexp(log_K)
                log_q_total = log_q_total + log_K_pre_total - log_K
            err = npx.sum(npx.square(log_K - log_K_prev))
            return RelaxedDykstraLogState(
                state.iteration + 1, err,
                log_K, log_q_p1_min, log_q_p1_max, 
                log_q_p2_min, log_q_p2_max,
                log_q_total
            )
        final = agt.while_loop(
            lambda s: npx.logical_and(
                s.iteration < self.max_iterations,
                s.err >= self.tolerance
            ), _step, state
        )
        final_cost = npx.sum(npx.exp(final.log_K) * C)
        return final_cost, npx.exp(final.log_K)

    @agt.jit
    def _solve(self, C, p1=None, p2=None):
        if p1 is None:
            p1 = npx.ones(C.shape[0])
        if p2 is None:
            p2 = npx.ones(C.shape[1])
        
        # normalize the log_p1 and log_p2 just in case
        p1 = p1 / npx.sum(p1)
        p2 = p2 / npx.sum(p2)
        p1_min, p1_max = p1 / self.relaxation_factor, p1 * self.relaxation_factor
        p2_min, p2_max = p2 / self.relaxation_factor, p2 * self.relaxation_factor

        K = npx.exp(-C / self.epsilon)
        if self.enforce_total_mass:
            K = K / npx.sum(K)
        state = RelaxedDykstraState(
            npx.zeros((), dtype=npx.uint32),
            npx.array(npx.inf),
            K, npx.ones_like(K), npx.ones_like(K),
            npx.ones_like(K), npx.ones_like(K),
            npx.ones_like(K) if self.enforce_total_mass else None,
        )
        def _step(state : RelaxedDykstraState) -> RelaxedDykstraState:
            (K, q_p1_min, q_p1_max, q_p2_min, 
                    q_p2_max, q_total) = (
                state.K, state.q_p1_min, state.q_p1_max, 
                state.q_p2_min, state.q_p2_max, state.q_total
            )
            K_prev = K
            def project(K, q, p_constr, upper: bool = False, axis=0):
                assert axis in (0, 1)
                K_prev = K
                K = K * q 
                # Project onto constraint sum[K, axis] <= log_q_p_max
                u = p_constr / npx.sum(K, axis=axis)
                if upper: u = npx.where(u > 1, 1, u)
                else: u = npx.where(u < 1, 1, u)
                K = K * (u[None, :] if axis == 0 else u[:,None])
                q = q * K_prev / K
                return K, q
            # jax.debug.print("K0 {}", K)
            K, q_p1_max = project(K, q_p1_max, p1_max, True, 1)
            # jax.debug.print("K1 {}", K)
            K, q_p1_min = project(K, q_p1_min, p1_min, False, 1)
            # jax.debug.print("K2 {}", K)
            K, q_p2_max = project(K, q_p2_max, p2_max, True, 0)
            # jax.debug.print("K3 {}", K)
            K, q_p2_min = project(K, q_p2_min, p2_min, False, 0)
            # jax.debug.print("K4 {}", K)
            # lastly, project onto the total mass constraint
            if self.enforce_total_mass:
                K_pre_total = K
                K = K * q_total
                K = K  /  npx.sum(K)
                q_total = q_total * K_pre_total / K
            err = npx.sum(npx.square(K - K_prev))
            # jax.debug.print("err {} K {}", err, K)
            return RelaxedDykstraState(
                state.iteration + 1, err,
                K, q_p1_min, q_p1_max, 
                q_p2_min, q_p2_max,
                q_total
            )
        final = agt.while_loop(
            lambda s: npx.logical_and(
                s.iteration < self.max_iterations,
                s.err >= self.tolerance
            ), _step, state
        )
        final_cost = npx.sum(final.K * C)
        return final_cost, final.K
    
    def solve(self, C, p1=None, p2=None):
        if self.use_log:
            return self._solve_log(C, 
                npx.log(p1) if p1 is not None else None,
                npx.log(p2) if p2 is not None else None)
        else:
            return self._solve(C, p1, p2)
