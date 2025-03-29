import argon.solver.termination
from argon.struct import struct

import argon.transforms as agt
import argon.numpy as npx
import argon.scipy as spx
import argon.typing as atyp

import argon.solver
import jax

from ..solver.r2hpdhg import r2HPDHG

@struct(frozen=True)
class MpaxSolver:
    relaxation_factor: float | None = None
    enforce_total_mass: bool = True

    @agt.jit
    def solve(self, C, p1=None, p2=None):
        if p1 is None:
            p1 = npx.ones(C.shape[0])/C.shape[0]
        if p2 is None:
            p2 = npx.ones(C.shape[1])/C.shape[1]
        solver = r2HPDHG(eps_abs=1e-4, eps_rel=1e-4)
        # flatten the cost matrix
        c = npx.reshape(C, -1)
        # matrices which sum over the columns or rows
        row_sum = npx.stack([
            npx.reshape(npx.zeros_like(C).at[i,:].set(1),-1)
            for i in range(C.shape[0])
        ])
        col_sum = npx.stack([
            npx.reshape(npx.zeros_like(C).at[:,i].set(1),-1)
            for i in range(C.shape[1])
        ])
        eq_constr = []
        ineq_constr = [] # Gx >= h
        if self.relaxation_factor is None:
            # sum over the columns or rows of the original C
            eq_constr.append((row_sum, p1))
            eq_constr.append((col_sum, p2))
        else:
            ineq_constr.append((row_sum, p1/self.relaxation_factor))
            ineq_constr.append((-row_sum, -p1*self.relaxation_factor))
            ineq_constr.append((col_sum, p2/self.relaxation_factor))
            ineq_constr.append((-col_sum, -p2*self.relaxation_factor))

        if self.enforce_total_mass:
            eq_constr.append((npx.ones_like(c)[None,:], npx.ones((1,))))

        if len(eq_constr) == 0:
            A, b = npx.zeros((0, len(c))), npx.zeros((0,))
        else:
            A, b = npx.concatenate(list(a for (a, _) in eq_constr)), npx.concatenate(list(b for (_, b) in eq_constr))

        if len(ineq_constr) == 0:
            G, h = npx.zeros((0, len(c))), npx.zeros((0,))
        else:
            G, h = npx.concatenate(list(g for (g, _) in ineq_constr)), npx.concatenate(list(b for (g, b) in ineq_constr))

        lp = argon.solver.create_lp(c, A, b, G, h, 
            npx.zeros_like(c), npx.ones_like(c), use_sparse_matrix=False)
        solution = solver.optimize(lp)
        P = npx.reshape(solution.primal_solution, C.shape)
        cost = solution.primal_objective
        return cost, P