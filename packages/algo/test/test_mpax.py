
import argon.numpy as npx
import argon.scipy as spx

import cvxpy as cp

from argon.ott.mpax import MpaxSolver

def relaxed_cvxpy(C, log_p1, log_p2, relaxation_factor, epsilon=1e-10):
    p1 = npx.exp(log_p1)
    p2 = npx.exp(log_p2)
    M = cp.Variable(C.shape)
    cost = cp.sum(cp.multiply(M, C))
    problem = cp.Problem(
        cp.Minimize(cost),
        [
            p1/relaxation_factor <= cp.sum(M, axis=1),
            cp.sum(M, axis=1) <= p1*relaxation_factor,
            p2/relaxation_factor <= cp.sum(M, axis=0),
            cp.sum(M, axis=0) <= p2*relaxation_factor,
            M >= 0, cp.sum(M) == 1,
        ] if relaxation_factor is not None else [
            p1 == cp.sum(M, axis=1),
            p2 == cp.sum(M, axis=0),
            M >= 0, cp.sum(M) == 1,
        ]
    )
    problem.solve(solver=cp.CLARABEL)
    return cost.value, M.value

def test_simple():
    a = npx.array([
        [-0.1],
        [0.],
        [0.1],
        [0.8]
    ])
    b = npx.array([
        [0.1],
        [0.8]
    ])
    C = npx.sum(npx.square(a[:, None, :] - b[None, :, :]), -1)
    log_p1 = npx.zeros(a.shape[0])
    log_p1 = log_p1 - spx.special.logsumexp(log_p1)
    log_p2 = npx.zeros(b.shape[0])
    log_p2 = log_p2 - spx.special.logsumexp(log_p2)
    for r in [None, 1.1, 2.0, 4.0, 8.0]:
        mpax = MpaxSolver(r)
        gt, P_star = relaxed_cvxpy(C, log_p1, log_p2, r)
        cost, P = mpax.solve(C, npx.exp(log_p1), npx.exp(log_p2))
        print("--------------------")
        print("relaxation factor: ", r)
        print("cost gt:", gt, " cost: ", cost)
        print("P_hat", P)
        print("P_star", P_star)
        assert npx.allclose(cost, gt, atol=1e-4)