import argon.numpy as npx
import argon.scipy as spx

import cvxpy as cp

from argon.ott.dykstra import RelaxedDykstra

def relaxed_cvxpy(C, log_p1, log_p2, relaxation_factor, epsilon=1e-10):
    p1 = npx.exp(log_p1)
    p2 = npx.exp(log_p2)
    M = cp.Variable(C.shape)
    cost = cp.sum(cp.multiply(M, C))
    entropy = cp.sum(cp.entr(M))
    problem = cp.Problem(
        cp.Minimize(cost - epsilon * entropy),
        [
            p1/relaxation_factor <= cp.sum(M, axis=1),
            cp.sum(M, axis=1) <= p1*relaxation_factor,
            p2/relaxation_factor <= cp.sum(M, axis=0),
            cp.sum(M, axis=0) <= p2*relaxation_factor,
            M >= 0, cp.sum(M) == 1,
        ]
    )
    problem.solve(solver=cp.CLARABEL)
    return cost.value, M.value

    # log_C = cp.log(C)
    # log_P = cp.Variable(C.shape)
    # P1 = cp.Variable(C.shape[0])
    # P2 = cp.Variable(C.shape[1])

    # log_relaxation_factor = npx.log(relaxation_factor)
    # cost = cp.log_sum_exp(log_P + log_C) 
    # entropy = 0.# cp.sum(-cp.multiply(cp.exp(log_P), log_P))
    # problem = cp.Problem(
    #     cp.Minimize(cost - epsilon * entropy),
    #     [
    #         cp.sum(cp.exp(log_P), axis=1) == P1,
    #         cp.sum(cp.exp(log_P), axis=0) == P2,

    #         npx.exp(log_p1 - log_relaxation_factor) <= P1,
    #         P1 <= npx.exp(log_p1 + log_relaxation_factor),
    #         npx.exp(log_p2 - log_relaxation_factor) <= P2,
    #         P2 <= npx.exp(log_p2 + log_relaxation_factor),
    #         cp.sum(cp.exp(log_P)) == 1
    #     ]
    # )
    # problem.solve(solver=cp.CLARABEL)
    # return cost.value

def test_simple():
    return
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
    for r in [1.1, 2.0, 4.0, 8.0]:
        relaxed_sinkhorn = RelaxedDykstra(r, epsilon=1e-4)
        gt, P_star = relaxed_cvxpy(C, log_p1, log_p2, r, epsilon=1e-3)
        cost, P = relaxed_sinkhorn.solve(C, npx.exp(log_p1), npx.exp(log_p2))
        print("cost gt:", gt, " cost: ", cost)
        print("P_hat", P)
        print("P_star", P_star)