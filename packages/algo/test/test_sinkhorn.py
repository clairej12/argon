import argon.numpy as npx
import argon.scipy as spx

from argon.ott.sinkhorn import RelaxedSinkhorn

def test_simple():
    log_p1 = npx.array([0.5, 0.5])
    log_p1 = log_p1 - spx.special.logsumexp(log_p1)
    log_p2 = npx.array([0.5, 0.5])
    log_p2 = log_p2 - spx.special.logsumexp(log_p2)

    relaxed_sinkhorn = RelaxedSinkhorn(0.1)
    pass