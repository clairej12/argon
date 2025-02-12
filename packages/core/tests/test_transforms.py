# import argon.transforms as agt
# import argon.numpy as npx
# import jax

# def test_jit():
#     trace_cnt = 0
#     def foo(a, b):
#         nonlocal trace_cnt
#         trace_cnt = trace_cnt + 1
#         return a + b

#     assert agt.jit(foo) is agt.jit(foo)
#     foo = agt.jit(foo)
#     with jax.log_compiles():
#         a = foo(npx.zeros((10,)), npx.ones((10,)))
#         b = foo(npx.zeros((10,)), npx.ones((10,)))
#     assert npx.all(a == b)
#     assert trace_cnt == 1
