# import argon.numpy as jnp
# import argon.random
# from argon.diffusion.ddpm import DDPMSchedule

# def test_schedule_mode():
#     schedule = DDPMSchedule.make_squaredcos_cap_v2(100, prediction_type="epsilon")
#     sample = argon.random.uniform(argon.random.key(40), (32, 32, 3))

#     noised, _, target = schedule.add_noise(argon.random.key(42), sample, 50)
#     test_target = schedule.output_from_denoised(noised, 50, sample)
#     norm = jnp.linalg.norm(target - test_target)
#     assert norm < 1e-4