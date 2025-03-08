from argon.struct import struct, field

import argon.numpy as npx
import argon.transforms as agt
import argon.tree

import jax.flatten_util
import jax
import chex

from functools import partial
from typing import Optional, TypeVar, Callable

Sample = TypeVar("Sample")

@struct(frozen=True)
class DDPMSchedule:
    """A schedule for a DDPM model. Implements https://arxiv.org/abs/2006.11239. """
    betas: jax.Array
    """ The betas for the DDPM. This corresponds to the forward process:

            q(x_t | x_{t-1}) = N(x_t | sqrt(1 - beta_t)x_{t-1}, beta_t I)

       Note that betas[1] corresponds to beta_1 and betas[T] corresponds to beta_T.
       betas[0] should always be 0.
    """
    alphas: jax.Array
    """ 1 - betas """
    alphas_cumprod: jax.Array
    """ The alphabar_t for the DDPM. alphabar_t = prod_(i=1)^t (1 - beta_i)
    Note that:

        alphas_cumprod[0] = alphabar_0 = 1

        alphas_cumprod[1] = alphabar_1 = alpha_1 = (1 - beta_1)

    """
    prediction_type: str = "epsilon"
    """ The type of prediction to make. If "epsilon", the model will predict the noise.
    If "sample", the model will predict the sample.
    """
    clip_sample_range: Optional[float] = None
    """ Whether to clip the predicted denoised sample in the reverse process to the range [-clip_sample_range, clip_sample_range].
       If None, no clipping is done.
    """

    @staticmethod
    def make_from_betas(betas: jax.Array, **kwargs) -> "DDPMSchedule":
        alphas = 1 - betas
        alphas_cumprod = npx.cumprod(alphas)
        return DDPMSchedule(
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            betas=betas,
            **kwargs
        )

    @staticmethod
    def make_from_alpha_bars(alphas_cumprod : jax.Array, max_beta : float = 1., **kwargs) -> "DDPMSchedule":
        """ Makes a DDPM schedule from the alphas_cumprod. """
        t1 = npx.roll(alphas_cumprod, 1, 0)
        t1 = t1.at[0].set(1)
        t2 = alphas_cumprod
        betas = 1 - t2/t1
        betas = npx.clip(betas, 0, max_beta)
        return DDPMSchedule.make_from_betas(
            betas=betas,
            **kwargs)

    @staticmethod
    def make_linear(num_timesteps : int, beta_start : float = 0.0001, beta_end : float = 0.1,
                    **kwargs) -> "DDPMSchedule":
        beta_end = npx.clip(beta_end, 0, 1)
        beta_start = npx.clip(beta_start, 0, 1)
        betas = npx.linspace(beta_start, beta_end, num_timesteps, dtype=npx.float32)
        betas = npx.concatenate((npx.zeros((1,), npx.float32), betas))
        """ Makes a linear schedule for the DDPM. """
        return DDPMSchedule.make_from_betas(
            betas=betas,
            **kwargs
        )

    @staticmethod
    def make_rescaled(num_timesteps, schedule, **kwargs):
        """ Rescales a schedule to have a different number of timesteps. """
        xs = npx.linspace(0, 1, num_timesteps + 1, dtype=npx.float32)
        old_xs = npx.linspace(0, 1, schedule.num_steps + 1, dtype=npx.float32)
        new_alphas_cumprod = npx.interp(xs, old_xs, schedule.alphas_cumprod)
        return DDPMSchedule.make_from_alpha_bars(new_alphas_cumprod, **kwargs)

    @staticmethod
    def make_squaredcos_cap_v2(num_timesteps : int, order: float = 2, offset : float | None = None, max_beta : float = 0.999, **kwargs) -> "DDPMSchedule":
        """ Makes a squared cosine schedule for the DDPM.
            Uses alpha_bar(t) = cos^2((t + 0.008) / 1.008 * pi / 2)
            i.e. the alpha_bar is a squared cosine function.

            This means a large amount of noise is added at the start, with
            decreasing noise added as time goes on.
        """
        t = npx.arange(num_timesteps, dtype=npx.float32)/num_timesteps
        offset = offset if offset is not None else 0.008
        def alpha_bar(t):
            t = (t + offset) / (1 + offset)
            return npx.pow(npx.cos(t * npx.pi / 2), order)
        # make the first timestep start at index 1
        alpha_bars = npx.concatenate(
            (npx.ones((1,), dtype=t.dtype), jax.vmap(alpha_bar)(t)),
        axis=0)
        # alpha_bars = alpha_bars.at[-1].set(0)
        return DDPMSchedule.make_from_alpha_bars(alpha_bars, max_beta=max_beta, **kwargs)

    @staticmethod
    def make_scaled_linear_schedule(num_timesteps : int,
                beta_start : float = 0.0001, beta_end : float = 0.02, **kwargs):
        """ Makes a scaled linear (i.e quadratic) schedule for the DDPM. """
        betas = npx.concatenate((npx.zeros((1,), dtype=npx.float32),
            npx.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=npx.float32)**2),
        axis=-1)
        return DDPMSchedule.make_from_betas(
            betas=betas,
            **kwargs
        )

    @property
    def reverse_variance(self):
        alpha_bars = self.alphas_cumprod[1:]
        alpha_bars_prev = self.alphas_cumprod[:-1]
        betas = self.betas[1:]
        variance = (1 - alpha_bars_prev) / (1 - alpha_bars) * betas
        variance = npx.concatenate((npx.zeros((1,)), variance), axis=0)
        return variance

    @property
    def num_steps(self) -> int:
        """ The number of steps T in the schedule. Note that betas has T+1 elements since beta_0 = 0."""
        return self.betas.shape[0] - 1

    @agt.jit
    def forward_trajectory(self, rng_key : jax.Array, sample : Sample) -> Sample:
        """ Given an x_0, returns a trajectory of samples x_0, x_1, x_2, ..., x_T.
            Args:
                rng_key: The random key to use for the noise.
                sample: The initial sample x_0.

            Returns:
                A pytree of the same structure as sample, with a batch dimension of t+1.
                The 0th index corresponds to x_0, the 1st index corresponds to x_1, and so on.
        """
        sample_flat, unflatten = jax.flatten_util.ravel_pytree(sample)
        unfaltten_vmap = agt.vmap(unflatten)
        noise_flat = jax.random.normal(rng_key, (self.num_steps + 1,) + sample_flat.shape)
        # sum up the noise added at each step
        def scan_fn(prev_noise_accum, noise_beta):
            noise, alpha, beta = noise_beta
            noise_accum = npx.sqrt(alpha)*prev_noise_accum + noise*npx.sqrt(beta)
            return noise_accum, noise_accum
        noise_flat = agt.scan(scan_fn, npx.zeros_like(noise_flat[0]),
                                  (noise_flat, self.alphas, self.betas))[1]
        noisy_flat = noise_flat + npx.sqrt(self.alphas_cumprod[:,None])*sample_flat[None,:]
        noise = unfaltten_vmap(noise_flat)
        noisy = unfaltten_vmap(noisy_flat)
        return noisy, noise

    # This will do the noising
    # forward process
    # will return noisy_sample, noise_eps, model_output
    @agt.jit
    def add_noise(self, rng_key : jax.Array, sample : Sample,
                  timestep : jax.Array) -> tuple[Sample, Sample, Sample]:
        """ Samples q(x_t | x_0). Returns a tuple containing (noisy_sample, noise, model_output).
        where model_output is based on the value of ``prediction_type``.

        Args:
            rng_key: The random key to use for the noise.
            sample: The initial sample x_0. Can be an arbitrary pytree.
            timestep: The timestep t to sample at.

        Returns:
            A tuple containing (noisy_sample, noise_epsilon, model_output).
            In the same structure as sample.
        """
        sqrt_alphas_prod = npx.sqrt(self.alphas_cumprod[timestep])
        sqrt_one_minus_alphas_prod = npx.sqrt(1 - self.alphas_cumprod[timestep])
        sample_flat, unflatten = jax.flatten_util.ravel_pytree(sample)
        noise_flat = jax.random.normal(rng_key, sample_flat.shape, dtype=sample_flat.dtype)
        noisy_flat = sqrt_alphas_prod * sample_flat + \
            sqrt_one_minus_alphas_prod*noise_flat
        noisy = unflatten(noisy_flat)
        noise = unflatten(noise_flat)
        if self.prediction_type == "epsilon":
            return noisy, noise, noise
        elif self.prediction_type == "sample":
            return noisy, noise, sample
        else:
            raise ValueError("Not supported prediction type")

    @agt.jit
    def add_sub_noise(self, rng_key : jax.Array,
                      sub_sample : Sample, sub_timestep : jax.Array,
                      timestep : jax.Array) -> tuple[Sample, Sample, Sample]:
        """ Like add_noise, but assumes that sub_sample is x_{sub_timestep}
        rather than x_0. Note that timestep > sub_timestep or the behavior is undefined!
        """
        alphas_shifted = npx.concatenate((npx.ones((1,)), self.alphas_cumprod), axis=-1)
        alphas_prod = self.alphas_cumprod[timestep] / alphas_shifted[sub_timestep]
        sqrt_alphas_prod = npx.sqrt(alphas_prod)
        sqrt_one_minus_alphas_prod = npx.sqrt(1 - alphas_prod)
        sample_flat, unflatten = jax.flatten_util.ravel_pytree(sub_sample)
        noise_flat = jax.random.normal(rng_key, sample_flat.shape, dtype=sample_flat.dtype)
        noisy_flat = sqrt_alphas_prod * sample_flat + \
            sqrt_one_minus_alphas_prod*noise_flat

        sqrt_one_minus_alphas_full_prod = npx.sqrt(1- self.alphas_cumprod[timestep])
        scaling = sqrt_one_minus_alphas_prod / sqrt_one_minus_alphas_full_prod
        noisy = unflatten(noisy_flat)
        noise = unflatten(noise_flat)
        scaled_noise = unflatten(scaling*noise_flat)
        if self.prediction_type == "epsilon":
            return noisy, noise, scaled_noise
        elif self.prediction_type == "sample":
            return noisy, noise, sub_sample
        else:
            raise ValueError("Not supported prediction type")

    # returns E[x_0 | model_output, current sample]
    @agt.jit
    def denoised_from_output(self, noised_sample : Sample, t : jax.Array, model_output : Sample) -> Sample:
        """ Returns E[x_0 | x_t] as computed by the model_output
        based on the value of ``prediction_type``
        """
        alpha_prod_t = self.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        sample_flat, unflatten = jax.flatten_util.ravel_pytree(noised_sample)
        model_output_flat, _ = jax.flatten_util.ravel_pytree(model_output)
        if self.prediction_type == "epsilon":
            pred_sample = (sample_flat - beta_prod_t ** (0.5) * model_output_flat) / npx.maximum(1e-6, alpha_prod_t ** (0.5))
        elif self.prediction_type == "sample":
            pred_sample = model_output_flat
        else:
            raise ValueError("Not supported prediction type")

        if self.clip_sample_range is not None:
            pred_sample = npx.clip(pred_sample, -self.clip_sample_range, self.clip_sample_range)
        return unflatten(pred_sample)

    @agt.jit
    def epsilon_from_output(self, noised_sample : Sample, t : jax.Array, model_output : Sample) -> Sample:
        """ Returns the noise epsilon = x_t - E[x_0 | x_t] as computed by the model_output. """
        if self.prediction_type == "epsilon":
            return model_output
        elif self.prediction_type == "sample":
            chex.assert_trees_all_equal_shapes_and_dtypes(noised_sample, model_output)
            sample_flat, unflatten = jax.flatten_util.ravel_pytree(noised_sample)
            model_output_flat, _ = jax.flatten_util.ravel_pytree(model_output)
            alpha_prod_t = self.alphas_cumprod[t]
            one_minus_alpha_prod_t = 1 - alpha_prod_t
            return unflatten(
                (sample_flat - npx.sqrt(alpha_prod_t) * model_output_flat) /
                        npx.maximum(1e-6, npx.sqrt(one_minus_alpha_prod_t))
            )

    @agt.jit
    def output_from_denoised(self, noised_sample : Sample, t : jax.Array, denoised_sample : Sample) -> Sample:
        """Returns the output a model should give given an x_t to denoise to x_0."""
        if self.prediction_type == "sample":
            return denoised_sample
        elif self.prediction_type == "epsilon":
            sqrt_alphas_prod = npx.sqrt(self.alphas_cumprod[t])
            sqrt_one_minus_alphas_prod = npx.sqrt(1 - self.alphas_cumprod[t])
            # noised_sample = sqrt_alphas_prod * denoised + sqrt_one_minus_alphas_prod * noise
            # therefore noise = (sample - sqrt_alphas_prod * denoised) / sqrt_one_minus_alphas_prod
            nosied_sample_flat, unflatten = jax.flatten_util.ravel_pytree(noised_sample)
            denoised_sample_flat, _ = jax.flatten_util.ravel_pytree(denoised_sample)
            noise = (nosied_sample_flat - sqrt_alphas_prod * denoised_sample_flat) / sqrt_one_minus_alphas_prod
            return unflatten(noise)

    @agt.jit
    def compute_denoised(self, noised_sample : Sample, t : jax.Array, data_batch : Sample, data_mask : jax.Array = None) -> Sample:
        """Computes the true E[x_0 | x_t] given a batch of x_0's."""
        noised_sample_flat, unflatten = jax.flatten_util.ravel_pytree(noised_sample)
        data_batch_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(data_batch)
        # compute the mean
        sqrt_alphas_prod = npx.sqrt(self.alphas_cumprod[t])
        one_minus_alphas_prod = 1 - self.alphas_cumprod[t]

        # forward diffusion equation for diffusing t timesteps:
        # noised_sample = sqrt_alphas_prod * denoised + sqrt_one_minus_alphas_prod * noise

        noise = (noised_sample_flat[None,:] - sqrt_alphas_prod * data_batch_flat)
        # the magnitude of the noise added
        noise_sqr = npx.sum(noise**2, axis=-1)
        # p(x_t | x_0) prop exp(-1/2(x - mu)^2/sigma^2)
        log_likelihood = -0.5*noise_sqr / npx.maximum(one_minus_alphas_prod, 1e-5)
        likelihood = jax.nn.softmax(log_likelihood, where=data_mask)
        likelihood = likelihood*data_mask if data_mask is not None else likelihood
        # # p(x_0 | x_t) = p(x_t | x_0) p(x_0) / p(x_t)
        # # where p(x_0) is uniform, so we effectively just to normalize the log likelihood
        # # over the x_0's
        # log_likelihood = log_likelihood - jax.scipy.special.logsumexp(log_likelihood, axis=0)
        # # log_likehood contains log p(x_0 | x_t) for all x_0's in the dataset

        # this is equivalent to the log-likelihood (up to a constant factor)
        denoised = npx.sum(likelihood[:,None]*data_batch_flat, axis=0)
        return unflatten(denoised)

    # This does a reverse process step
    @agt.jit
    def reverse_step(self, rng_key : jax.Array, sample : Sample,
                     timestep : jax.Array, delta_steps: jax.Array, model_output : Sample,
                     eta : jax.Array = 1.0) -> Sample:
        """ Does a reverse step of the DDPM given a particular model output. Given x_t returns x_{t-delta_steps}. """
        chex.assert_trees_all_equal_shapes_and_dtypes(sample, model_output)
        sample_flat, unflatten = jax.flatten_util.ravel_pytree(sample)
        model_output_flat, _ = jax.flatten_util.ravel_pytree(model_output)

        t = timestep
        prev_t = timestep - delta_steps

        beta_t = self.betas[t]
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]

        pred_sample = self.denoised_from_output(sample_flat, t, model_output_flat)
        pred_epsilon = self.epsilon_from_output(sample_flat, t, model_output_flat)

        variance = (eta**2) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * beta_t

        # original DDPM equatins
        # pred_original_sample_coeff = npx.sqrt(alpha_prod_t_prev) * beta_t / beta_prod_t
        # current_sample_coeff = npx.sqrt(alpha_t) * beta_prod_t_prev / beta_prod_t

        orig_sample_coeff = npx.sqrt(alpha_prod_t_prev)
        noise_coeff = npx.sqrt(1 - alpha_prod_t_prev - variance)
        pred_prev_sample = orig_sample_coeff * pred_sample + noise_coeff * pred_epsilon

        sigma = npx.sqrt(variance)
        noise = sigma*jax.random.normal(rng_key, pred_prev_sample.shape, pred_prev_sample.dtype)
        return unflatten(pred_prev_sample + noise)

    @agt.jit
    def sample(self, rng_key : jax.Array,
                    model : Callable[[jax.Array, Sample, jax.Array], Sample],
                    sample_structure: Sample,
                    *,
                    eta : float = 1.0,
                    start_time : Optional[int] = None,
                    final_time : Optional[int] = None,
                    # Note: this is not the number of steps to take (which is determined by start_time - final_time)
                    num_steps : Optional[int] = None,
                    start_sample : Optional[Sample] = None,
                    trajectory : bool = False):
        """ Runs the reverse process, given a denoiser model, for a number of steps. """
        if start_time is None: start_time = self.num_steps
        if final_time is None: final_time = 0
        if num_steps is None: num_steps = start_time - final_time
        step_ratio = (start_time - final_time) / num_steps
        # sample initial noise
        if start_sample is not None:
            random_sample = start_sample
        else:
            flat_structure, unflatten = argon.tree.ravel_pytree(argon.tree.map(lambda x: npx.zeros_like(x), sample_structure))
            random_sample = unflatten(jax.random.normal(rng_key, flat_structure.shape, flat_structure.dtype))
        # If we want to return the trajectory, do a scan. In that case, num_steps must be statically known
        timesteps = (npx.arange(0, num_steps + 1) * step_ratio + final_time).round()[::-1].astype(npx.int32)
        curr_timesteps, prev_timesteps = timesteps[:-1], timesteps[1:]

        @agt.scan(in_axes=(None, None, 0, 0, agt.Carry))
        def step(self, model, curr_t, prev_t, carry):
            x_t, rng_key = carry
            m_rng, s_rng, n_rng = jax.random.split(rng_key, 3)
            model_output = model(m_rng, x_t, curr_t)
            x_prev = self.reverse_step(s_rng, x_t, curr_t, curr_t - prev_t, 
                                    model_output, eta=eta)
            return (x_prev, n_rng), x_prev

        (sample, _), traj = step(self, model, 
                            curr_timesteps, prev_timesteps, 
                            (random_sample, rng_key))

        traj = argon.tree.map(
            lambda x, s: npx.concatenate((x, s[None]), axis=0), 
            traj, random_sample
        )
        if trajectory:
            return sample, traj
        else:
            return sample


    def loss(self, rng_key : jax.Array,
             model : Callable[[jax.Array, Sample, jax.Array], Sample],
             sample : Sample, t : Optional[jax.Array] = None, *,
             target_model : Callable[[jax.Array, Sample, jax.Array], Sample] | None = None,
             target_clip : float | None = None,
             model_has_state_updates=False):
        """
        Computes the loss for the DDPM model.
        If t is None, a random t in [1, T] is chosen.
        """
        s_rng, t_rng, m_rng, tar_rng = jax.random.split(rng_key, 4)
        if t is None:
            t = jax.random.randint(t_rng, (), 0, self.num_steps) + 1
        noised_sample, _, target = self.add_noise(s_rng, sample, t)
        pred = model(m_rng, noised_sample, t)
        if model_has_state_updates:
            pred, state = pred
        if target_model is not None:
            target = target_model(tar_rng, noised_sample, t)
        chex.assert_trees_all_equal_shapes_and_dtypes(pred, target)

        pred_flat = jax.flatten_util.ravel_pytree(pred)[0]
        target_flat = jax.flatten_util.ravel_pytree(target)[0]
        if target_clip is not None and target_clip > 0:
            target_flat = target_flat.clip(-target_clip, target_clip)
        loss = npx.mean((pred_flat - target_flat)**2)
        if model_has_state_updates:
            return loss, state
        else:
            return loss

import argon.graph