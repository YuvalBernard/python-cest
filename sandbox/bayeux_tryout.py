# %%
from functools import partial
import optax
import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import jax.random
import lmfit
import numpy as np
import numpyro
import numpyro.distributions as dist
from blackjax.progress_bar import gen_scan_fn
from numpyro.infer.util import initialize_model

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
numpyro.set_host_device_count(8)
import matplotlib.pyplot as plt

from solve_bloch_mcconnell import gen_spectrum_symbolic


@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4, 5], signature="()->()")  # offsets
def batch_gen_spectrum_symbolic(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    return gen_spectrum_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp)


offsets = np.linspace(-6, 6, 75)
powers = np.array([0.5, 2, 5])
b0 = 7.0
gamma = 267.522
tp = 10
args = (offsets, powers, b0, gamma, tp)

model_parameters = lmfit.Parameters()
model_parameters.add("R1a", 0.33, True, 0.1, 1)
model_parameters.add("R2a", 0.67, True, 0.1, 1)
model_parameters.add("dwa", 0, True, -1, 1)
model_parameters.add("R1b", 1, True, 0.1, 10)
model_parameters.add("R2b", 66.67, True, 1, 100)
model_parameters.add("kb", 30, True, 1, 1000)
model_parameters.add("fb", 0.007, True, 1e-6, 1e-2)
model_parameters.add("dwb", 3.5, True, 3, 4)

for par in list(model_parameters.keys()):
    if model_parameters[par].vary:
        model_parameters[par].prior = dist.TruncatedNormal(
            (model_parameters[par].min + model_parameters[par].max) / 2,
            (model_parameters[par].max - model_parameters[par].min) / 6,
            low=model_parameters[par].min,
            high=model_parameters[par].max,
        )

model_args = (offsets, powers, b0, gamma, tp)
fit_pars = list(model_parameters.valuesdict().values())

Z = batch_gen_spectrum_symbolic(fit_pars, offsets, powers, b0, gamma, tp)
rng = np.random.default_rng()
sigma = 0.02
data = rng.normal(Z, sigma)


def probabilistic_model(model_parameters, model_args, data, method) -> None:
    offsets, powers, b0, gamma, tp = model_args
    fit_pars = jnp.asarray(
        [
            numpyro.sample(model_parameters[par].name, model_parameters[par].prior)
            if model_parameters[par].vary
            else model_parameters[par].value
            for par in list(model_parameters.keys())
        ]
    )
    sigma = numpyro.sample("sigma", dist.Gamma(2, 100))
    model_pred = method(fit_pars, offsets, powers, b0, gamma, tp)
    numpyro.sample("obs", dist.Normal(model_pred, sigma), obs=data)


num_chains = 4
num_warmup = 2500
num_sample = 2500

rng_key = jax.random.key(42)
rng_key, init_key = jax.random.split(rng_key)
init_keys = jax.random.split(init_key, num_chains)

rng_key, warmup_key = jax.random.split(rng_key)
rng_key, sample_key = jax.random.split(rng_key)

from blackjax.mcmc.adjusted_mclmc_dynamic import rescale
from blackjax.util import run_inference_algorithm


def _adjusted_mclmc_dynamic(
    key,
    initial_position,
    logp_fn,
    tune,
    draws,
    transform=lambda state, _: state.position,
    diagonal_preconditioning=True,
    random_trajectory_length=True,
    L_proposal_factor=jnp.inf,
    progress_bar: bool = True,
):
    init_key, tune_key, run_key = jax.random.split(key, 3)

    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logp_fn,
        random_generator_arg=init_key,
    )

    if random_trajectory_length:
        integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps)
        )
    else:
        integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(avg_num_integration_steps)

    kernel = (
        lambda rng_key,
        state,
        avg_num_integration_steps,
        step_size,
        inverse_mass_matrix: blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
            integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
            inverse_mass_matrix=inverse_mass_matrix,
        )(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            logdensity_fn=logp_fn,
            L_proposal_factor=L_proposal_factor,
        )
    )

    target_acc_rate = 0.9  # our recommendation

    (blackjax_state_after_tuning, blackjax_mclmc_sampler_params, _) = blackjax.adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=tune,
        state=initial_state,
        rng_key=tune_key,
        target=target_acc_rate,
        frac_tune1=0.1,
        frac_tune2=0.1,
        frac_tune3=0.1,  # our recommendation
        diagonal_preconditioning=diagonal_preconditioning,
    )

    step_size = blackjax_mclmc_sampler_params.step_size
    L = blackjax_mclmc_sampler_params.L

    alg = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logp_fn,
        step_size=step_size,
        integration_steps_fn=lambda key: jnp.ceil(jax.random.uniform(key) * rescale(L / step_size)),
        inverse_mass_matrix=blackjax_mclmc_sampler_params.inverse_mass_matrix,
        L_proposal_factor=L_proposal_factor,
    )

    _, samples = run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=alg,
        num_steps=draws,
        transform=transform,
        progress_bar=progress_bar,
    )

    return samples


def _blackjax_inference_loop(
    seed, init_position, logp_fn, draws, tune, target_accept=0.8, progress_bar=True, dense_matrix=False
):
    adapt = blackjax.window_adaptation(
        algorithm=blackjax.nuts,
        logdensity_fn=logp_fn,
        target_acceptance_rate=target_accept,
        progress_bar=progress_bar,
        is_mass_matrix_diagonal=dense_matrix,
    )
    (last_state, tuned_params), _ = adapt.run(seed, init_position, num_steps=tune)
    kernel = blackjax.nuts(logp_fn, **tuned_params).step

    @jax.jit
    def _one_step(state, xs):
        _, rng_key = xs
        state, info = kernel(rng_key, state)
        position = state.position
        return state, position

    keys = jax.random.split(seed, draws)
    scan_fn = gen_scan_fn(draws, progress_bar)
    _, samples = scan_fn(_one_step, last_state, (jnp.arange(draws), keys))

    return samples

def run_inference_multiple_chains(
    tune: int,
    draws: int,
    chains: int,
    random_seed: int,
    inference_algorithm_loop=_blackjax_inference_loop,
) -> az.InferenceData:
    seed = jax.random.key(random_seed)
    init_key, seed = jax.random.split(seed)
    sample_key, seed = jax.random.split(seed)
    init_keys = jax.random.split(init_key, chains)
    sample_keys = jax.random.split(sample_key, chains)

    param_info, potential_fn, postprocess_fn, *_ = initialize_model(
        init_keys,
        probabilistic_model,
        model_args=(model_parameters, model_args, data, batch_gen_spectrum_symbolic),
        dynamic_args=True,  # <- this is important!
    )
    initial_points = param_info.z

    def logp_fn(position):
        func = potential_fn(model_parameters, model_args, data, batch_gen_spectrum_symbolic)
        return -func(position)

    def transform_fn(position):
        func = postprocess_fn(model_parameters, model_args, data, batch_gen_spectrum_symbolic)
        return func(position)

    get_posterior_samples = partial(inference_algorithm_loop, logp_fn=logp_fn, tune=tune, draws=draws)
    raw_mcmc_samples = jax.pmap(get_posterior_samples)(sample_keys, initial_points)
    transformed_samples = transform_fn(raw_mcmc_samples)
    idata = az.from_dict(posterior=transformed_samples)

    return idata

# %%
algorithm = _blackjax_inference_loop
idata = run_inference_multiple_chains(2000, 500, 8, 0, algorithm)
print(az.summary(idata, round_to=5))

az.plot_posterior(idata)
plt.show()
# %%