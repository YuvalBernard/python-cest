# %%
"""
Module comprising different methods to fit Z-spectra (multi-B1),
MUST Given one of the following methods of solving Bloch-McConell equations:
    - Analytical solution by Moritz Zaiss
    - Numerical solution
    - Symbolic solution
Includes:
    - Nonlinear least squares (Levenberg-Marquardt)
    - Bayesian, Markov Chain Monte Carlo (NUTS sampler)
    - Bayesian, Stochastic Variational Inference (MultiVariate Normal guide)

Optional fitting parameters (any can be static):
    - R1a
    - R2a
    - R1b
    - R2b
    - k
    - f
    - dwa
    - dwb
Other parameters:
    - offsets
    - powers
    - B0
    - gamma
    - tp
"""

from typing import Callable

import arviz as az
import jax
import jax.numpy as jnp
import jax.random
import lmfit
import numpy as np
import numpyro
import numpyro.distributions as dist

jax.config.update("jax_enable_x64", True)
from simulate_spectra import (
    batch_gen_spectrum_analytical,
    batch_gen_spectrum_numerical,
    batch_gen_spectrum_symbolic,
)


def least_squares(
    model_parameters: lmfit.Parameters,
    model_args: tuple,
    data: jax.Array | np.typing.ArrayLike,
    method: Callable,
):
    """
    Fit data to Bloch-McConnell equations, with option to pick method of solving the equations.
    Solver is Levenberg-Marquardt; basically interface for lmfit.
    Returns best-fit as Parameters class (See https://lmfit.github.io/lmfit-py/ for more info.)
    """
    if method not in [
        batch_gen_spectrum_numerical,
        batch_gen_spectrum_symbolic,
        batch_gen_spectrum_analytical,
    ]:
        raise NameError("Please insert a valid method of solving Bloch-McConnell equations from the list.")
    method_jitted = jax.jit(method)

    def objective(model_parameters, args, data, method) -> jax.Array:
        offsets, powers, B0, gamma, tp = args
        fit_pars = jnp.array(list(model_parameters.valuesdict().values()))
        resid = data - method(fit_pars, offsets, powers, B0, gamma, tp)
        return resid.flatten()

    fit = lmfit.minimize(
        fcn=objective,
        params=model_parameters,
        method="least_squares",
        args=(model_args, data, method_jitted),
    )
    # Return best-fit parameters
    return fit.params


def bayesian_mcmc(
    model_parameters: lmfit.Parameters,
    model_args: tuple,
    data: jax.Array | np.ndarray,
    method: Callable,
    num_warmup: int | None = 1000,
    num_samples: int | None = 2000,
    num_chains: int | None = 4,
):
    """
    Fit data to Bloch-McConnell equations, with option to pick method of solving the equations.
    Use a Bayesian scheme and the NUTS sampler for MCMC; basically interface for numpyro.
    Returns
        - posterior samples in arviz `idata' format
        - summary statistics
    """
    num_warmup = num_warmup if num_warmup is not None else 1000
    num_samples = num_samples if num_samples is not None else 2000
    num_chains = num_chains if num_chains is not None else 4

    # set priors for model parameters if they are set to vary.
    # Each parameter gets a Normal distribution centered about (min, max),
    # such that the distance between the mean and (min, max) is 3*sigma.
    for par in list(model_parameters.keys()):
        if model_parameters[par].vary:
            if par in ["dwa", "dwb"]:
                model_parameters[par].prior = dist.Uniform(model_parameters[par].min, model_parameters[par].max)
            elif par in ["R1a", "R2a", "R1b", "R2b", "kb", "fb"]:
                model_parameters[par].prior = dist.TruncatedNormal(
                    (model_parameters[par].min + model_parameters[par].max) / 2,
                    (model_parameters[par].max - model_parameters[par].min) / 6,
                    low=0,
                )

    # Define probabilistic model for both Bayesian protocols
    def probabilistic_model(model_parameters, model_args, data, method) -> None:
        offsets, powers, B0, gamma, tp = model_args
        fit_pars = jnp.asarray(
            [
                numpyro.sample(model_parameters[par].name, model_parameters[par].prior)
                if model_parameters[par].vary
                else model_parameters[par].value
                for par in list(model_parameters.keys())
            ]
        )
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.03))
        model_pred = method(fit_pars, offsets, powers, B0, gamma, tp)
        numpyro.sample("obs", dist.Normal(model_pred, sigma), obs=data)

    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(probabilistic_model, init_strategy=numpyro.infer.init_to_uniform),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=True,
    )
    mcmc.run(jax.random.key(1), model_parameters, model_args, data, method)
    idata = az.from_numpyro(posterior=mcmc)
    fit_summary = az.summary(
        idata,
        round_to=5,
        stat_funcs={
            "median": np.median,
            "mode": lambda x: az.plots.plot_utils.calculate_point_estimate("mode", x),
        },
        var_names=["~sigma"],
    )
    posterior_samples = mcmc.get_samples(group_by_chain=True)
    return posterior_samples


def bayesian_vi(
    model_parameters: lmfit.Parameters,
    model_args: tuple,
    data: jax.Array | np.ndarray,
    method: Callable,
    optimizer_step_size: float | None = 1e-3,
    num_steps: int | None = 80_000,
    num_samples: int | None = 4000,
):
    """
    Fit data to Bloch-McConnell equations, with option to pick method of solving the equations.
    Use a Bayesian scheme and ADVI; basically interface for numpyro.
    Returns
        - posterior samples in arviz `idata' format
        - summary statistics
    """
    optimizer_step_size = optimizer_step_size if optimizer_step_size is not None else 1e-3
    num_steps = num_steps if num_steps is not None else 80_000
    num_samples = num_samples if num_samples is not None else num_samples

    # set priors for model parameters if they are set to vary
    for par in list(model_parameters.keys()):
        if model_parameters[par].vary:
            if par in ["dwa", "dwb"]:
                model_parameters[par].prior = dist.Uniform(model_parameters[par].min, model_parameters[par].max)
            elif par in ["R1a", "R2a", "R1b", "R2b", "kb", "fb"]:
                model_parameters[par].prior = dist.TruncatedNormal(
                    (model_parameters[par].min + model_parameters[par].max) / 2,
                    (model_parameters[par].max - model_parameters[par].min) / 6,
                    low=0,
                )

    # Define probabilistic model for both Bayesian protocols
    def probabilistic_model(model_parameters, model_args, data, method) -> None:
        offsets, powers, B0, gamma, tp = model_args
        fit_pars = jnp.array(
            [
                numpyro.sample(model_parameters[par].name, model_parameters[par].prior)
                if model_parameters[par].vary
                else model_parameters[par].value
                for par in list(model_parameters.keys())
            ]
        )
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.03))
        model_pred = method(fit_pars, offsets, powers, B0, gamma, tp)
        numpyro.sample("obs", dist.Normal(model_pred, sigma), obs=data)

    guide = numpyro.infer.autoguide.AutoMultivariateNormal(probabilistic_model)
    optimizer = numpyro.optim.ClippedAdam(step_size=optimizer_step_size)
    svi = numpyro.infer.SVI(probabilistic_model, guide, optimizer, loss=numpyro.infer.TraceMeanField_ELBO())
    svi_result = svi.run(jax.random.key(1), num_steps, model_parameters, model_args, data, method, progress_bar=True)
    # Get posterior samples
    posterior_samples = guide.sample_posterior(jax.random.key(2), svi_result.params, sample_shape=(num_samples,))
    idata = az.from_dict(posterior_samples)
    fit_summary = az.summary(
        idata,
        kind="stats",
        round_to=5,
        stat_funcs={
            "median": np.median,
            "mode": lambda x: az.plots.plot_utils.calculate_point_estimate("mode", x),
        },
        var_names=["~sigma"],
    )
    return posterior_samples


# %% TEST

# model_parameters = lmfit.Parameters()
# # add with tuples: (NAME INIT_VALUE VARY MIN  MAX)

# model_parameters.add_many(
#     ("R1a", 0.33, False, None, None),
#     ("R2a", 0.5, False, None, None),
#     ("dwa", 0.0, False, None, None),
#     ("R1b", 5.0, True, 0.1, 10.0),
#     ("R2b", 1.0, True, 0.1, 100.0),
#     ("kb", 300.0, True, 1.0, 500.0),
#     ("fb", 5e-5, True, 1e-5, 5e-3),
#     ("dwb", 3.7, True, 3.0, 4.0),
# )

# offsets = jnp.arange(-6, 6, 0.1, dtype=float)
# powers = jnp.array([1.0, 3.0])
# B0 = 4.7
# gamma = 267.522
# tp = 15.0
# args = (offsets, powers, B0, gamma, tp)
# #           R1b   R2b  dwa  R1b  R2b   k     f     dwb
# fit_pars = jnp.array([0.33, 0.5, 0.0, 1.0, 30.0, 200.0, 5e-4, 3.5])
# Z = batch_gen_spectrum_numerical(fit_pars, offsets, powers, B0, gamma, tp)
# data = Z + 0.02 * jax.random.normal(jax.random.key(0), jnp.shape(Z))

# best_fit_nls = least_squares(model_parameters, args, data, batch_gen_spectrum_symbolic)
# best_fit_nls.pretty_print()
# # posterior_samples_mcmc = bayesian_mcmc(model_parameters, args, data, batch_gen_spectrum_analytical)
# # posterior_samples_vi = bayesian_vi(model_parameters, args, data, batch_gen_spectrum_analytical)
# # %%
# best_fit_pars = tuple(best_fit_nls.valuesdict().values())
# batch_gen_spectrum_symbolic(best_fit_pars, *args)

# tuple(
#     np.median(posterior_samples_mcmc[par]) if model_parameters[par].vary else model_parameters[par].value
#     for par in model_parameters.keys()
# )
