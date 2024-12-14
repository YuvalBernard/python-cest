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

import jax
import jax.numpy as jnp
import jax.random
import lmfit
import numpy as np
import numpyro
import numpyro.distributions as dist

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
numpyro.set_host_device_count(4)

from simulate_spectra import (
    batch_gen_MTR_REX_analytical,
    batch_gen_MTR_REX_numerical,
    batch_gen_MTR_REX_ss_analytical,
    batch_gen_MTR_REX_ss_direct_symbolic,
    batch_gen_MTR_REX_ss_indirect_symbolic,
    batch_gen_MTR_REX_ss_numerical,
    batch_gen_MTR_REX_symbolic,
)


def least_squares(
    model_parameters: lmfit.Parameters,
    model_args: tuple,
    data: jax.Array | np.typing.ArrayLike,
    method: Callable,
    algorithm: str,
):
    """
    Fit data to Bloch-McConnell equations, with option to pick method of solving the equations.
    Solver is Levenberg-Marquardt; basically interface for lmfit.
    Returns best-fit as Parameters class (See https://lmfit.github.io/lmfit-py/ for more info.)
    """
    if method not in [
        batch_gen_MTR_REX_analytical,
        batch_gen_MTR_REX_ss_analytical,
        batch_gen_MTR_REX_numerical,
        batch_gen_MTR_REX_ss_numerical,
        batch_gen_MTR_REX_symbolic,
        batch_gen_MTR_REX_ss_direct_symbolic,
        batch_gen_MTR_REX_ss_indirect_symbolic,
    ]:
        raise NameError("Please insert a valid method of solving Bloch-McConnell equations from the list.")
    method_jitted = jax.jit(method)

    def objective(model_parameters, args, data, method) -> jax.Array:
        offsets, powers, B0, gamma, tp = args
        fit_pars = jnp.array(list(model_parameters.valuesdict().values()))
        resid = data - method(fit_pars, offsets, powers, B0, gamma, tp)
        return resid.flatten()

    fitter = lmfit.Minimizer(userfcn=objective, params=model_parameters, fcn_args=(model_args, data, method_jitted))
    match algorithm:
        case "Levenberg-Marquardt":
            fit = fitter.leastsq()
        case "Trust Region Reflective":
            fit = fitter.least_squares()
        case "Basin-Hopping":
            fit = fitter.basinhopping()
        case "Adaptive Memory Programming for Global Optimization":
            fit = fitter.ampgo()
    return {"fit": fit.params}


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
                    low=model_parameters[par].min,
                    high=model_parameters[par].max,
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
        numpyro.infer.NUTS(
            probabilistic_model,
        ),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=True,
    )
    mcmc.run(jax.random.key(1), model_parameters, model_args, data, method)
    return {"fit": mcmc}


def bayesian_vi(
    model_parameters: lmfit.Parameters,
    model_args: tuple,
    data: jax.Array | np.ndarray,
    method: Callable,
    optimizer_step_size: float | None = 1e-3,
    num_steps: int | None = 75_000,
    num_samples: int | None = 100_000,
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
                    low=model_parameters[par].min,
                    high=model_parameters[par].max,
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
    optimizer = numpyro.optim.RMSProp(step_size=optimizer_step_size)
    svi = numpyro.infer.SVI(probabilistic_model, guide, optimizer, loss=numpyro.infer.Trace_ELBO())
    svi_result = svi.run(jax.random.key(1), num_steps, model_parameters, model_args, data, method, progress_bar=False)
    # Get posterior samples
    posterior_samples = guide.sample_posterior(jax.random.key(2), svi_result.params, sample_shape=(num_samples,))
    return {"fit": posterior_samples, "loss": svi_result.losses}
