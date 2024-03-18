import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize

from hypothesis_testing import quad

from typing import NamedTuple, Any
from LombScargle import periodogram, psd

scheme_d2 = quad.get_scheme(2, 6)
scheme_d3 = quad.get_scheme(3, 7)


# optimization and integration will be performed with respect to unconstrained parameters y = log z,
# where z are either (freq, sigma, tau) or (sigma tau), all positive.


class ModeInfo(NamedTuple):
    y: Any # parameter location of the mode
    nlogp: float # -log p(z(y))
    cov: Any # Covariance matrix (in y parametrization) = Hessian^-1, where H_ij = partial_i partial_j -log p(y)


def optimize(nlogp, init):
    """Given the initial guess for y = init, do optimization to maximize -logp.
       Returns:
            ModeInfo object
    """
    #opt = minimize(jax.value_and_grad(nlogp), x0 = init, method= 'Newton-CG', 
    #               jac= True, hess= jax.hessian(nlogp))
    opt = minimize(nlogp, x0 = init, method = 'BFGS')
    hess = jax.hessian(nlogp)(opt.x)
    return ModeInfo(opt.x, opt.fun, jnp.linalg.inv(hess))



def quadrature(nlogp, MAP, quad_scheme):
    """ 3d Gaussian quadrature integration for the Bayesian evidence. We want to evaluate 
            I = \int p(z | data) dz = \int e^{-nlogp(z | data)} dz
        The Gaussian quadrature is based on the Laplace approximation of the Hessian around MAP.
        Args:
            nlogp: a function, z is an array with 2 or 3 elements.
            MAP: MAPInfo object
    
       Returns: log I (ignoring the amplitude parameters)
    """
    
    d = len(MAP.y)
    ### go in the basis where the covariance matrix is the identity ###
    detCov = jnp.linalg.det(MAP.cov)
    D, Q = jnp.linalg.eigh(MAP.cov)
    
    M = jnp.sqrt(2)* jnp.dot(Q, jnp.diag(jnp.sqrt(D))) 
    
    def residual_integrand(X):
        z = jnp.dot(M, X) + MAP.y
        return jnp.exp(-nlogp(z) + MAP.nlogp + jnp.sum(jnp.square(X))) # residual integrand in this basis
    
    ### do the integral ###
    val = quad.integrate(residual_integrand, *quad_scheme, d) 
    
    val *= jnp.sqrt(2**d * detCov) # for the change of basis
    log_evidence = - MAP.nlogp + jnp.log(val)
    return log_evidence



def logB(time, data, err_data, freq, prior_freq, prior_null, floating_mean= True):
    """log Bayes factor for the sinusoidal variability in the correlated Gaussian noise (ignores the marginalization over the amplitude parameters)
        priors are -log density and are in terms of log parameters"""
    

    null, alternative, null_lik, alternative_lik = psd.nlog_density(time, data, err_data, prior_freq, prior_null, floating_mean)
  
    ### analyze the null model ###
    y_init = jnp.log(jnp.array([0.1, 120])) # mode of the prior distribution
    map0 = optimize(null, y_init)
    log_ev0 = quadrature(null, map0, scheme_d2)
    
    ### analyze the alternative model ###
    # find the best candidate for the alternative
    cov = psd.covariance(time, psd.drw_kernel(*jnp.exp(map0.y)), err_data)
    score, _ = jax.vmap(periodogram.func(time, data, floating_mean, jnp.linalg.cholesky(cov)))(freq)
    freq_best = freq[jnp.argmax(score)]
    
    # compute the evidence
    y_init = jnp.array([jnp.log(freq_best), *map0.y])
    map1 = optimize(alternative, y_init)
    log_ev1 = quadrature(alternative, map1, scheme_d3)

    # - log likelihood ratio
    E = null_lik(map0.y) - alternative_lik(map1.y)

    ### return the log Bayes factor and the optimal parameters ###
    params = jnp.exp(map1.y)
    return {'logB': log_ev1 - log_ev0, 'log_lik_ratio': E, 
            'period': 1/params[0], 'sigma': params[1], 'tau': params[2]}
    