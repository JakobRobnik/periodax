import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import NamedTuple, Any

from . import quad
from ..LombScargle import periodogram, drw

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
    # opt = minimize(jax.value_and_grad(nlogp), x0 = init, method= 'Netown-CG', 
    #               jac= True, hess= jax.hessian(nlogp), options= {'maxiter': 50})
    opt = minimize(jax.value_and_grad(nlogp), x0 = init, method= 'BFGS', jac= True, options= {'maxiter': 50})
    hess = jax.hessian(nlogp)(opt.x)
    return ModeInfo(opt.x, opt.fun, jnp.linalg.inv(hess)), opt.success



def quadrature(nlogp, MAP, quad_scheme):
    """ 3d Gaussian quadrature integration for the Bayesian evidence. We want to evaluate 
            Z = \int p(z | data) dz = \int e^{-nlogp(z | data)} dz
        The Gaussian quadrature is based on the Laplace approximation of the Hessian around MAP.

        Args:
            nlogp: a function, z is an array with 2 or 3 elements.
            MAP: MAPInfo object
    
       Returns: 
            log Z (ignoring the amplitude parameters)
    """
    
    d = len(MAP.y)
    ### go in the basis where the covariance matrix is the identity ###
    detCov = jnp.linalg.det(MAP.cov)
    D, Q = jnp.linalg.eigh(MAP.cov)
    
    if jnp.any(D < 0.):
        return 0., False
        
    M = jnp.sqrt(2)* jnp.dot(Q, jnp.diag(jnp.sqrt(D))) 
    
    def residual_integrand(x):
        z = jnp.dot(M, x) + MAP.y
        return jnp.exp(-nlogp(z) + MAP.nlogp + jnp.sum(jnp.square(x))) # residual integrand in this basis
    
    ### do the integral ###
    points, weights = quad_scheme
    vals = jax.vmap(residual_integrand)(points)
    val = quad.integrate(vals, weights, d)
    val *= jnp.sqrt(detCov * 2**d) # for the change of basis
    log_evidence = - MAP.nlogp + jnp.log(val)
    
    return log_evidence, True
    


def logB(time, data, err_data, freq, nlogpr_logfreq, nlogpr_null, floating_mean= True, temp_func= periodogram.basic, plot_name= None):
    """log Bayes factor for the sinusoidal variability in the correlated Gaussian noise (ignores the marginalization over the amplitude parameters)
        priors are -log density and are in terms of log parameters"""
    
    
    nlogpost0, nlogpost1, nloglik0, nloglik1, get_amp = psd.nlog_density(time, data, err_data, nlogpr_logfreq, nlogpr_null, floating_mean, temp_func)

    ### analyze the null model ###
    y_init = jnp.log(jnp.array([0.1, 120])) # mode of the prior distribution
    map0, success1 = optimize(nlogpost0, y_init)
    log_ev0, success2 = quadrature(nlogpost0, map0, scheme_d2)

    ### analyze the alternative model ###
    # find the best candidate for the alternative
    cov = psd.covariance(time, psd.drw_kernel(*jnp.exp(map0.y)), err_data)
    score, _ = jax.vmap(periodogram.lomb_scargle(time, data, floating_mean, jnp.linalg.cholesky(cov), temp_func= temp_func))(freq)
    score_adjusted = score - 2 * (nlogpr_logfreq(jnp.log(freq)) - nlogpr_logfreq(jnp.log(freq[len(freq)//2])))
    freq_best = freq[jnp.argmax(score_adjusted)]
    
    if plot_name != None:
        visualization.main(nlogpost0, map0, scheme_d2, freq, score, score_adjusted, plot_name)
        
    # compute the evidence and the uncertainty in parameters
    y_init = jnp.array([jnp.log(freq_best), *map0.y])
    map1, success3 = optimize(nlogpost1, y_init)
    log_ev1, success4 = quadrature(nlogpost1, map1, scheme_d3)

    
    # the suboptimal test statistics, just for demonstration
    # - log likelihood ratio
    E = nloglik0(map0.y) - nloglik1(map1.y)
    
    # white noise periodogram score
    score_white, _ = jax.vmap(periodogram.lomb_scargle(time, data, floating_mean, err_data, temp_func= temp_func))(freq)
    score_white = jnp.max(score_white)
    
    ### return the log Bayes factor and the optimal parameters ###
    params = jnp.exp(map1.y)
    amp = get_amp(map1.y)
    period = 1/params[0]
    T = jnp.max(time) - jnp.min(time)
    
    return {'logB': log_ev1 - log_ev0, 'log_lik_ratio': E, 'white_periodogram': score_white,
            'cycles': T/period,
            'period': period, 'sigma': params[1], 'tau': params[2],
            'A_const': amp[0], 'A_sin': amp[1], 'A_cos': amp[2],
            'success': success1 and success3 and success2 and success4
            }

