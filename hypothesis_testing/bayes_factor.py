import jax
import jax.numpy as jnp

from hypothesis_testing import quad

from typing import NamedTuple, Any
from LombScargle import periodogram, psd

e2r2 = quad.get_scheme(2, 7)
e3r2 = quad.get_scheme(3, 7)


class ModeInfo(NamedTuple):
    z: Any # parameter location of the mode (can be e.g. MAP, MLE or mean value)
    logp: float # -log p(z)
    cov: Any # Covariance matrix


def optimize(logp, z0):
    nlogp = lambda logz: -logp(jnp.exp(logz))
    opt = minimize(nlogp, x0 = jnp.log(z0), method = 'BFGS')
    
    hess = jax.hessian(nlogp)(opt.x)
    
    return nlogp, ModeInfo(opt.x, opt.fun, jnp.linalg.inv(hess))



def logB(time, data, err_data, freq, floating_mean= False):
    
    cov_func = drw_kernel
    
    log_lik, log_lik_null = psd.log_likelihood(time, data, cov_func, err_data)
    
    # analyze the null model
    map = optimize(log_lik_null)
    log_ev0 = quad(nlogp, map, e2r2)
    
    # find the best candidate for the alternative
    score, best_params = jax.vmap(periodogram.func(time, data, floating_mean, jax.linalg.cholesky(cov)))(freq)
    z = best_params[jnp.argmax(score)]
    map = optimize(nlog_lik)
        
    # compute the evidence for the alternative
    log_ev1 = quad(nlogp, map, e3r2)
    
    # return the log Bayes factor and the optimal parameters
    return log_ev1 - log_ev0, map.z



def quad(nlogp, MAP, quad_scheme):
    """ 3d Gaussian quadrature integration for the Bayes factor. We want to evaluate 
            B = \int \frac{p(z | H1)}{p(H0)} dz = \int e^{-nlogp(z)} dz
        The Gaussian quadrature is based on the Laplace approximation of the Hessian around MAP.
        Args:
            nlogp: a function, z is an array with 3 elements.
            MAP: MAPInfo object
    
       Returns: log B (ignoring the amplitude parameter)
    """
    
    d = len(MAP.z)
        
    ### go in the basis where the covariance matrix is the identity ###
    detCov = jnp.linalg.det(MAP.cov)
    D, Q = jnp.linalg.eigh(MAP.cov)
    
    M = jnp.sqrt(2)* jnp.dot(Q, jnp.diag(jnp.sqrt(D))) 
    
    def residual_integrand(X):
        z = jnp.dot(M, X) + MAP.z
        return jnp.exp(-nlogp(z) + MAP.nlogp + jnp.sum(jnp.square(X))) # residual integrand in this basis
    
    ### do the integral ###
    val = quad.evidence_and_cov(residual_integrand, *quad_scheme) 
    
    val *= jnp.sqrt(2**d * detCov) # for the change of basis
    logB = - MAP.nlogp + jnp.log(val)

    return logB
