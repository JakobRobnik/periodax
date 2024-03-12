import jax
import jax.numpy as jnp

from hypothesis_testing import quad

from typing import NamedTuple, Any
from LombScargle import periodogram, psd

e2r2 = quad.get_scheme(2, 7)
e3r2 = quad.get_scheme(3, 7)


# optimization and integration will be performed with respect to unconstrained parameters y = log z,
# where z are either (freq, sigma, tau) or (sigma tau), all positive.



class ModeInfo(NamedTuple):
    y: Any # parameter location of the mode
    nlogp: float # -log p(z(y))
    cov: Any # Covariance matrix (in y parametrization)


def optimize(nlogp, init):
    
    opt = jax.scipy.minimize(nlogp, x0 = init, method = 'BFGS')
    hess = jax.hessian(nlogp)(opt.x)
    
    return ModeInfo(opt.x, opt.fun, jnp.linalg.inv(hess))



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



def logB(time, data, err_data, freq, floating_mean= False):
    
    null, alternative = psd.nlog_density(time, data, err_data, freq[0], freq[-1])
    
    # analyze the null model
    y_init = jnp.log(jnp.array([0.1, 120])) # mode of the prior distribution
    map = optimize(null, y_init)
    log_ev0 = quad(null, map, e2r2)
    
    # find the best candidate for the alternative
    cov = psd.covariance(time, psd.drw_kernel(*jnp.exp(map.y)), err_data)
    score, _ = jax.vmap(periodogram.func(time, data, floating_mean, jax.linalg.cholesky(cov)))(freq)
    freq_best = freq[jnp.argmax(score)]
    map = optimize(nlog_lik)
        
    # compute the evidence for the alternative
    log_ev1 = quad(nlogp, map, e3r2)
    
    # return the log Bayes factor and the optimal parameters
    return log_ev1 - log_ev0, map.z
