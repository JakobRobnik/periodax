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



def quadrature(nlogp, MAP, quad_scheme, observables= False):
    """ 3d Gaussian quadrature integration for the Bayesian evidence. We want to evaluate 
            Z = \int p(z | data) dz = \int e^{-nlogp(z | data)} dz
        The Gaussian quadrature is based on the Laplace approximation of the Hessian around MAP.
        Optionally, expectation values of some observables are also computed.
        Args:
            nlogp: a function, z is an array with 2 or 3 elements.
            MAP: MAPInfo object
            observables: a list of functions. Each function takes y and returns a scalar.
    
       Returns: 
            log Z (ignoring the amplitude parameters)
            [E[f] for f in observables]
    """
    
    d = len(MAP.y)
    ### go in the basis where the covariance matrix is the identity ###
    detCov = jnp.linalg.det(MAP.cov)
    D, Q = jnp.linalg.eigh(MAP.cov)
    
    M = jnp.sqrt(2)* jnp.dot(Q, jnp.diag(jnp.sqrt(D))) 
    
    def residual_integrand(x):
        z = jnp.dot(M, x) + MAP.y
        return jnp.exp(-nlogp(z) + MAP.nlogp + jnp.sum(jnp.square(x))) # residual integrand in this basis
    
    def expectation(observable, normalization):
        vals= jax.vmap(lambda x: residual_integrand(x) * observable(jnp.dot(M, x) + MAP.y))(points)
        return quad.integrate(vals, weights, d) / normalization

    ### do the integral ###
    points, weights = quad_scheme
    vals = jax.vmap(residual_integrand)(points)
    _val = quad.integrate(vals, weights, d)
    val = _val * jnp.sqrt(detCov * 2**d) # for the change of basis
    log_evidence = - MAP.nlogp + jnp.log(val)
    
    if not observables:
        return log_evidence
    else:
        observable = lambda y: jnp.exp(-y[0])
        Ef = expectation(observable, _val)
        print(Ef)
        print(observable(MAP.y))
        squared_centered_observable = lambda y: jnp.square(observable(y) - Ef)
        Varf = expectation(squared_centered_observable, _val)

        return log_evidence, jnp.sqrt(Varf)
    


def logB(time, data, err_data, freq, nlogpr_logfreq, nlogpr_null, floating_mean= True, temp_func= periodogram.basic):
    """log Bayes factor for the sinusoidal variability in the correlated Gaussian noise (ignores the marginalization over the amplitude parameters)
        priors are -log density and are in terms of log parameters"""
    

    nlogpost0, nlogpost1, nloglik0, nloglik1 = psd.nlog_density(time, data, err_data, nlogpr_logfreq, nlogpr_null, floating_mean, temp_func)
  
    ### analyze the null model ###
    y_init = jnp.log(jnp.array([0.1, 120])) # mode of the prior distribution
    map0 = optimize(nlogpost0, y_init)
    log_ev0 = quadrature(nlogpost0, map0, scheme_d2)
    
    ### analyze the alternative model ###
    # find the best candidate for the alternative
    cov = psd.covariance(time, psd.drw_kernel(*jnp.exp(map0.y)), err_data)
    score, _ = jax.vmap(periodogram.lomb_scargle(time, data, floating_mean, jnp.linalg.cholesky(cov), temp_func= temp_func))(freq)
    score_adjusted = score - 2 * (nlogpr_logfreq(jnp.log(freq)) - nlogpr_logfreq(jnp.log(freq[len(freq)//2])))
    freq_best = freq[jnp.argmax(score_adjusted)]
    
    # compute the evidence and the uncertainty in parameters
    y_init = jnp.array([jnp.log(freq_best), *map0.y])
    map1 = optimize(nlogpost1, y_init)
    log_ev1, err_period = quadrature(nlogpost1, map1, scheme_d3, observables= True)
    
    import matplotlib.pyplot as plt
    period_best = jnp.exp(-map1.y[0])
    print(err_period)
    err_period = jnp.sqrt(map1.cov[0, 0]) / freq_best
    print(err_period)
    t = jnp.linspace(-1, 1, 30) * 3 * err_period + period_best
    
    nlogpost1_cross = lambda y0: nlogpost1(jnp.array([y0, *map1.y[1:]]))
    score = jax.vmap(nlogpost1_cross)(-jnp.log(t))
    plt.plot(t, score, color = 'tab:red')
    plt.plot(t, jnp.min(score) + 0.5 * jnp.square((t - period_best) / err_period), color= 'black')
    plt.xlim(180, 200)
    plt.savefig('neki.png')
    plt.close()
    exit()
    
    # the suboptimal test statistics, just for demonstration
    # - log likelihood ratio
    E = nloglik0(map0.y) - nloglik1(map1.y)
    
    # white noise periodogram score
    score_white, _ = jax.vmap(periodogram.lomb_scargle(time, data, floating_mean, err_data, temp_func= temp_func))(freq)
    score_white = jnp.max(score_white)
    
    ### return the log Bayes factor and the optimal parameters ###
    params = jnp.exp(map1.y)
    return {'logB': log_ev1 - log_ev0, 'log_lik_ratio': E, 'white_periodogram': score_white,
            'period': 1/params[0], 'err_period': err_period, 'sigma': params[1], 'tau': params[2]}

