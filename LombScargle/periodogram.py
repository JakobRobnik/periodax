import jax 
import jax.numpy as jnp


def inverse2(A):
    """inverse of a 2x2 matrix"""
    
    det = A[0, 0] * A[1, 1] - A[0, 1]**2
    return jnp.array([[A[1, 1], -A[0, 1]],
                      [-A[0, 1], A[0, 0]]]) / det


def inverse3(A):
    """inverse of a 3x3 matix"""
    
    # subdeterminants
    sub01 = A[0, 1] * A[2, 2] - A[0, 2] * A[1, 2]
    sub02 = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    sub12 = A[0, 0] * A[1, 2] - A[0, 1] * A[0, 2]
    sub00 = A[1, 1] * A[2, 2] - A[1, 2]**2
    sub11 = A[0, 0] * A[2, 2] - A[0, 2]**2
    sub22 = A[0, 0] * A[1, 1] - A[0, 1]**2
    
    #determinant
    det = A[0, 0] * sub00 - A[0, 1] * sub01 + A[0, 2] * sub02
    
    return jnp.array([[sub00, -sub01, sub02],
                      [-sub01, sub11, -sub12],
                      [sub02, -sub12, sub22]]) / det


def lomb_scargle(t, freq):
    """templates for the Lomb-Scargle periodogram"""
    return jnp.array([jnp.sin(2 * jnp.pi * freq * t), jnp.cos(2 * jnp.pi * freq * t)])


def floating_mean_lomb_scargle(t, freq):
    """templates for the floating mean Lomb-Scargle periodogram"""
    return jnp.array([jnp.ones(t.shape), jnp.sin(2 * jnp.pi * freq * t), jnp.cos(2 * jnp.pi * freq * t)])
    
    
def compute2(time, data, freq, weight):
    """Lomb-Scargle periodogram"""
    
    templates = lomb_scargle(time, freq)
    
    weighted_template0 = weight(templates[0])
    weighted_template1 = weight(templates[1])
    g00 = jnp.dot(templates[0], weighted_template0)
    g01 = jnp.dot(templates[0], weighted_template1)
    g11 = jnp.dot(templates[1], weighted_template1)
    metric = jnp.array([[g00, g01], [g01, g11]])
    inv_metric = inverse2(metric)
    
    overlap = jnp.array([jnp.dot(data, weighted_template0), jnp.dot(data, weighted_template1)])
    
    return metric_to_score(overlap, inv_metric)
    

def compute3(time, data, freq, weight):
    """floating mean Lomb-Scargle periodogram"""
    
    templates = floating_mean_lomb_scargle(time, freq)
    
    weighted_template0 = weight(templates[0])
    weighted_template1 = weight(templates[1])
    weighted_template2 = weight(templates[2])
    
    g00 = jnp.dot(templates[0], weighted_template0)
    g11 = jnp.dot(templates[1], weighted_template1)
    g22 = jnp.dot(templates[2], weighted_template2)
    g01 = jnp.dot(templates[0], weighted_template1)
    g02 = jnp.dot(templates[0], weighted_template2)
    g12 = jnp.dot(templates[1], weighted_template2)
    
    metric = jnp.array([[g00, g01, g02], 
                        [g01, g11, g12],
                        [g02, g12, g22]])
    inv_metric = inverse3(metric)
    
    overlap = jnp.array([jnp.dot(data, weighted_template0), jnp.dot(data, weighted_template1), jnp.dot(data, weighted_template2)])
    
    return metric_to_score(overlap, inv_metric)

    
def metric_to_score(overlap, inv_metric):
    """given the template metric compute the periodogram score and the optimal linear parameters"""
    score = overlap.T @ inv_metric @ overlap
    opt_parameters = inv_metric @ overlap
    return score, opt_parameters


def get_weight_func(sqrt_cov):
    """ noise weighting, computes Simga^-1 x """
    
    if sqrt_cov == None:
        return lambda x: x
    elif len(sqrt_cov.shape) == 1:
        return lambda x: x / jnp.square(sqrt_cov)
    else:
        return lambda x: jax.scipy.linalg.cho_solve((sqrt_cov, True), x)
    
    
def func(time, data, floating_mean= False, sqrt_cov= None):
    """sqrt_cov is the square root of a noise covariance matrix. It can be: 
            None: periodogram will assume equal error, non-correlated noise
            1d array: non-equal error, non-correlated noise. In this case sqrt_cov[i] is the error of data[i]
            2d array: correlated noise, this is the square root of the covariance matrix, meaning that it is L in its Cholesky decomposition Cov = L L^T. It can be obtained e.g. by L = jnp.linalg.cholesky(Cov)
    """
    
    weight_func = get_weight_func(sqrt_cov)
    
    ### periodogram computation
    if floating_mean:
        return lambda freq: compute3(time, data, freq, weight_func)

    else:
        return lambda freq: compute2(time, data, freq, weight_func)


def log_prob_null(data, sqrt_cov):
    weight_func = get_weight_func(sqrt_cov)
    log_det = jnp.sum(jnp.log(jnp.square(jnp.diag(sqrt_cov))))
    return -0.5 * jnp.dot(data, weight_func(data)) -0.5 * log_det
    

def _drifting_freq(t, freq, mode_spread):
    """null template with the drifting frequency
    freq: base frequency"""
    tmin, tmax = jnp.min(t), jnp.max(t)
    time_span = tmax-tmin
    return freq + (((t-tmin) / time_span) - 0.5) * mode_spread / time_span


drifting_freq = jax.vmap(_drifting_freq, (None, 0, None)) #vectorized over the base frequency


