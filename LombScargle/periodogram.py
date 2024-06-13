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



def basic(t, freq):
    """Templates for the Lomb-Scargle periodogram (sine and cosine)."""
    return jnp.sin(2 * jnp.pi * freq * t), jnp.cos(2 * jnp.pi * freq * t)



def get_y(key, num, spread):
    """Generates the periods for the null signal template. The first few periods are fixed (were optimized to minimize the overlap with the true template), the others are drawn from the uniform distribution."""
    y_fixed = jnp.array([0.8635225711753599, 3.0004414364876055, 1.7112635323263894, 0.4661426878500358, 0.3608930521377926, 0.32117450191005287, 0.27656221811276294]) # their average should be 1
    y_random = jax.random.uniform(key, shape= (num, ), minval= 1./jnp.sqrt(spread), maxval= jnp.sqrt(spread))
    
    return jnp.concatenate((y_fixed, y_random))



def null_signal_template(key, num, spread= 3.):
    """Null signal template for the Lomb-Scargle periodogram.
        num: the number of random periods to be generated. Should be larger than T / minimal period that will be later used with the template
        spread: width of the period randomization distribution. We recomend using the default value.
    """
    
    y = get_y(key, num, spread)
        
    def get_periods(freq, total_time):
        """Convert the y to the periods of cycles.
            cycles is a float = freq * total time"""
        cycles= freq * total_time
        cycles_frac, _cycles_integer = jnp.modf(cycles)
        cycles_integer = _cycles_integer.astype(int)

        weights = (jnp.arange(1, len(y)+1) <= cycles).astype(float)  
        weights = weights.at[cycles_integer].set(cycles_frac)
        return total_time * y / jnp.sum(y * weights)


    def temp(_t, freq):
        t = _t - jnp.min(_t)
        periods = get_periods(freq, jnp.max(t)) 
        grid = jnp.cumsum(periods)
        grid_paddled = jnp.insert(grid, 0, 0.)
        which_period = jnp.searchsorted(grid, t)
        x = (t - grid_paddled[which_period]) / periods[which_period]
        return jnp.sin(2 * jnp.pi * x), jnp.cos(2 * jnp.pi * x)
    
    return temp



def remove_mean(uncentered_data, weight):
    """remove the data mean"""
    ones = jnp.ones(uncentered_data.shape)
    weighted_ones = weight(ones)
    avg = jnp.dot(uncentered_data, weighted_ones) / jnp.dot(ones, weighted_ones)
    return uncentered_data - avg, avg


    
def compute2(time, uncentered_data, freq, weight, temp_func):
    """Standard Lomb-Scargle periodogram"""

    data, amp = remove_mean(uncentered_data, weight)

    temp0, temp1 = temp_func(time, freq)

    wtemp0 = weight(temp0)
    wtemp1 = weight(temp1)
    g00 = jnp.dot(temp0, wtemp0)
    g01 = jnp.dot(temp0, wtemp1)
    g11 = jnp.dot(temp1, wtemp1)
    metric = jnp.array([[g00, g01], [g01, g11]])
    inv_metric = inverse2(metric)
    
    overlap = jnp.array([jnp.dot(data, wtemp0), jnp.dot(data, wtemp1)])
    
    score, opt_params = metric_to_score(overlap, inv_metric)
    
    return score, jnp.array([amp, *opt_params]) # we add the average to the optimal parameters



def compute3(time, uncentered_data, freq, weight, temp_func):
    """Floating mean Lomb-Scargle periodogram (recommended)"""
    
    data, amp = remove_mean(uncentered_data, weight)
    
    temp0 = jnp.ones(time.shape)
    temp1, temp2 = temp_func(time, freq)
    
    wtemp0 = weight(temp0)
    wtemp1 = weight(temp1)
    wtemp2 = weight(temp2)
    
    g00 = jnp.dot(temp0, wtemp0)
    g11 = jnp.dot(temp1, wtemp1)
    g22 = jnp.dot(temp2, wtemp2)
    g01 = jnp.dot(temp0, wtemp1)
    g02 = jnp.dot(temp0, wtemp2)
    g12 = jnp.dot(temp1, wtemp2)
    
    metric = jnp.array([[g00, g01, g02], 
                        [g01, g11, g12],
                        [g02, g12, g22]])
    inv_metric = inverse3(metric)
    
    overlap = jnp.array([jnp.dot(data, wtemp0), jnp.dot(data, wtemp1), jnp.dot(data, wtemp2)])
    
    score, opt_params = metric_to_score(overlap, inv_metric)
    shift= jnp.array([amp, 0., 0.]) # we have taken the average out, let's add it back
    return score, opt_params + shift


    
def metric_to_score(overlap, inv_metric):
    """given the template metric compute the periodogram score and the optimal linear parameters"""
    score = overlap.T @ inv_metric @ overlap
    opt_parameters = inv_metric @ overlap
    return score, opt_parameters



def get_weight_func(sqrt_cov):
    """noise weighting, computes Simga^-1 x """
    
    if sqrt_cov == None:
        return lambda x: x
    elif len(sqrt_cov.shape) == 1:
        return lambda x: x / jnp.square(sqrt_cov)
    else:
        return lambda x: jax.scipy.linalg.cho_solve((sqrt_cov, True), x)
    
    
    
def zero_for_zero_freq(freq, output):
    """Handle the freq = 0 case. In this case null = alternative and the periodogram score = 0.
        output is the Lomb-Scargle periodogram output which is nan if freq = 0. This function is the jax equivalent of
        if close(freq, 0.0):
            return 0 * output.shape
        else:
            return output
    """
    nonzero = jnp.abs(freq) > 1e-13
    return jax.tree_util.tree_map(lambda _output: nonzero* jnp.nan_to_num(_output), output)
    
    

def lomb_scargle(time, data, floating_mean= True, sqrt_cov= None, temp_func= basic):
    """Lomb-Scargle periodogram.
        Args:
            time: array of times where measurements are taken
            data: array of measurements
            floating_mean: weather the (frequency dependent) constant is also fitted (i.e. generalized periodogram). This is recommended.
            sqrt_cov: square root of the noise covariance matrix. It can be: 
                None: periodogram will assume equal error, non-correlated noise
                1d array: non-equal error, non-correlated noise. In this case sqrt_cov[i] is the error of data[i]
                2d array: correlated noise, this is the square root of the covariance matrix, meaning that it is L in its Cholesky decomposition Cov = L L^T. It can be obtained e.g. by L = jnp.linalg.cholesky(Cov)
            temp_func: a function frequency -> ("sinus" template, "cosinus" template). Two options are currently implemented (but custom functions with the same signature can also be used):
                temp_func = basic : the regular periodogram with sinus and cosinus templates
                temp_func = null_signal_template(key, num_periods) : the modified periodogram, where period of each cycle is random. Used as an effective null simulation.
                
                Note that templates which are not sinus and cosinus cannot be used straightforwardly, because the optimization over the phase cannot be done analytically then. 
                One can however directly use models of the form  A_1 f_1(t, x) + A_2 f_2(t, x)
                where x is an arbitrary non-linear parameter, simply by passing 
                temp_func = lambda t, x: (f1(t, x), f2(t, x))
                
        Returns:
            a function: frequency -> (score, amplitudes)
    """
    
    weight_func = get_weight_func(sqrt_cov)
    
    ### periodogram computation
    if floating_mean:
        return lambda freq: zero_for_zero_freq(freq, compute3(time, data, freq, weight_func, temp_func))

    else:
        return lambda freq: zero_for_zero_freq(freq, compute2(time, data, freq, weight_func, temp_func))



def fit(time, freq, amp, temp_func= basic):
    """Best fit model.
       Args:
            freq: frequency of the signal
            amp: best fit amplitudes at this frequency
       Returns:
            time series of shape time.shape
    """
    sin, cos = temp_func(time, freq)
    return amp[0] + amp[1]*sin + amp[2]*cos    



def loglik_null(uncentered_data, sqrt_cov):
    """log likelihood under the null hypothesis"""
    
    weight_func = get_weight_func(sqrt_cov)
    data, _ = remove_mean(uncentered_data, weight_func)
    log_det = jnp.sum(jnp.log(2 * jnp.pi * jnp.square(jnp.diag(sqrt_cov))))
    return -0.5 * jnp.dot(data, weight_func(data)) -0.5 * log_det
    