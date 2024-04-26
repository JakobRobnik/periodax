import jax
import jax.random as rand
import jax.numpy as jnp

### alternative prior



class SmoothenEdge:
    """Smoothen the edges of the distribution to zero. 
        base distribution stays the same for x > a h and zero for x < a
    """
        
    def __init__(self, Base, h):

        # everything operates in terms of y = log x
        self.a = jnp.log(Base.freq_min)
        self.nlogp_base, self.rvs_base = Base.nlogp, Base.rvs
        self.h = jnp.log(h) 
        
        # compute the correction to the normalization factor

        self.log_norm_correction = self.normalization_correction()
        self.log_norm = self.log_norm_correction + Base.log_norm
        
    
    def normalization_correction(self, num_integration_points = 1000):

        dx = self.h / num_integration_points
        x = self.a + (jnp.arange(num_integration_points) + 0.5) * dx
        
        integrand = jnp.exp(-self.nlogp_base(x)) * (self.window(x) - 1.)
        
        correction_integral = jnp.sum(integrand) * dx
        return jnp.log(1 + correction_integral)
        

    def window(self, x):
        """1 in the untouched region"""
        y = (self.a + self.h - x) / self.h
        edge = 0.5 * ( 1 + jnp.cos(jnp.pi * y))
        is_inside = y < 0
        return is_inside + (1.-is_inside) * edge

    
    def nlog_window(self, x):
        y = (self.a + self.h - x) / self.h
        edge = 0.5 * ( 1 + jnp.cos(jnp.pi * y))
        return jax.lax.select(y < 1, -jnp.log((y >= 0) * edge + (y < 0.)), jnp.inf * jnp.ones(y.shape))


    def nlogp(self, y):
        return self.nlog_window(y) + self.nlogp_base(y) + self.log_norm_correction

    
    def rvs(self, key):
        """we generate samples by rejection sampling with the base distribution proposal"""
        
        
        def proposal(state):
            _, _, key = state
            key1, key2, key3 = jax.random.split(key, 3)
            x = self.rvs_base(key1)
            acc_prob = self.window(x)
            reject = jax.random.bernoulli(key2, 1.-acc_prob).astype(bool)
            return x, reject, key3

        cond = lambda state: state[1]
        init= (0., True, key)
        state= jax.lax.while_loop(cond, proposal, init)
        return state[0]
    


class PowerLaw:
  """p(x) \propto x^alpha within the bounds a < x < b"""

  def __init__(self, alpha, freq_min):
    """a and b are bounds for x, but rvs and nlogp operate in terms of y = log x"""

    self.log_norm = jnp.log( - jnp.power(freq_min, alpha + 1)/(alpha + 1))
    self.freq_min, self.alpha = freq_min, alpha

  def rvs(self, key):
    """returns y = log x"""
    U = jax.random.uniform(key)
    x= self.freq_min * jnp.power(U, 1./(1+self.alpha))
    return jnp.log(x)

  def nlogp(self, y):
    return -y*(self.alpha+1) + self.log_norm



class Uniform_with_smooth_edge:


    def __init__(self, low, high, edge_factor):
        """Uniform prior with smooth edges.
            Uniform in the range [a, b] and decays to zero in the regions [a-h, a] and [b, b+h].
            Normalization constant is 1/ (b - a + h).
            Returns:
                nlogp function, taking x and returning -log p(x)
        """
        
        h= jnp.log(edge_factor)
        
        a = low+h
        b = high-h
        
        self.log_norm = jnp.log(b-a+h)
        self.a = a
        self.b = b
        self.h = h
        self.c = (b-a) * 0.5
        
    def nlogp(self, x):
        centered_x = x - (self.a + self.c)
        y = (jnp.abs(centered_x) - self.c) / self.h
        edge = 0.5 * ( 1 + jnp.cos(jnp.pi * y))
        return jax.lax.select(y < 1, -jnp.log((y >= 0) * edge + (y < 0.)) + self.log_norm, jnp.inf * jnp.ones(y.shape))
    
    ### we generate samples by rejection sampling with the uniform distribution proposal 
    
    def importance_weight(self, x):
        centered_x = x - (self.a + self.c)
        y = (jnp.abs(centered_x) - self.c) / self.h
        edge = 0.5 * ( 1 + jnp.cos(jnp.pi * y))
        return jax.lax.select(y < 0, 1., edge)
    
    def proposal(self, state):
        _, _, key = state
        key1, key2, key3 = jax.random.split(key, 3)
        x = jax.random.uniform(key1, minval= self.a - self.h, maxval= self.b + self.h)
        acc_prob = self.importance_weight(x)
        reject = jax.random.bernoulli(key2, 1.-acc_prob).astype(bool)
        return x, reject, key3

    def rv(self, key):
        cond = lambda state: state[1]
        init= (0., True, key)
        state= jax.lax.while_loop(cond, self.proposal, init)
        return state[0]
    

### null priors

class Normal:
    
    def __init__(self):
        mu = jnp.log(jnp.array([0.1, 120.]))
        sigma = jnp.array([0.2, 0.9])
    
        self.nlogp = lambda y: jnp.sum(0.5 * jnp.square((y - mu)/sigma) + 0.5 * jnp.log(2 * jnp.pi * jnp.square(sigma)))    
        self.rvs = lambda key: rand.normal(key, shape= (2,)) * sigma + mu
    


def prepare(freq, redshift):    
    
    alpha= 8./3. # power law exponent for the period prior
    #PriorAlterantive = SmoothenEdge(PowerLaw(-alpha - 2., freq[0]), 1.2)
    PriorAlterantive = Uniform_with_smooth_edge(jnp.log(freq[0]), jnp.log(freq[-1]), 2./1.5)
    PriorNull = Normal()
    
    ### prior odds ###
    period_cut = 500. # arbitrary constant to make prior odds on order one (doesn't matter for the test statistic)
    base_log_norm = jnp.log(jnp.power(period_cut, alpha + 1) / (alpha + 1.))
    log_prior_odds = (alpha + 1.) * jnp.log(1+redshift) + PriorAlterantive.log_norm - base_log_norm
    
    return PriorAlterantive, PriorNull, log_prior_odds
    