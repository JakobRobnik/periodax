import jax
import jax.random as rand
import jax.numpy as jnp

### alternative prior



class SmoothenEdge:
    """Smoothen the edges of the distribution to zero. 
        base distribution stays as is in the range [a h, b/h] and zero outside [a, b].
    """
        
    def __init__(self, Base, h):

        # everything operates in terms of y = log x
        self.a, self.b = jnp.log(Base.a), jnp.log(Base.b)
        self.nlogp_base, self.rvs_base = Base.nlogp, Base.rvs
        self.h = jnp.log(h) 
        self.c = (self.b-self.a) * 0.5 # half width of the full interval
    
        # compute the correction to the normalization factor

        self.log_norm = self.normalization_correction()

    
    def normalization_correction(self, num_integration_points = 1000):

        dx = self.h / num_integration_points
        t = (jnp.arange(num_integration_points) + 0.5) * dx
        x = jnp.array([self.a + t,  # left edge
                    self.b - self.h + t]) # right edge
        
        integrand = jnp.exp(-self.nlogp_base(x)) * (self.window(x) - 1.)
        
        correction_integral = jnp.sum(jax.scipy.integrate.trapezoid(integrand, x))
        #correction_integral = jnp.sum(integrand) * dx
        print(correction_integral)
        return jnp.log(1 + correction_integral)
        

    def window(self, x):
        """1 in the untouched region"""
        centered_x = x - (self.a + self.c)
        y = (jnp.abs(centered_x) - self.c + self.h) / self.h
        edge = 0.5 * ( 1 + jnp.cos(jnp.pi * y))
        return jax.lax.select(y < 0, jnp.ones(x.shape), edge)

    
    def nlog_window(self, x):
        centered_x = x - (self.a + self.c) # x, relative to the ceneter of the interval
        y = (jnp.abs(centered_x) - self.c + self.h) / self.h # y = 0 at the edge of the untouched region and y > 1 outside of [a, b]
        edge = 0.5 * ( 1 + jnp.cos(jnp.pi * y))
        return jax.lax.select(y < 1, -jnp.log((y >= 0) * edge + (y < 0.)), jnp.inf * jnp.ones(y.shape))


    def nlogp(self, y):
        return self.nlog_window(y) + self.nlogp_base(y) + self.log_norm

    
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

  def __init__(self, alpha, a, b):
    """a and b are bounds for x, but rvs and nlogp operate in terms of y = log x"""

    self.logC = jnp.log((jnp.power(b, alpha + 1) - jnp.power(a, alpha + 1))/(alpha + 1))
    self.a, self.b, self.alpha = a, b, alpha

  def rvs(self, key):
    """returns y = log x"""
    U = jax.random.uniform(key)
    x= jnp.power(jnp.power(self.a, 1+self.alpha) * (1-U) + jnp.power(self.b, 1+self.alpha) * U, 1./(1+self.alpha))
    return jnp.log(x)

  def nlogp(self, y):
    return -y*(self.alpha+1) + self.logC




### null priors

class Normal:
    
    def __init__(self):
        mu = jnp.log(jnp.array([0.1, 120.]))
        sigma = jnp.array([0.2, 0.9])
    
        self.nlogp = lambda y: jnp.sum(0.5 * jnp.square((y - mu)/sigma) + 0.5 * jnp.log(2 * jnp.pi * jnp.square(sigma)))    
        self.rvs = lambda key: rand.normal(key, shape= (2,)) * sigma + mu
    
