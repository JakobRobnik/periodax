import jax
import jax.random as rand
import jax.numpy as jnp

### alternative prior

def uniform_with_smooth_edge(a, b, h):
    """Uniform prior with smooth edges.
        Uniform in the range [a, b] and decays to zero in the regions [a-h, a] and [b, b+h].
        Normalization constant is 1/ (b - a + h).
        Returns:
            nlogp function, taking x and returning -log p(x)
    """
    
    nlog_normalization = jnp.log(b-a+h)
    c = (b-a) * 0.5
    
    def nlogp(x):
        centered_x = x - (a + c)
        y = (jnp.abs(centered_x) - c) / h
        edge = 0.5 * ( 1 + jnp.cos(jnp.pi * y))
        return jax.lax.select(y < 1, -jnp.log((y >= 0) * edge + (y < 0.)) + nlog_normalization, jnp.inf * jnp.ones(y.shape))
    
    ### we generate samples by rejection sampling with the uniform distribution proposal 
    
    def importance_weight(x):
        centered_x = x - (a + c)
        y = (jnp.abs(centered_x) - c) / h
        edge = 0.5 * ( 1 + jnp.cos(jnp.pi * y))
        return jax.lax.select(y < 0, 1., edge)
    
    def proposal(state):
        _, _, key = state
        key1, key2, key3 = jax.random.split(key, 3)
        x = jax.random.uniform(key1, minval= a - h, maxval= b + h)
        acc_prob = importance_weight(x)
        reject = jax.random.bernoulli(key2, 1.-acc_prob).astype(bool)
        return x, reject, key3

    def generate(key):
        cond = lambda state: state[1]
        init= (0., True, key)
        state= jax.lax.while_loop(cond, proposal, init)
        return state[0]
    
    return nlogp, generate



### null priors

def normal():
    mu = jnp.log(jnp.array([0.1, 120.]))
    sigma = jnp.array([0.2, 0.9])
    nlogp = lambda y: jnp.sum(0.5 * jnp.square((y - mu)/sigma) + 0.5 * jnp.log(2 * jnp.pi * jnp.square(sigma)))    
    generate = lambda key: rand.normal(key, shape= (2,)) * sigma + mu
    
    return nlogp, generate
    
