# Quadrature schemes for solving an integral of type
# \int e^{-|x|^2 } f(x) d x,
# where x is 3-dimensional (this is called e3r2 in quadpy)


import jax
import jax.numpy as jnp


def integrate(f, x, w, d):
    """approximates the integral I = int e^{-x^2} f(x) dx for a three dimensional x by a quadrature:
        I = sum_k f(x_k) w_k
        x: points where the f is evaluated
        w: weights for those points
    """
    
    return jnp.sum(w * jax.vmap(f)(x)) * jnp.power(jnp.pi, d * 0.5) # this factor is for the integral e^{-x^2} dx = pi^d/2



def get_scheme(d, order):
    """reads schemes that were copied from quadpy"""
    dirr = 'hypothesis_testing/quadpy/e'+str(d)+'r2/'
    x = jnp.load(dirr + 'points'+str(order)+'.npy').T
    w = jnp.load(dirr + 'weights'+str(order)+'.npy')
    
    return x, w


