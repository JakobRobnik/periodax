# Quadrature schemes for solving an integral of type
# \int e^{-|x|^2 } f(x) d x,
# where x is low dimensional


import jax
import jax.numpy as jnp
import os

def integrate(vals, w, d):
    """Approximates the integral I = int e^{-x^2} f(x) dx for a d dimensional (typically d = 2 or 3) x by a quadrature:
        I = sum_k f(x_k) w_k
        x: points where f is evaluated
        w: weights for those points
    """
    
    return jnp.sum(w * vals) * jnp.power(jnp.pi, d * 0.5) # this factor is for the integral e^{-x^2} dx = pi^d/2



def get_scheme(d, order):
    """Downloads good integration schemes."""
    
    dirr = os.path.dirname(os.path.realpath(__file__)) + '/quad_schemes/d'+str(d)+'/'
    
    x = jnp.load(dirr + 'points'+str(order)+'.npy').T
    w = jnp.load(dirr + 'weights'+str(order)+'.npy')
    
    return x, w


