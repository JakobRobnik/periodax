import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def get_quad_points(MAP, scheme):
    
    x, w = scheme
    d = len(MAP.y)
    
    ### go in the basis where the covariance matrix is the identity ###
    D, Q = jnp.linalg.eigh(MAP.cov)
    
    M = jnp.sqrt(2)* jnp.dot(Q, jnp.diag(jnp.sqrt(D))) 
    
    return jnp.array([jnp.dot(M, X) + MAP.y for X in x])


def quad2(nlogp, MAP, scheme):
    """plot the 2d posterior"""
    
    # quadrature points
    quad_points = get_quad_points(MAP, scheme)
    
    # grid
    minn, maxx = jnp.min(quad_points, axis= 0), jnp.max(quad_points, axis= 0)
    edge = (maxx - minn) * 0.1
    
    x= jnp.linspace(minn[0] - edge[0], maxx[0] + edge[0], 20)
    y= jnp.linspace(minn[1] - edge[1], maxx[1] + edge[1], 20)
    X, Y = jnp.meshgrid(x, y)
    XY = jnp.array([X, Y])
    
    # nlogp on the grid
    Z = jax.vmap(jax.vmap(nlogp, 1), 1)(XY) - MAP.nlogp
    plt.title('- log posterior')
    plt.plot(MAP.y[0], MAP.y[1], '*', markersize=10, color= 'tab:orange')
    plt.plot(quad_points[:, 0], quad_points[:, 1], 'o', color = 'tab:red')
    plt.contourf(np.array(X), np.array(Y), np.exp(-Z), cmap = 'Greys_r')
    plt.colorbar()
    plt.xlabel(r'$\log \sigma$')
    plt.ylabel(r'$\log \tau$')


# def quad3(nlogp, MAP, scheme):
#     """plot the 2d posterior"""
    
#     # quadrature points
#     quad_points = get_quad_points(MAP, scheme)
    
#     # grid
#     minn, maxx = jnp.min(quad_points, axis= 0), jnp.max(quad_points, axis= 0)
#     edge = (maxx - minn) * 0.1
    
#     x= [jnp.linspace(minn[i] - edge[i], maxx[i] + edge[i], 20) for i in range(len(minn))]
    
#     plt.title('- log posterior')
    
#     I = [[0, 1], [0, 2], [1, 2]]
    
#     for i in range(3):
#         ix, iy = I[i]    
#         # nlogp on the grid
#         X, Y = jnp.meshgrid(x[ix], x[iy])
#         XY = jnp.array([X, Y])
#         Z = jax.vmap(jax.vmap(nlogp, 1), 1)(XY) - MAP.nlogp
        
#         plt.plot(MAP.y[0], MAP.y[1], '*', markersize=10, color= 'ta:orange')
        
#         plt.plot(quad_points[:, 0], quad_points[:, 1], 'o', color = 'tab:red')
#         plt.contourf(np.array(X), np.array(Y), np.exp(-Z), cmap = 'Greys_r')
#         plt.colorbar()
        


def periodogram(freq, score, score_adjusted):
    
    plt.plot(1./freq, score, '.:', label= r'$\Delta \chi^2$', color= 'tab:red')
    plt.plot(1./freq, score_adjusted, '.:', label= r'$\Delta \chi^2 + 2 \log \mathrm{prior}$', color= 'tab:blue')
    
    plt.plot([1./freq[np.argmax(score_adjusted)], ],
             [np.max(score_adjusted), ], '*', markersize= 15, color = 'tab:orange')
    plt.xlabel('period [days]')
    plt.ylabel('score')
    plt.legend()
    


def main(nlogpost0, map0, scheme_d2, freq, score, score_adjusted, plot_name):
    plt.figure(figsize= (10, 10))
    
    plt.subplot(2, 1, 1)
    quad2(nlogpost0, map0, scheme_d2)
    
    plt.subplot(2, 1, 2)
    periodogram(freq, score, score_adjusted)
    
    plt.savefig(plot_name)
    plt.tight_layout()
    plt.close()
