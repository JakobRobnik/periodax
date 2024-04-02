import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu') # we will use cpu here (because we can reserve so many), this line is only to avoid jax warning
jax.config.update("jax_enable_x64", True)

import pandas as pd
import numpy as np

from LombScargle import periodogram, psd
from quasars import parallel


scratch_base = '/pscratch/sd/j/jrobnik/'
scratch = scratch_base + 'quasars_scratch/'
dir_data = scratch_base + 'quasars/'


ids = jnp.load(dir_data + 'ids.npy')
#ids = jnp.array(pd.read_csv('quasars/known.csv')['id'], dtype= int)



def prepare_data(myid):
    
    df = pd.read_csv(dir_data + str(ids[myid]) + '.csv')
    time = jnp.array(df['time'])
    mag = jnp.array(df['mag'])
    mag_err = jnp.array(df['mag_err'])
    T = jnp.max(time) - jnp.min(time)
    fmin, fmax = 2./T, 1./60.
    factor = 2./1.5 # prior starts to die off at 2/T and is zero at 1.5/T
    prior_params = (jnp.log(fmin), jnp.log(fmax), jnp.log(factor))
    freq = jnp.logspace(jnp.log10(fmin/factor), jnp.log10(fmax*factor), 1000)
    
    return time, mag, mag_err, freq, prior_params

    

def post(myid, freq, score):
    ibest = jnp.argmax(score)
    score, period_best, amplitudes = score[ibest], 1./freq[ibest], best_params[ibest]

    # save the results
    df = pd.DataFrame([[score, myid, ids[myid], period_best, *amplitudes]],
                      columns = ['score', 'myid', 'id', 'period', 'A_const', 'A_sin', 'A_cos'])
    
    df.to_csv(scratch + 'candidates/' + str(myid) + '.csv', index= False)




def process(myid):
    
    time, mag, mag_err, freq, prior_params = prepare_data(myid)

    score, best_params = jax.vmap(periodogram.lomb_scargle(time, mag, sqrt_cov= mag_err))(freq)
    
    post(myid, freq, score)



# prep = parallel.error_handling(process)
# parallel.for_loop_with_job_manager(prep, 33, 33)
