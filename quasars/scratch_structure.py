import sys, shutil, os
import pandas as pd
import numpy as np


# Intermediate results are saved in scratch folder.
# Here are functions that handle the scratch folder (clean it, extract the results in single files, etc.)



scratch_base = '/pscratch/sd/j/jrobnik/'
home = '/global/homes/j/jrobnik/periodogram/'
scratch = scratch_base + 'quasars_scratch/'
dir_data = scratch_base + 'quasars/'
dir_results = home + 'data/'


def base_name(mode, temp, which_amp):
    return mode + '_' + str(temp) + '_' +str(which_amp)
    

def start(base):
    """empty scratch and make the structure again"""
    _base = scratch + base + '/'
    if os.path.exists(_base):
        shutil.rmtree(_base)
    os.mkdir(_base)
    
    
def finish(base):
    
    join(dir_in= scratch + base + '/',
         file_out= dir_results + base) + '_2tausim'
    
    
def join(dir_in, file_out):

    DF = []
    
    for file in os.listdir(dir_in):
        df = pd.read_csv(dir_in + file)
        DF.append(df)

    if len(DF) == 0:  # there may be no files in the directory
        print('scratch is empty.')
        return

    DF = pd.concat(DF)
    DF.to_csv(file_out + '.csv', index=False)

    

if __name__ == '__main__':
    
    start_finish = sys.argv[1]
    
    mode= sys.argv[2]
    temp= sys.argv[3]
    amp= sys.argv[4]
    base = base_name(mode, temp, amp)

    
    if start_finish == 'start':
        start(base)
        
    elif start_finish == 'finish':
        finish(base)

    else:
        raise ValueError("start_finish= " + mode + " is not a valid option. Should be 'start' or 'finish'.")    
