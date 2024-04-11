import sys, shutil, os
import pandas as pd
import numpy as np


# Intermediate results are saved in scratch folder.
# Here are functions that handle the scratch folder (clean it, extract the results in single files, etc.)


scratch_base = '/pscratch/sd/j/jrobnik/'
scratch = scratch_base + 'quasars_scratch/'
dir_data = scratch_base + 'quasars/'
dir_results = 'data/'



def clean(mode, temp):
    """empty scratch and make the structure again"""
    base= scratch + mode + '_' + temp + '/'
    if os.path.exists(base):
        shutil.rmtree(base)
    for folder in ['', 'candidates/', 'error_log/']:
        os.mkdir(base + folder)
    
    
    
def finish(mode, temp, extra_name):
    
    base = scratch + mode + '_' + temp + '/'
    error_log(base)
    out = mode + ('_randomized' if (temp == 'randomized') else '')
    join(base + 'candidates/', out + extra_name)
    
    
def join(folder, name):

    DF = []
    
    for file in os.listdir(folder):
        df = pd.read_csv(folder + file)
        DF.append(df)

    if len(DF) == 0:  # there may be no files in the directory
        print('scratch is empty.')
        return

    DF = pd.concat(DF)
    DF.to_csv(dir_results + name + '.csv', index=False)




def error_log(base):
    """Combine the error logs from individual workers in a single file"""
    dir_base = base + 'error_log/'
    log_name = dir_base + 'combined_error_log.txt'
    
    # error log
    log = open(log_name, 'wt') #combined results
    for subdir in os.listdir(dir_base):
        #if subdir[:9] == 'error_log':
        for line in open(dir_base + subdir, 'r'):
            log.write(line)
    log.close()

    simplify_error_log(log_name, dir_results + 'error_log')



def simplify_error_log(file_in, file_out):

    #convert to a list
    log = []
    numbers = []
    text = False
    barrier_word = '------------\n'
    for line in open(file_in):
        if line == barrier_word:
            text = False
        elif not text:
            numbers.append((int)(line[:-3]))
            log.append('')
            text = True
        else:
            log[-1] += line

    os.remove(file_in)

    #reorganize list
    short_log = []
    short_num = []
    for i in range(len(numbers)):
        if log[i] not in short_log:
            short_log.append(log[i])
            short_num.append([numbers[i], ])
        else:
            short_num[short_log.index(log[i])].append(numbers[i])

    sort = np.argsort([-len(nums) for nums in short_num])
    sorted_short_num, sorted_short_log = [0, ] * len(short_num), ['', ]*len(short_num)
    for i in range(len(short_num)):
        sorted_short_num[i] = short_num[sort[i]]
        sorted_short_log[i] = short_log[sort[i]]

    #print to file
    file = open(file_out + '.txt', 'w')
    for i in range(len(short_num)):
        print(sorted_short_num[i], file = file)
        print(sorted_short_log[i], file=file)
        print(barrier_word, file = file)
    file.close()
    
    np.save(file_out + '_ids.npy', numbers) # save the ids of the stars where the errors occured



    

if __name__ == '__main__':
    
    extra_name = sys.argv[1]
    
    mode= sys.argv[2]
    temp= sys.argv[3]
    
    
    if extra_name == 'start':
        clean(mode, temp)
        
    elif extra_name == 'finish':
        finish(mode, temp, '')

    else:
        finish(mode, temp, extra_name)