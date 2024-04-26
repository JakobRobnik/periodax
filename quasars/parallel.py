from mpi4py import MPI
import sys, shutil, os, traceback

import numpy as np
from time import time, sleep


def add_args(f, args):
    """gets f with additional arguments in the form with one argument"""    
    if args != None:
        return lambda k: f(k, args)
    else:
        return f



def error_handling(f, base, args= None):
    """converts f to also handle errors"""
    
    _f = add_args(f, args)
    
    def init(rank):
        error_file = base + 'error_log/'+str(rank)+'.txt'
        if os.path.exists(error_file):
            os.remove(error_file)
        error_log = open(error_file, 'w')
        return error_log
    
    def func(k, error_log):
        
        try:
            _f(k)

        except: #if runing f raised an exception we store it to the error_log for later debugging
            print('------------', file = error_log)
            print(str(k) + ': ', file = error_log)
            traceback.print_exception(*sys.exc_info(), limit=10, file=error_log)
            print('------------', file = error_log)
    
    
    def finish(error_log):
        
        error_log.close()
        

    return init, func, finish


def basic(f, args= None):
    """if no error handling is required"""
    _f = add_args(f, args)
    return lambda rank: None, _f, lambda args: None
    

# in the cases below, each worker first calls extra_args = init(), exectue the given jobs f(k, extra_args) and wraps up by calling finish(extra_args).


def for_loop(prep, num_threads, total_jobs, begin = 0):
    """runs = how many calls of f does each thread make.
    f(n) will be evaluated for n in range(begin, begin + (total_jobs//num_threads) * num_threads)"""

    init, f, finish = prep
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        t1 = time()

    extra_args = init(rank)
    
    for i in range(total_jobs//num_threads):
        print('Started: ' + str(begin + rank + num_threads * i))
        sys.stdout.flush()
        f(begin + rank + num_threads*i, extra_args)
        print('Finished: ' + str(begin + rank + num_threads * i))
        sys.stdout.flush()

        
    comm.Barrier()

    if rank == 0:
        print("Total time: " + str(np.round((time() - t1) / 60.0, 2)) + " min")

    finish(extra_args)



def for_loop_with_job_manager(prep, num_threads, total_jobs, begin = 0):
    """evaluates f(n) for n in range(begin, begin + total_jobs)
    by dividing the evaluations among num_threads - 1 workers, one thread is the work manager"""
    
    init, f, finish = prep

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0: #this thread is the job manager
        t1 = time()
        jobs_started = 0
        threads_finished = 0

        #first give everybody a job
        for i in range(1, num_threads):
            comm.send(begin + jobs_started, dest=i)
            jobs_started += 1

        while threads_finished < num_threads - 1: #send jobs until all but the job manager are finished
            state = MPI.Status()
            recieved_message = comm.Iprobe(source= MPI.ANY_SOURCE, status = state) #check if any worker is without work
            if not recieved_message: #wait a bit and check again
                sleep(0.01)
                
            else: #give him a job
                rank = state.Get_source() #which worker is it
                rank = comm.recv(source = rank)
                if jobs_started < total_jobs:
                    comm.send(begin + jobs_started, dest = rank)
                    jobs_started += 1

                else: #if there is no work left
                    comm.send(-1, dest=rank) #send him the termination message
                    threads_finished += 1 #this worker is finished, let's wait for the others to finish what they are doing


        print("Total time: " + str(np.round((time() - t1) / 60.0, 2)) + " min") #everybody is finished

    else: #these are workers
        extra_args = init(rank)
        
        while True:
            i = comm.recv(source=0) #recieve a job

            if i == -1: #if recieved a termination message, stop working
                break
            else: #do a job
                #print('Started: '+str(i))
                f(i, extra_args)
                #print('Finished: '+str(i))
                
            comm.send(rank, dest = 0) #request a new job

        finish(extra_args)

    comm.Barrier()