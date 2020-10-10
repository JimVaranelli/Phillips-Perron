import sys
import os
import multiprocessing as mp
import numpy as np
import time
from ppclass import PhillipsPerron
from arfimasim import ARFIMA_sim


# each thread will perform (end - start) trials:
#   1) generate a non-stationary random AR(1) series
#   2) perform Phillips-Perron unit test on the random
#      series according to the requested model type
#   3) store Z-rho stat in RawArray shared memory
def _cv_thread(start, end, res, nobs, model, procidx):
    print("    _cv_thread...procidx = {}".format(procidx))
    np.random.seed(procidx * 1001)
    pp = PhillipsPerron()
    for idx in range(start, end):
        if idx % 1000 == 0:
            print("      procidx = {}  trial = {}".format(procidx, idx))
        # generate a random non-stationary AR(1) series
        # with nobs=5000 in order to simulate asymptotic
        # test behavior under the null hypothesis
        series = ARFIMA_sim(p_coeffs=[1], q_coeffs=[], d=0, slen=nobs, alpha=0,
                            sigma=1, numseas=100)
        # get the Phillips-Perron Z-rho stat for the series
        z_rho = pp(series, regression=model, trunclag=0)[3]
        res[idx] = z_rho


# save 51 (pval, cv) pairs from (100/trials)% to
# (100-(1/trials))% making sure the 1%, 5%, and 10%
# pvalues are included to avoid interpolation for
# those values.
def _output_cvs(res_nps, outfile, trials):
    delta = int(trials / 50)
    cvs = np.zeros((51, 2))
    for i in range(0, 51):
        # check for 1% CV inclusion
        if i == 1 and np.allclose(cvs[0][0], 1) is False:
            cvs[i][1] = res_nps[int(trials * .01)]
            cvs[i][0] = 1.000000
        # check for 5% CV inclusion
        elif i == 3:
            cvs[i][1] = res_nps[int(trials * .05)]
            cvs[i][0] = 5.000000
        # check for 100% and save as (100 - (1/trials))%
        elif i == 50:
            cvs[i][1] = res_nps[-1]
            cvs[i][0] = 100 * (1 - (1 / trials))
        # all others are multiples of 2%
        else:
            cvs[i][1] = res_nps[i * delta]
            # save the first value as (100/trials)%
            if i == 0:
                cvs[i][0] = 100 / trials
            else:
                cvs[i][0] = 100 * (i * delta) / trials
    print("Writing output to file: {}...".format(outfile))
    np.savetxt(outfile, cvs, fmt=('%0.6f'), delimiter=',')


# monte carlo simulation used to estimate the
# asymptotic critical values for the Z-rho statistic
# in the Phillips-Perron unit root test
def main(model='ct', trials=1000000, nobs=5000, nproc=2):
    print("Running Monte Carlo simulation...model = {}".format(model),
          "  trials = {}  nobs = {}".format(trials, nobs))
    if model not in ['nc', 'c', 'ct']:
        raise ValueError(
            'MC: model option \'{}\' not understood'.format(model))
    if trials < 100:
        raise ValueError('MC: number of trials must be >=100')
    if nobs < 100:
        raise ValueError('MC: number of observations must be >=100')
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    run_dir = os.path.join(cur_dir, "results")
    outfile = os.path.join(run_dir, "zrho_cv_{}.csv".format(model))
    st = time.time()
    # start_period = 0, end_period = periods - 1
    start_period = 0
    end_period = trials
    # set up multi-processing - use 3/4 available CPUs/cores
    # if the desired number of processes is not specified
    # PLEASE NOTE: the multithreading from underlying numpy/BLAS
    # library competes for resources with the processes spawned
    # here. Speedup beyond two processes is unlikely unless
    # numpy multithreading is disabled.
    ncpu = mp.cpu_count()
    print(" Number of available CPUs = {}".format(ncpu))
    if nproc is None:
        nproc = int(3 * mp.cpu_count() / 4)
    # nproc should be in the range [1, ncpu-1]
    if nproc > ncpu - 1:
        nproc = ncpu - 1
    elif nproc < 1:
        nproc = 1
    # number of processes should not be larger
    # than the number of requested trials
    if nproc > trials:
        nproc = trials
    print("  Number of CPUs utilized = {}".format(nproc))
    numperproc = (end_period - start_period) / nproc
    procs = [None] * nproc
    # store stats in shared memory array
    res = mp.RawArray('d', trials)
    startper = start_period
    if nproc > 1:
        endper = startper + int(numperproc)
    else:
        endper = end_period
    tot = startper + numperproc
    for procidx in range(len(procs)):
        print("   Launching process: procidx = {}".format(procidx),
              " startper = {}".format(startper), " endper = {}".format(endper))
        procs[procidx] = mp.Process(
            target=_cv_thread,
            args=(startper, endper, res, nobs, model, procidx))
        procs[procidx].start()
        startper = endper
        endper = startper + int(numperproc)
        tot += numperproc
        if tot - endper >= 1:
            endper += 1
    for proc in range(len(procs)):
        procs[proc].join()
    res_np = np.frombuffer(res).reshape(trials, 1)
    # sort in ascending order
    res_nps = np.sort(res_np, axis=0)
    # output the CVs
    _output_cvs(res_nps, outfile, trials)
    print("    time = {0:0.5f}".format(time.time() - st))


if __name__ == "__main__":
    sys.exit(int(main() or 0))
