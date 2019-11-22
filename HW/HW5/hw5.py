from subprocess import call
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import newton_krylov
import numpy as np
from scipy.stats import norm
import argparse

PETSC_BASE_ARGS = ['-ts_type','rk','-ts_rk_type','4','-ts_max_time','50.0', '-ts_rtol','1e-12']

PETSC_MPIEXEC_DIR = '/home/diffeoinvariant/petsc-3.12.0/arch-linux2-c-debug/bin/'
MAX_K = 4.5
MIN_K = 1.0
K_INCREMENT = 0.05

def form_uniprocess_arglist(n, k, f, mdb=False):
    argls = ['./KuramotoSolver']
    if mdb:
        argls.append('-malloc')
        argls.append('off')
        
    argls += ['-N',str(n), '-K', str(k), '-f', str(f)] + PETSC_BASE_ARGS
    return argls

def form_mpi_arglist(np, N, k, f, memdebug=False):

    if memdebug:
        argls = [str(PETSC_MPIEXEC_DIR+'mpiexec'), '-n',str(np),'valgrind', '--tool=memcheck', '-q', '--log-file=valgrind.log.%p']
    else:
        argls = [str(PETSC_MPIEXEC_DIR+'mpiexec'),'-n',str(np)]
        
    argls += form_uniprocess_arglist(N, k, f, memdebug)
    return argls



def form_resid_function(K, zpts):
    #curry K and zpts into the residual function for newton_krylov

    def resid(r):
        integrand = np.sqrt(1 - zpts * zpts) * norm.pdf(K * zpts * r)
        #integral of `integrand` from -1 to 1 is (1 - (-1)) = 2 times its mean
        # over [-1,1]. This is equiv to integrating w/ the trapezoid rule
        return r - K * r * 2 * integrand.mean()

    return resid



def p1c_solve(K, zpts, init_guess, solver_method='lgmres'):
    resid_func = form_resid_function(K, zpts)
    return newton_krylov(resid_func, init_guess, method=solver_method)

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-p1asolve", type=str, help="Solve the Kuramoto model for problem 1a?", choices=['','no','No' 'False','True','yes','Yes'], default='')
    parser.add_argument("-p1csolve", type=str, help="Solve the integral equation for <r> for problem 1a?", choices=['','no','No' 'False','True','yes','Yes'], default='')
    '''
    fname = "hw5kuramotoFineResolution.csv"

    do_1asolve = True
    do_1csolve = True
    do_p1plots = True
    use_exact_hw_nlist = False

    debug_memory = False#True

    klist = [MIN_K + K_INCREMENT * i for i in range(int((MAX_K - MIN_K)/K_INCREMENT) + 1)]
    nlist = [int(2e6)]

    if use_exact_hw_nlist:
        nlist = [200, 1000, 5000]

    if debug_memory:
        klist = [1.0]
        nlist = [1000]

    if do_1asolve:
        with open(fname, 'w') as hwfile:
            hwfile.write('N,K,r\n')
    
        for n in nlist:
            for k in klist:
                if n > 1e4 or debug_memory:
                    args = form_mpi_arglist(4, n, k, fname, debug_memory)
                else:
                    args = form_uniprocess_arglist(n, k, fname)
                call(args)
                print()

    if do_1csolve:
        num_zpts = 100
        zlow = -1.0
        zhigh = 1.0
        zpoints = np.linspace(zlow, zhigh, num=num_zpts)
    
        init_rguess = np.array([1.0])
        rsol_list = []
    
        for k in klist:
            r = p1c_solve(k, zpoints, init_rguess)
            rsol_list.append({'K':k,'r':r})

        kvals = [val['K'] for val in rsol_list]
        rvals = [val['r'] for val in rsol_list]        
        
        dat = pd.read_csv(fname)
        dat = dat.rename(columns={'N':'N', 'K':'K','r':'<r>'})
    
        n200 = dat[dat['N'] == nlist[0]]
        #n1k = dat[dat['N'] == nlist[1]]
        #n5k = dat[dat['N'] == nlist[2]]
    
        plt.figure()
        plt.plot(n200['K'], n200['<r>'],'b')
        plt.plot(kvals, rvals, 'r')
        plt.xlabel('K')
        plt.ylabel('<r>')
        plt.title("Long-Term Average r vs K with N=2,000,000")
        '''
        plt.figure()
        plt.plot(n1k['K'], n1k['<r>'],'b')
        plt.plot(kvals, rvals, 'r')
        plt.xlabel('K')
        plt.ylabel('<r>')
        plt.title("Long-Term Average r vs K with N=100,000")
    
        plt.figure()
        plt.plot(n5k['K'], n5k['<r>'],'b')
        plt.plot(kvals, rvals, 'r')
        plt.xlabel('K')
        plt.ylabel('<r>')
        plt.title("Long-Term Average r vs K with N=2,000,000")
        '''
        plt.show()
    
    

    
