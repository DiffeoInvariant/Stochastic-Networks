from subprocess import call
import matplotlib.pyplot as plt
import pandas as pd
PETSC_BASE_ARGS = ['-ts_max_steps','10000','-ts_max_time','100.0', 'ts_type','TSRK']


MAX_K = 4.5
K_INCREMENT = 0.025

def form_uniprocess_arglist(n, k, f):
    argls = ['./HW5','-N',str(n), '-K', str(k), '-f', str(f)] + PETSC_BASE_ARGS
    return argls


if __name__ == '__main__':

    fname = "hw5kuramoto.csv"
    
    with open(fname, 'w') as hwfile:
        hwfile.write('N,K,r\n')
    
    nlist = [200, 1000, 5000]
    klist = [K_INCREMENT * i for i in range(int(MAX_K/K_INCREMENT) + 1)]
    
    for n in nlist:
        for k in klist:
            args = form_uniprocess_arglist(n, k, fname)
            call(args)

            
    
    dat = pd.read_csv(fname)
    dat = dat.rename(columns={'N':'N', 'K':'K','r':'<r>'})
    
    n200 = dat[dat['N'] == 200]
    n1k = dat[dat['N'] == 1000]
    n5k = dat[dat['N'] == 5000]

    plt.figure()
    plt.plot(n200['K'], n200['<r>'])
    plt.xlabel('K')
    plt.ylabel('<r>')
    plt.title("Long-Term Average r vs K with N=200")

    plt.figure()
    plt.plot(n1k['K'], n1k['<r>'])
    plt.xlabel('K')
    plt.ylabel('<r>')
    plt.title("Long-Term Average r vs K with N=1000")
    
    plt.figure()
    plt.plot(n5k['K'], n5k['<r>'])
    plt.xlabel('K')
    plt.ylabel('<r>')
    plt.title("Long-Term Average r vs K with N=5000")

    plt.show()
    
    

    
