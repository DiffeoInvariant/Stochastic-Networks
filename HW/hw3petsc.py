import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_petsc_vec(filename, ret_type='numpy'):
    #ret_type can be numpy or petsc
    viewer = PETSc.Viewer().createBinary(filename, 'r')
    v = PETSc.Vec().load(viewer)

    if ret_type == 'numpy':
        return v.getArray()
    elif ret_type == 'petsc':
        return v
    else:
        raise NotImplementedError




if __name__ == '__main__':

    TARG_FILENAMES = ['/home/diffeoinvariant/Stochastic-Networks/HW/ChungLuTargetDDn5000k0100g4.000000.csv', '/home/diffeoinvariant/Stochastic-Networks/HW/ChungLuTargetDDn5000k0100g3.000000.csv', '/home/diffeoinvariant/Stochastic-Networks/HW/ChungLuTargetDDn5000k0100g2.500000.csv','/home/diffeoinvariant/Stochastic-Networks/HW/ChungLuTargetDDn5000k0100g2.000000.csv']
    TARG_GAMMAS = [4.0, 3.0, 2.5, 2.0]

    N = 5000

    KIN_FILENAMES = ['/home/diffeoinvariant/Stochastic-Networks/HW/p2c2In.bin','/home/diffeoinvariant/Stochastic-Networks/HW/p2c3In.bin','/home/diffeoinvariant/Stochastic-Networks/HW/p2c4In.bin','/home/diffeoinvariant/Stochastic-Networks/HW/p2c5In.bin']
    KOUT_FILENAMES = ['/home/diffeoinvariant/Stochastic-Networks/HW/p2c2out.bin','/home/diffeoinvariant/Stochastic-Networks/HW/p2c3out.bin','/home/diffeoinvariant/Stochastic-Networks/HW/p2c4out.bin','/home/diffeoinvariant/Stochastic-Networks/HW/p2c5out.bin']
    targ_deg_data = []
    for fnm, tg in zip(TARG_FILENAMES, TARG_GAMMAS):
        datarray = pd.read_csv(fnm)
        tkin, tkout = datarray['Target in-degree'], datarray[' Target out-degree']
        targ_deg_data.append((tg, tkin,tkout))

    deg_data = []

    for kif, kof in zip(KIN_FILENAMES, KOUT_FILENAMES):
        kin, kout = read_petsc_vec(kif), read_petsc_vec(kof)
        deg_data.append((kin, kout))

    '''
    for dd, tdd in zip(deg_data, targ_deg_data):
        kin, kout = dd
        gamma, tkin, tkout = tdd
            
        plt.figure()
        plt.scatter(tkin, kin)
        plt.xlabel('Target in-degree')
        plt.ylabel('Sampled in-degree')

        plt.figure()
        plt.hist(tkin)
        plt.title('Target in-degree distribution, gamma = %f' % gamma)

        plt.figure()
        plt.hist(kin)
        plt.title('Sampled in-degree distribution, gamma = %f' % gamma)
    '''

    oderes = pd.read_csv('hw3p4res.csv', header=None, names=['K','xnorm'])

    plt.figure()
    plt.plot(oderes['K'], oderes['xnorm'])
    plt.xlabel('K')
    plt.ylabel('xnorm')
    plt.title("Long-Run ||x|| vs K")

    oderes = pd.read_csv('hw3p4fine.csv', header=None, names=['K','xnorm'])

    plt.figure()
    plt.plot(oderes['K'], oderes['xnorm'])
    plt.xlabel('K')
    plt.ylabel('xnorm')
    plt.title("Long-Run ||x|| vs K")
    

    

    plt.show()
    
        
        
        

    
    
