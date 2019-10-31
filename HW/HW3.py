import numpy as np
import matplotlib.pyplot as plt

def random_adjmat(n, p):
    return np.random.binomial(n*n, p, size=(n,n))

def plot_spectrum(A):
    evals = np.linalg.eigvals(A)
    plt.figure()
    plt.scatter(np.real(evals), np.imag(evals))

def perturb_zeros(A, eps):
    #NOTE: we perturb the zeros by _negative_ eps
    pmat = np.ones_like(A, dtype='float64')
    #matrix of ones where A has zeros
    pmat -= A

    pmat *= eps

    return A - pmat
    


if __name__ == '__main__':
    A = random_adjmat(500, 0.05)

    print(A.size)

    plot_spectrum(A)

    eps = 1e-5

    B = perturb_zeros(A, eps)

    plot_spectrum(B)

    plt.show()
    

    

