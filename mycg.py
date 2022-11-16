import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp


            

def mycg(A, b, x0, t, tol=1e-5, maxiter=None, convcontrol=None):
    b = b*x0
    n = A.shape[0]
    if maxiter==None:
        maxiter = n*10
    x = np.zeros(A.shape[0])
    r = b-A*x
    p = r
    alpha = np.inner(r,r)

    ks = []
    e_iters = []
    e_iter = 0
    e_discs = []
    e_disc = 0
    k=0
    exit_code = -1
    while (exit_code < 0):
        v = A*p
        lam = alpha/np.inner(v,p)
        x = x+lam*p
        r = r-lam*v
        alphanew = np.inner(r,r)
        p = r+alphanew/alpha*p
        alpha = alphanew
        k+=1
        ks.append(k)
        if  convcontrol is not None:
            e_disc, e_iter = convcontrol(x,x0)
            e_iters.append(e_iter)
            e_discs.append(e_disc)
            if abs(e_iter) < abs(e_disc):
                exit_code = 2
        elif(np.sqrt(alpha)<=tol):
            exit_code = 0
        if(k>=maxiter):
            exit_code = 1
    if False:
        plt.plot(ks,e_iters,label='iteration error estimate')
        plt.plot(ks,e_discs, label='discretization error estimate')
        plt.yscale('log')
        plt.legend()
        plt.show()
    return [x,exit_code, k, np.asarray([e_iter, e_disc])]
