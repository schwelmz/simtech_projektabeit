from tkinter import E
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

###################################################################################################################################################################################################
#auxiliary methods
###################################################################################################################################################################################################
def arr2str(arr, **kwargs):
    return np.array2string(arr, formatter={'float_kind': lambda x: '{:+.2e}'.format(x)}, **kwargs).replace('+0.00e+00', '    -    ')

def numberofnodes(i,n0):
    nodes = n0
    for i in range(0,i):
        nodes = nodes*2-1
    return nodes

def make_fine(u, iter, xsi, Nx_fine, xs_fine):
    #make finer discretization
    Nx = u.shape[0]
    #linear interpolation in the intervals to make u finer
    nodes_in_interval = numberofnodes(iter,2)-2
    u_filled = np.zeros(Nx_fine)
    for j in range(0,Nx-1):
        [m,c] = first_order_coefficients(j, u, xsi)
        k0 = j*(nodes_in_interval+1)
        for k in range(0, nodes_in_interval+2):
            u_filled[k0+k] = c + m*xs_fine[k0+k]
    return u_filled

###################################################################################################################################################################################################
#methods for error estimation
###################################################################################################################################################################################################
def make_discretization(Nx,x_start,x_end):
    Ne = Nx-1
    xs = np.linspace(x_start,x_end,Nx)
    hxs = xs[1:]-xs[:-1]
    print('Created discretization for Nx =',Nx,'nodes.')
    return[Ne, xs, hxs]

def make_laplace(N, hx = 1, bounds=None):
    if hasattr(hx, "__len__"):
        """
        0    1    2
        |----|----|----|--
        h0   h1   h2
        """
        assert(len(hx) == N - 1), f"len(hx) = {len(hx)}, N = {N}"
        h = lambda i: hx[i]
    else:
        h = lambda i: hx


    rows = []
    cols = []
    vals = []
    for i in range(N):
        if bounds=='skip' and i in [0, N-1]:
            continue
        if bounds=='dirichlet' and i in [0, N-1]:
            rows.append(i)
            cols.append(i)
            vals.append(1)
            continue
        if bounds=='neumann' and i in [0, N-1]:
            if i == 0:
                rows.append(i)
                cols.append(i)
                vals.append(-1/h(0))
                rows.append(i)
                cols.append(i+1)
                vals.append(1/h(0))
            else:
                rows.append(i)
                cols.append(i-1)
                vals.append(1/h(N-2))
                rows.append(i)
                cols.append(i)
                vals.append(-1/h(N-2))
            continue

        if i != 0:
            rows.append(i)
            cols.append(i-1)
            vals.append(1/h(i-1)) # ∇φ_i ∇φ_i-1
        rows.append(i)
        cols.append(i)
        vals.append(-1/h(i-1) - 1/h(i)) # ∇φ_i ∇φ_i
        if i != N-1:
            rows.append(i)
            cols.append(i+1)
            vals.append(1/h(i)) # ∇φ_i ∇φ_i+1
    # negate as Δ = - <∇φ, ∇φ>
    return -sparse.csr_matrix((vals, (rows, cols)), shape=(N,N))

def make_lhs_rhs(Nx, hxs):
    #lhs
    lhs = make_laplace(Nx, hxs, bounds='dirichlet')
    #rhs
    rhs = np.zeros(Nx)
    rhs[1:-1] = hxs[1:]
    return [lhs, rhs]

###################################################################################################################################################################################################
#methods for error estimation
###################################################################################################################################################################################################
def second_order_coefficients(j, u, x):
    xi = [x[j-1], x[j], x[j+1]]
    ui = [u[j-1], u[j], u[j+1]]
    a = ui[0]*xi[1]*xi[2]/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]*xi[0]*xi[2]/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]*xi[0]*xi[1]/((xi[2]-xi[0])*(xi[2]-xi[1]))
    b = -(ui[0]*(xi[1]+xi[2])/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]*(xi[0]+xi[2])/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]*(xi[0]+xi[1])/((xi[2]-xi[0])*(xi[2]-xi[1])))
    c = ui[0]/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]/((xi[2]-xi[0])*(xi[2]-xi[1]))
    return [c,b,a]

def first_order_coefficients(j, u, x):
    x = [x[j], x[j+1]]
    u = [u[j], u[j+1]]
    c = (u[1]*x[0]-u[0]*x[1])/(x[0]-x[1])
    m = (u[0]-u[1])/(x[0]-x[1])
    return [m, c]

def calc_iter_error(x, z, u, Ne, hx):
    if hasattr(hx, "__len__"):
        h = lambda i: hx[i]
    else:
        h = lambda i: hx
    u_prime = (u[1:] - u[:-1]) / hx # on each element
    z_prime = (z[1:] - z[:-1]) / hx # on each element
    #print('p', u_prime)
    jumps_u = u_prime[1:] - u_prime[:-1] # on the inner nodes
    jumps_z = z_prime[1:] - z_prime[:-1] # on the inner nodes
    #print('j', jumps_u)
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    for ie in range(Ne):
        t1 += h(ie) * (z[ie] + z[ie+1])/2
        [m, c] = first_order_coefficients(ie, z, x)
        #m = (z[ie+1]-z[ie])/(x[ie+1]-x[ie])
        #c = z[ie+1]-(z[ie]-z[ie+1])/(x[ie]-x[ie+1])*x[ie+1]
        t4 += c*(x[ie+1]-x[ie]) + 1/2*m*(x[ie+1]**2-x[ie]**2)
        if ie != 0:
            t2 += .5 * jumps_u[ie - 1] * z[ie]
        if ie != Ne - 1:
            t3 += .5 * jumps_u[ie] * z[ie+1]
    #print(t1, t4, t2, t3)
    #print(t1 + t2 + t3)
    return (t1+t2+t3)

def calc_disc_error(x, z, Ne, hx):
    if hasattr(hx, "__len__"):
        h = lambda i: hx[i]
    else:
        h = lambda i: hx
    t1 = 0
    t2 = 0
    for ie in range(0, Ne):
        [a,b,c] = second_order_coefficients(ie, z, x)
        t1 += a/3*(x[ie+1]**3-x[ie]**3) + b/2*(x[ie+1]**2-x[ie]**2) + c*(x[ie+1]-x[ie])
        t2 += -h(ie) * (z[ie] + z[ie+1])/2
    return (t1+t2)

def calc_error_exact(u, u_exact, Ne, hx):
    if hasattr(hx, "__len__"):
        h = lambda i: hx[i]
    else:
        h = lambda i: hx

    error = 0
    for ie in range(Ne):
        t1 = (u_exact[ie]+u_exact[ie+1]-u[ie]-u[ie+1])/2*h(ie)
        error += t1
    return error

###################################################################################################################################################################################################
#main
###################################################################################################################################################################################################
def test_iter_est():
    #Create discretization
    Nx = 200
    Ne, xs, hxs = make_discretization(Nx,0,1)

    #Create system matrices
    lhs, rhs = make_lhs_rhs(Nx, hxs)
    #print('left hand solution:')
    #print('  '+arr2str(lhs.todense(), prefix='  '))
    #print('right hand solution:')
    #print(rhs)

    #Solve the dual problem
    zh = sparse.linalg.cg(lhs, rhs, tol=1e-10)
    assert zh[1]==0; zh = zh[0]
    #print('solution z_h of the dual problem')
    #print(zh)

    #exact solution
    u_exact = -1/2*(xs**2-xs)

    #error convergence tests
    print('Initializing test for iteration error convergence...')
    k = 10
    iter_error = 1
    disc_error = 0

    iters = []
    iter_error_list = []
    disc_error_list = []
    iter_error_exact_list = []

    while iter_error > disc_error:
        #compute the approximate solution
        u = sparse.linalg.cg(lhs, rhs, maxiter = k, tol=0)
        print('Calculating error after ',u[1],'CG iterations.')
        u = u[0]
        #error calculation
        h = hxs[1]  #!!!!!!!!!!!!!!
        iter_error = calc_iter_error(xs, zh, u, Ne, hxs)
        disc_error = calc_disc_error(xs, zh, Ne, hxs)
        iter_error_exact = calc_error_exact(u, u_exact, Ne, hxs)
        
        iter_error_list.append(iter_error)
        disc_error_list.append(disc_error)
        iter_error_exact_list.append(iter_error_exact)
        k += 5
        iters.append(k)

    #plot
    print('list of iter error estimates:')
    print(iter_error_list)
    print('list of exact iter errors:')
    print(iter_error_exact_list)
    print('2norm of exact iter error - est iter error',np.linalg.norm(np.array(iter_error_list)-np.array(iter_error_exact_list)))
    plt.plot(iters, np.array(iter_error_list),'magenta', label = 'estimated iteration error')
    plt.plot(iters, np.array(disc_error_list),'orange', label = 'estimated discretization error')
    plt.plot(iters, np.array(iter_error_exact_list),'k--', label = 'exact iteration error')
    plt.xlabel('#iterations')
    plt.yscale('log')
    plt.legend()
    plt.show()

def test_disc_est(imax, N0):
    print('Initializing test for discretization error convergence...')
    #exact solution
    Nx_fine = numberofnodes(imax, N0)
    xs_fine = np.linspace(0,1,Nx_fine)
    u_exact = -1/2*(xs_fine**2-xs_fine)

    Nx_list = []
    e_disc_est_list = []
    e_disc_exact_list = []
    for i in range(0,6):
        Nx = numberofnodes(i, 50)
        #make discretization
        Ne, xs, hxs = make_discretization(Nx, 0, 1)

        #create system matrices
        lhs, rhs = make_lhs_rhs(Nx, hxs)

        #solve the dual problem
        zh = sparse.linalg.cg(lhs, rhs, tol=1e-10)
        assert zh[1]==0; zh = zh[0]

        #solve the discretized problem exactly
        uh_exact = sparse.linalg.cg(lhs, rhs, tol=1e-10)
        assert uh_exact[1] == 0; uh_exact = uh_exact[0]

        #calculate exact disc error and disc error estimation
        h = hxs[1] #!!!!!!!!!!!!!!!
        e_disc_est = calc_disc_error(xs, zh, Ne, hxs)
        uh_exact_fine = make_fine(uh_exact, imax-i, xs, Nx_fine, xs_fine)
        e_disc_exact = calc_error_exact(uh_exact_fine, u_exact, Ne, hxs)

        #append
        Nx_list.append(Nx)
        e_disc_est_list.append(e_disc_est)
        e_disc_exact_list.append(e_disc_exact)
    
    #plot results
    print('list of disc error estimates:')
    print(e_disc_est_list)
    print('list of exact disc errors:')
    print(e_disc_exact_list)
    print('2norm of exact disc error - est disc error', np.linalg.norm(np.asarray(e_disc_est_list)-np.asarray(e_disc_exact_list)))
    plt.plot(np.array(Nx_list), np.array(e_disc_est_list),'orange', label = 'disc error (estimate)')
    plt.plot(np.array(Nx_list), np.array(e_disc_exact_list),'black', label = 'disc error (exact)')
    plt.yscale('log', base=2)
    plt.xscale('log', base=2)
    plt.xlabel('#iterations')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test_iter_est()
    print('------------------------------------------------------------------------------------')
    test_disc_est(6, 50)