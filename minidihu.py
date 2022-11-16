from ctypes.wintypes import HTASK
import sys
from tkinter import N
from turtle import title
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
from functools import lru_cache
import matplotlib.pyplot as plt
from mycg import mycg

############################################################################################
# auxiliary methods
############################################################################################
def extract_channels(file):
    import numpy as np
    if file.endswith('.py'):
        channel_names = ['membrane/V', 'sodium_channel_m_gate/m', 'sodium_channel_h_gate/h', 'potassium_channel_n_gate/n']
        return extractX(file, 'solution', channel_names)
    raise "FileType not understood: "+file

def extract_xyz(file):
    import numpy as np
    if file.endswith('.py'):
        channel_names = ['membrane/V', 'sodium_channel_m_gate/m', 'sodium_channel_h_gate/h', 'potassium_channel_n_gate/n']
        return extractX(file, 'solution', channel_names)
    raise "FileType not understood: "+file

def extractX(file, name, channel_names):
    # name = solution or geometry
    import numpy as np
    if file.endswith('.py'):
        # data = py_reader.load_data([file])
        data = np.load(file, allow_pickle=True)
        data = [data]
        if data == []:
            print("  \033[1;31mCould not parse file\033[0m '{}'".format(file))
        # data[0]['timeStepNo'] is current time step number IN THE SPLITTING (i.e. not global)
        # data[0]['currentTime'] is current SIMULATION time
        tcurr = data[0]['currentTime']
        print('  Loaded data at simulation time:', '\033[1;92m'+str(tcurr)+'\033[0m') # bold green
        data = data[0]['data']
        solution  = next(filter(lambda d: d['name'] == name, data))
        componentX = lambda x: next(filter(lambda d: d['name'] == str(x), solution['components']))
        return {'val':np.vstack([componentX(i)['values'] for i in channel_names]).T, 'tcurr':tcurr}
    raise "FileType not understood: "+file

def arr2str(arr, **kwargs):
        return np.array2string(arr, formatter={'float_kind': lambda x: '{:+.2e}'.format(x)}, **kwargs).replace('+0.00e+00', '    -    ')

def compute_error(u1, u2, hx):
    if hasattr(hx, "__len__"):
        h = lambda i: hx[i]
    else:
        h = lambda i: hx
    
    N = u1.shape[0]
    f = 2**(-(np.linspace(-10,10,N))**2)
    error = 0
    for i in range(0, N-1):
        error +=  f[i]*((u1[i+1]-u2[i+1]) + (u1[i]-u2[i])) / 2 * h(i)
    return abs(error)

def O2_coeff(j, u, x):
    xi = [x[j-1], x[j], x[j+1]]
    ui = [u[j-1], u[j], u[j+1]]
    a = ui[0]*xi[1]*xi[2]/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]*xi[0]*xi[2]/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]*xi[0]*xi[1]/((xi[2]-xi[0])*(xi[2]-xi[1]))
    b = -(ui[0]*(xi[1]+xi[2])/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]*(xi[0]+xi[2])/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]*(xi[0]+xi[1])/((xi[2]-xi[0])*(xi[2]-xi[1])))
    c = ui[0]/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]/((xi[2]-xi[0])*(xi[2]-xi[1]))
    return [a,b,c]  #a+b*x+c*x²

def O1_coeff(j, u, x):
    x = [x[j], x[j+1]]
    u = [u[j], u[j+1]]
    a = (u[1]*x[0]-u[0]*x[1])/(x[0]-x[1])
    b = (u[0]-u[1])/(x[0]-x[1])
    return [a, b]   #a+b*x

def numberofnodes(i,n0):
    nodes = n0
    for i in range(0,i):
        nodes = nodes*2-1
    return nodes

def make_fine(u_start, xs_start, factor):
    #make finer discretization
    Nx = u_start.shape[0]
    Nx_fine = numberofnodes(factor,Nx) #Nx = 1191 Nx_fine(3,Nx)=9521
    xs_fine = np.linspace(0,11.9, Nx_fine)
    #linear interpolation in the intervals to make u finer
    nodes_in_interval = numberofnodes(factor,2)-2    #7
    u_filled = np.zeros(Nx_fine)
    for j in range(0,Nx-1):
        [a,b] = O1_coeff(j, u_start, xs_start)
        k0 = j*(nodes_in_interval+1)
        for k in range(0, nodes_in_interval+2):
            u_filled[k0+k] = a + b*xs_fine[k0+k]
    return u_filled, Nx_fine
    
def create_initial_fiber(initial_value_file, Nx, x_start, x_end, Vmhn_initial=None):
    if initial_value_file != '':
        print('initial values:', initial_value_file)
        xyz = extractX(initial_value_file, 'geometry', 'xyz')['val'] # fiber location in space
        Vmhn0 = extract_channels(initial_value_file)['val'] # initial values
        hxs = np.sum((xyz[1:,:] - xyz[:-1,:])**2, axis=1)**.5
        xs = np.zeros(hxs.shape[0] + 1) # 1D location
        xs[1:] = np.cumsum(hxs)
        Nx = xs.shape[0]
        print("Loaded fiber")
    else:
        xs = np.linspace(x_start,x_end, Nx)
        hxs = xs[1:] - xs[:-1]
        #initial values
        Vmhn0 = np.zeros((Nx, 4))
        if Vmhn_initial is not None:
            Vmhn0[:,0] = Vmhn_initial[:,0]
            Vmhn0[:,1] = Vmhn_initial[:,1]
            Vmhn0[:,2] = Vmhn_initial[:,2]
            Vmhn0[:,3] = Vmhn_initial[:,3]
        else:
            Vmhn0[:,0] = -75.0,
            Vmhn0[Nx//2 - 3 : Nx//2 + 3, 0] = 50
            Vmhn0[:,1] =   0.05,
            Vmhn0[:,2] =   0.6,
            Vmhn0[:,3] =   0.325,
        print(f"Created fiber for Nx={Nx}")
    if False:
        print(f"  length: {xs[-1]:>5.2f}cm")
        print(f"  nodes:  {Nx:>4}")
        print(f"Initial values:")
        print(f"  V: {np.min(Vmhn0[:,0]):>+.2e} -- {np.max(Vmhn0[:,0]):>+.2e}")
        print(f"  m: {np.min(Vmhn0[:,1]):>+.2e} -- {np.max(Vmhn0[:,1]):>+.2e}")
        print(f"  h: {np.min(Vmhn0[:,2]):>+.2e} -- {np.max(Vmhn0[:,2]):>+.2e}")
        print(f"  n: {np.min(Vmhn0[:,3]):>+.2e} -- {np.max(Vmhn0[:,3]):>+.2e}")
        print(f"Model parameters:")
        print(f"  sigma: {Conductivity:>7.3f}")
        print(f"  Am:    {Am:>7.3f}")
        print(f"  Cm:    {Cm:>7.3f}")
    return [Nx, xs, hxs, Vmhn0]

############################################################################################
# create matrices
############################################################################################
"""
model parameters for diffusion term
"""
Conductivity = 3.828    # sigma, conductivity [mS/cm]
Am = 500.0              # surface area to volume ratio [cm^-1]
Cm = 0.58               # membrane capacitance [uF/cm^2]
prefactor = Conductivity / Am / Cm
print('prefactor =', prefactor)

def rhs_hh(Vmhn, style='numpy'):
    # numpy takes [[v,m,h,n]]*N: shape=(n,4)
    # jax takes [[v,m,h,n]]*N: shape=(n,4)
    # jax1 takes [v,m,h,n]: shape=(4)
    # sympy only variables [[v,m,h,n]]
    if style=='numpy':
        exp = np.exp
        pow = np.power
        STATES = Vmhn.T
        array = np.array
    elif style=='sympy':
        exp = sp.exp
        pow = lambda x,y: x**y
        STATES = Vmhn[0]
        array = lambda a: np.empty(a,dtype=object)
    elif style=='jax':
        exp = jnp.exp
        pow = jnp.power
        STATES = Vmhn.T
        array = jnp.array
    elif style=='jax1':
        exp = jnp.exp
        pow = jnp.power
        STATES = Vmhn
        array = jnp.array
    else:
        raise RuntimeError(f"Unknown array style '{style}'")

    # copied from inputs/hodgkin_huxley_1952.c

    # init constants
    # STATES[0] = -75;
    CONSTANTS_0 = -75;
    CONSTANTS_1 = 1;
    CONSTANTS_2 = 0;
    CONSTANTS_3 = 120;
    # STATES[1] = 0.05;
    # STATES[2] = 0.6;
    CONSTANTS_4 = 36;
    # STATES[3] = 0.325;
    CONSTANTS_5 = 0.3;
    CONSTANTS_6 = CONSTANTS_0+115.000;
    CONSTANTS_7 = CONSTANTS_0 - 12.0000;
    CONSTANTS_8 = CONSTANTS_0+10.6130;

    # compute rates
    ALGEBRAIC_1 = ( - 0.100000*(STATES[0]+50.0000))/(exp(- (STATES[0]+50.0000)/10.0000) - 1.00000);
    ALGEBRAIC_5 =  4.00000*exp(- (STATES[0]+75.0000)/18.0000);
    RATES_1 =  ALGEBRAIC_1*(1.00000 - STATES[1]) -  ALGEBRAIC_5*STATES[1];
    ALGEBRAIC_2 =  0.0700000*exp(- (STATES[0]+75.0000)/20.0000);
    ALGEBRAIC_6 = 1.00000/(exp(- (STATES[0]+45.0000)/10.0000)+1.00000);
    RATES_2 =  ALGEBRAIC_2*(1.00000 - STATES[2]) -  ALGEBRAIC_6*STATES[2];
    ALGEBRAIC_3 = ( - 0.0100000*(STATES[0]+65.0000))/(exp(- (STATES[0]+65.0000)/10.0000) - 1.00000);
    ALGEBRAIC_7 =  0.125000*exp((STATES[0]+75.0000)/80.0000);
    RATES_3 =  ALGEBRAIC_3*(1.00000 - STATES[3]) -  ALGEBRAIC_7*STATES[3];
    ALGEBRAIC_0 =  CONSTANTS_3*pow(STATES[1], 3.00000)*STATES[2]*(STATES[0] - CONSTANTS_6);
    ALGEBRAIC_4 =  CONSTANTS_4*pow(STATES[3], 4.00000)*(STATES[0] - CONSTANTS_7);
    ALGEBRAIC_8 =  CONSTANTS_5*(STATES[0] - CONSTANTS_8);
    RATES_0 = - (- CONSTANTS_2+ALGEBRAIC_0+ALGEBRAIC_4+ALGEBRAIC_8)/CONSTANTS_1;

    RATES = array([RATES_0, RATES_1, RATES_2, RATES_3])
    return RATES.T

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
    # return as Δ =  <∇φ, ∇φ>
    return -sparse.csr_matrix((vals, (rows, cols)), shape=(N,N))

def make_lumped_mass_matrix(N, hx):
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
        lumped_val = 0
        if i != 0:
            lumped_val += h(i-1) / 3 # φ_i|_left φ_i|_left
            lumped_val += h(i-1) / 6 # φ_i|_left φ_i-1|_right
        if i != N-1:
            lumped_val += h(i) / 3 # φ_i|_right φ_i|_right
            lumped_val += h(i) / 6 # φ_i|_right φ_i+1|_left
        rows.append(i)
        cols.append(i)
        vals.append(lumped_val)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(N,N))

def make_mass_matrix(N, hx):
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
        m_ii = 0
        m_il = 0
        m_ir = 0
        lumped_val = 0
        if i != 0:
            m_ii += h(i-1) / 3 # φ_i|_left φ_i|_left
            m_il += h(i-1) / 6 # φ_i|_left φ_i-1|_right
            lumped_val += h(i-1) / 3 # φ_i|_left φ_i|_left
            lumped_val += h(i-1) / 6 # φ_i|_left φ_i-1|_right
        if i != N-1:
            m_ii += h(i) / 3 # φ_i|_right φ_i|_right
            m_ir += h(i) / 6 # φ_i|_right φ_i+1|_left
            lumped_val += h(i) / 3 # φ_i|_right φ_i|_right
            lumped_val += h(i) / 6 # φ_i|_right φ_i+1|_left
        rows.append(i)
        cols.append(i)
        vals.append(m_ii)
        if m_ir != 0:
            rows.append(i)
            cols.append(i+1)
            vals.append(m_ir)
        if m_il != 0:
            rows.append(i)
            cols.append(i-1)
            vals.append(m_il)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(N,N))

def create_stiffness(Nx, hxs, inner=False):
    # cite opendihu in crank_nicolson.tpp: "compute the system matrix (I - dt*M^{-1}K) where M^{-1} is the lumped mass matrix"
    # matrices marked with `[opendihu: ...]` have been checked to match the opendihu matrices on 18.06.2021 with
    #    env PETSC_OPTIONS="-mat_view ascii" ./multiple_fibers_CN ../settings_multiple_fibers.py --nt_0D 1 --nt_1D 1 --dt_splitting 1e-3 | less
    # and the simulation outputs match in picture-norm.
    # The effect, that the V-channel increases just before the end of the fibers is also present in the opendihu results.
    # opendihu commit: 44cadd4060552f6d1ad4e89153f37d1b843800da
    #laplace:
    prefactor = Conductivity / Am / Cm # mS / uF = [mS/cm] / ([cm^-1] * [uF/cm^2])
    if inner == False:
        laplace = make_laplace(Nx, hxs, bounds='neumann')
        #print("Laplace")
        #print('  '+arr2str(laplace.todense(), prefix='  '))
        stiffness = prefactor * laplace
        #print("Stiffness \033[90m[opendihu: `Mat Object: stiffnessMatrix`]\033[m")
        #print('  '+arr2str(stiffness.todense(), prefix='  '))
        return stiffness
    elif inner == True:
        laplace = make_laplace(Nx, hxs, bounds='skip')
        laplace_inner = laplace[1:-1,1:-1]
        laplace_inner[0,0] /= 2
        laplace_inner[-1,-1] /= 2
        #print("Laplace_inner")
        #print('  '+arr2str(laplace_inner.todense(), prefix='  '))
        stiffness_inner = prefactor * laplace_inner
        #print("Stiffness_inner \033[90m[opendihu: `Mat Object: stiffnessMatrix`]\033[m")
        #print('  '+arr2str(stiffness_inner.todense(), prefix='  '))
        return stiffness_inner

def create_mass(Nx, hxs, inner=False):
    if inner == False:
        mass = make_mass_matrix(Nx, hxs)
        #print("Mass \033[90m[opendihu: `Mat Object: inverseLumpedMassMatrix`]\033[m")
        #print('  '+arr2str(mass.todense(), prefix='  '))
        return mass
    elif inner == True:
        mass_inner = make_mass_matrix(Nx, hxs)[1:-1,1:-1]
        mass_inner[0,0] = 4/3*hxs[0]
        mass_inner[-1,-1] = 4/3*hxs[-1]
        #print("Mass_inner \033[90m[opendihu: `Mat Object: inverseLumpedMassMatrix`]\033[m")
        #print('  '+arr2str(mass_inner.todense(), prefix='  '))
        return mass_inner

############################################################################################
# time stepping methods
############################################################################################
def heun_step(Vmhn, rhs, t, ht):
    Vmhn0 = Vmhn + ht * rhs(Vmhn, t)
    Vmhn1 = Vmhn + ht * rhs(Vmhn0, t + ht)
    return (Vmhn0 + Vmhn1) / 2

def crank_nicolson_FE_step(Vmhn0, sys_expl_impl, t, ht, error_est=None, maxit=1000, eps=1e-10):
    # get explicit and implicit system matrix
    cn_sys_expl, cn_sys_impl = sys_expl_impl
    Vmhn0 = np.array(Vmhn0)

    # only act on V channel
    V0 = Vmhn0[:,0]             #V0
    Nx = Vmhn0.shape[0]

    lhs = cn_sys_impl(ht,Nx)
    rhs = cn_sys_expl(ht,Nx)
    Vmhn0[1:-1,0], exit_code, iters, e_ests = mycg(lhs, rhs, V0[1:-1], t, maxiter=maxit, tol=eps, convcontrol=error_est)        #V1
    print('exit_code=',exit_code,'CG Iterations',iters,'t=',t)
    Vmhn0[0,0] = Vmhn0[1,0]
    Vmhn0[-1,0] = Vmhn0[-2,0]
    return Vmhn0, iters, e_ests

def stepper(integator, Vmhn0, rhs, t0, t1, ht, traj=False, **kwargs):
    Vmhn = Vmhn0

    if not traj:
        result = Vmhn
    else:
        result = [Vmhn]

    n_steps = max(1, int((t1-t0)/ht + 0.5)) # round to nearest integer
    ht_ = (t1-t0) / n_steps

    iters_list = []
    e_iters = []
    e_discs = []
    for i in range(n_steps):
        #print(i, '/', n_steps, 'timesteps')
        Vmhn, iters, e_est = integator(Vmhn, rhs, t0+i*ht_, ht_, **kwargs)
        iters_list.append(iters)
        e_iters.append(e_est[0])
        e_discs.append(e_est[1])
        if not traj:
            result = Vmhn
        else:
            result.append(Vmhn)
    return np.asarray(result), np.asarray(iters_list), [e_iters, e_discs] # cast list to array if we store the trajectory

def strang_step_1H_1CN_FE(Vmhn0, rhs, t, ht, **kwargs):
    # unpack rhs for each component
    rhs_reaction, system_matrices_expl_impl = rhs

    # 1/2 interval for reaction term with Heun
    Vmhn = heun_step(Vmhn0, rhs_reaction, t, ht/2)
    # 1 interval for diffusion with Crank-Nicolson
    Vmhn, iters, e_ests = crank_nicolson_FE_step(Vmhn, system_matrices_expl_impl, t, ht, **kwargs)
    # 1/2 interval for reaction term with Heun
    Vmhn = heun_step(Vmhn, rhs_reaction, t+ht/2, ht/2)
    return Vmhn, iters, e_ests

def strang_1H_1CN_FE(Vmhn, rhs0, system_matrices_expl_impl, t0, t1, hts, maxit=1000, eps=1e-10, traj=False, error_est=None):
    return stepper(strang_step_1H_1CN_FE, Vmhn, (rhs0, system_matrices_expl_impl), t0, t1, hts, error_est=error_est, maxit=maxit, eps=eps, traj=traj)

############################################################################################
# error estimation
############################################################################################
def disc_error_est(u0,u1,z,x):
    Nx = x.shape[0]
    error = 0
    ix = 1
    while (ix < Nx-1):
        #Interpolation 2.Ordnung
        [a2,b2,c2] = O2_coeff(ix, z, x)
        #left side
        [a0,b0] = O1_coeff(ix-1, u0, x)
        [a1,b1] = O1_coeff(ix-1, u1, x)
        [a3,b3] = O1_coeff(ix-1, z, x)
        t1 = (a0-a1)*(a2-a3)*(x[ix]-x[ix-1]) + 1/2*((a0-a1)*(b2-b3)+(b0-b1)*(a2-a3))*(x[ix]**2-x[ix-1]**2) + 1/3*((a0-a1)*c2+(b0-b1)*(b2-b3))*(x[ix]**3-x[ix-1]**3) + 1/4*(b0-b1)*c2*(x[ix]**4-x[ix-1]**4)
        #right side
        [a0,b0] = O1_coeff(ix, u0, x)
        [a1,b1] = O1_coeff(ix, u1, x)
        [a3,b3] = O1_coeff(ix, z, x)
        t2 = (a0-a1)*(a2-a3)*(x[ix+1]-x[ix]) + 1/2*((a0-a1)*(b2-b3)+(b0-b1)*(a2-a3))*(x[ix+1]**2-x[ix]**2) + 1/3*((a0-a1)*c2+(b0-b1)*(b2-b3))*(x[ix+1]**3-x[ix]**3) + 1/4*(b0-b1)*c2*(x[ix+1]**4-x[ix]**4)
        #update
        error += t1+t2
        ix += 2
    #plt.show()
    return error

def iter_error_est(u0,u1,z,x,hx,ht,prefactor):
    Nx = x.shape[0]
    Ne = Nx-1
    #jumps
    u1_prime = (u1[1:] - u1[:-1]) # on each element
    for ie in range(0, Ne):
        u1_prime[ie] = u1_prime[ie]/hx[ie]
    jumps_u1 = u1_prime[1:] - u1_prime[:-1] # on the inner nodes

    error = 0
    for ie in range(Ne):
        t2 = 0
        t3 = 0
        [a0,b0] = O1_coeff(ie, u0, x)
        [a1,b1] = O1_coeff(ie, u1, x)
        [c,d] = O1_coeff(ie, z, x)
        t1 = (a0-a1)*c*(x[ie+1]-x[ie]) + 1/2*((a0-a1)*d+(b0-b1)*c)*(x[ie+1]**2-x[ie]**2) + 1/3*(b0-b1)*d*(x[ie+1]**3-x[ie]**3)
        if ie != 0:
            t2 = prefactor * ht * 0.5 * jumps_u1[ie - 1] * z[ie]
        if ie != Ne - 1:
            t3 = prefactor * ht * 0.5 * jumps_u1[ie] * z[ie+1]
        error += t1+t2+t3
    return error

############################################################################################
# main
############################################################################################

def main(tolerance, initial_value_file):
    print('-------------------------------main-------------------------------')
    tend = float(sys.argv[2])
    hts = float(sys.argv[3])
    if len(sys.argv) > 4:
        ht0 = hts / int(sys.argv[4])
        ht1 = hts / int(sys.argv[5])
    else:
        # strang splitting with one step for each component
        ht0 = hts / 2 # first and last step in strang splitting
        ht1 = hts / 1 # central step in strang splittin

    #create discretization
    Nx, xs, hxs, Vmhn0 = create_initial_fiber(initial_value_file, 1191, 0, 11.9)

    # system matrices for implicit euler
    @lru_cache(maxsize=8)
    def ie_sys_expl(ht, Nx):
        print(f"> build expl. IE matrix for ht={ht} and Nx={Nx}\033[90m\033[m")
        xs = np.linspace(0,11.9,Nx)
        hxs = xs[1:]-xs[:-1]
        return create_mass(Nx, hxs, inner=True)
    @lru_cache(maxsize=8)
    def ie_sys_impl(ht, Nx):
        print(f"> build impl. IE matrix for ht={ht} and Nx={Nx}\033[90m\033[m")
        xs = np.linspace(0,11.9,Nx)
        hxs = xs[1:]-xs[:-1]
        return create_mass(Nx, hxs, inner=True) + ht * create_stiffness(Nx,hxs, inner=True)

    def rhs_hodgkin_huxley(Vmhn, t):
        return rhs_hh(Vmhn)

    #solve the dual problem
    lhs_dual = ie_sys_impl(hts, Nx).T
    f = 2**(-(np.linspace(-10,10,Nx-2))**2)
    rhs_dual = np.zeros(Nx-2)
    for k in range(0,Nx-3):
        rhs_dual[k] = f[k]*hxs[k]/2 + f[k+1]*hxs[k+1]/2
    zh = sparse.linalg.cg(lhs_dual, rhs_dual, tol=1e-14)
    assert zh[1]==0; zh = zh[0]
    zh = np.insert(zh, 0, zh[0])
    zh = np.insert(zh, -1, zh[-1])

    def error_est(u1, u0):
        u0 = np.insert(u0,0,u0[0])
        u0 = np.insert(u0,-1,u0[-1])
        u1 = np.insert(u1,0,u1[0])
        u1 = np.insert(u1,-1,u1[-1])
        #print(u0)
        #print(u1)
        e_iter = iter_error_est(u0,u1,zh,xs,hxs,hts,prefactor)    #!!!!!!!!!!!!!!!!!!!!!!!!
        e_disc = disc_error_est(u0,u1,zh,xs)
        return[e_disc, e_iter]

    # Solve the equation
    eps = 1e-12
    if tolerance != None:
        eps = tolerance
        error_est = None

    time_discretization=dict(
        t0=0, t1=tend, # start and end time of the simulation
        hts=hts, # time step width of the splitting method
        ht0=ht0, # time step width for the reaction term (Hodgkin-Huxley)
        ht1=ht1, # time step width for the diffusion term
        eps=eps, # stopping criterion for the linear solver in the diffusion step
        maxit=1000, # maximum number of iterations for the implicit solver
    )

    # reuse CN code by replacing the implicit and explicit system matrices
    trajectory, iters, e_ests = strang_1H_1CN_FE(
        Vmhn0,
        rhs_hodgkin_huxley,
        (ie_sys_expl, ie_sys_impl),
        time_discretization['t0'],
        time_discretization['t1'],
        time_discretization['hts'],
        traj=True,
        eps=time_discretization['eps'],
        maxit=time_discretization['maxit'],
        error_est = error_est
    )

    # indices: trajectory[time_index, point alog x-axis, variable]
    out_stride = 1
    np.save("out.npy",trajectory[::out_stride, :, :])

    ######## plot results
    print('Voltage range in the last timestep:')
    print(f"  V: {np.min(trajectory[-1][:,0]):>+.2e} -- {np.max(trajectory[-1][:,0]):>+.2e}")

    time_steps = trajectory.shape[0]
    step_stride = 400
    cs = np.linspace(0,1, time_steps // step_stride + 1)
    fig = plt.figure()

    ax = fig.add_subplot(111)
    # plot the transmembrane voltage
    ax.plot(xs, trajectory[::-step_stride, :, 0].T)
    for idx,line in enumerate(ax.lines): line.set_color((cs[idx], 0.5, 0.5))
    ax.plot(xs, trajectory[0, :, 0], '--', color='black',label='V_h(t=0)')
    ax.plot(xs, trajectory[-1, :, 0], color='black', label='V_h(t=5)')
    plt.xlabel('fiber length [cm]')
    plt.ylabel('transmembrane Potental V_m [mV]')
    plt.legend()
    plt.show()
    ######### plot results 2
    if tolerance == None:
        timesteps = (tend)/hts
        print('timesteps', timesteps)
        print('total # of iterations: ', np.sum(iters))
        print('avg iterations perr timestep: ', np.sum(iters)/timesteps)
        e_discs = e_ests[1]
        fig, axs = plt.subplots(3,1)
        timesteps = np.arange(0,timesteps,1)
        axs[0].plot(timesteps, iters, label='CG Iterations')
        axs[0].set_xlabel('#timesteps')
        axs[0].legend()
        axs[1].plot(timesteps, abs(np.asarray(e_discs)), label='|Discretization error|')
        axs[1].set_xlabel('#timesteps')
        axs[1].set_yscale('log')
        axs[1].legend()
        axs[2].plot(timesteps, e_discs, label='Discretization error')
        axs[2].set_xlabel('#timesteps')
        axs[2].legend()
        plt.show()

def disc_error_convtest(Nx_start, steps):
    print('-------------------------------disc_error_convtest-------------------------------')
    #read initial values and arguments
    initial_value_file = ''
    tend = 5
    hts = float(sys.argv[3])
    if len(sys.argv) > 4:
        ht0 = hts / int(sys.argv[4])
        ht1 = hts / int(sys.argv[5])
    else:
        # strang splitting with one step for each component
        ht0 = hts / 2 # first and last step in strang splitting
        ht1 = hts / 1 # central step in strang splittin
    
    #only if started without inital value file
    if initial_value_file == '':
        #create fiber
        Nx, xs, hxs, Vmhn0 = create_initial_fiber(initial_value_file, Nx_start, 0, 11.9)

        #create system matrices
        @lru_cache(maxsize=8)
        def ie_sys_expl(ht, Nx):
            #print(f"> build expl. IE matrix for ht={ht}, hx(1)={hxs[1]} and Nx={Nx}\033[90m\033[m")
            return create_mass(Nx, hxs, inner=True)
        @lru_cache(maxsize=8)
        def ie_sys_impl(ht, Nx):
            #print(f"> build impl. IE matrix for ht={ht}, hx(1)={hxs[1]} and Nx={Nx}\033[90m\033[m")
            return create_mass(Nx, hxs, inner=True) + ht * create_stiffness(Nx,hxs, inner=True)

        def rhs_hodgkin_huxley(Vmhn, t):
            return rhs_hh(Vmhn)

        # Solve the equation until t = 5 (tstep=4999)
        time_discretization=dict(
            t0=0, t1=5, # start and end time of the simulation
            hts=hts, # time step width of the splitting method
            ht0=ht0, # time step width for the reaction term (Hodgkin-Huxley)
            ht1=ht1, # time step width for the diffusion term
            eps=1e-12, # stopping criterion for the linear solver in the diffusion step
            maxit=1000, # maximum number of iterations for the implicit solver
        )
        trajectory = strang_1H_1CN_FE(
            Vmhn0,
            rhs_hodgkin_huxley,
            (ie_sys_expl, ie_sys_impl),
            time_discretization['t0'],
            time_discretization['t1'],
            time_discretization['hts'],
            traj=True,
            eps=time_discretization['eps'],
            maxit=time_discretization['maxit'],
        )[0]
        Vmhn0_ = trajectory[-1,:,:]
        xs_init = xs

        #compute half a timestep with heun scheme
        Vmhn = heun_step(Vmhn0_, rhs_hodgkin_huxley, 5, hts/2)
        Vmhn0_ = np.array(Vmhn)

        #compute one timestep with by factor finer discretizations and solve the respective dual problem
        V1s = []
        V0s = []
        zhs = []
        xss = []
        hxss = []
        Nxs = []
        max_factor = steps
        for factor in range(0,max_factor):
            #print(f'--factor={factor}--')
            #create fiber
            if factor != 0:
                Nx_fine = numberofnodes(factor, Nx_start)
                Vmhn0_fine = np.zeros((Nx_fine, 4))
                Vmhn0_fine[:,0] = make_fine(Vmhn0_[:,0],xs_init, factor)[0]
                Vmhn0_fine[:,1] = make_fine(Vmhn0_[:,1],xs_init, factor)[0]
                Vmhn0_fine[:,2] = make_fine(Vmhn0_[:,2],xs_init, factor)[0]
                Vmhn0_fine[:,3] = make_fine(Vmhn0_[:,3],xs_init, factor)[0]
            else: 
                Nx_fine = Vmhn0_[:,0].shape[0]
                Vmhn0_fine = Vmhn0_
            Nx, xs, hxs, Vmhn0 = create_initial_fiber(initial_value_file, Nx_fine, 0, 11.9, Vmhn_initial=Vmhn0_fine)
            V0 = Vmhn0[:,0] 

            #solve primal problem: compute one timestep with implicit euler scheme
            lhs = ie_sys_impl(hts,Nx)
            rhs = ie_sys_expl(hts,Nx)
            V1_ = mycg(lhs, rhs,V0[1:-1],1, maxiter=None, tol=1e-14)
            assert V1_[1]==0; V1=V1_[0]
            V1 = np.insert(V1,0,V1[0])
            V1 = np.insert(V1,-1,V1[-1])

            #solve the dual problem
            lhs_dual = lhs.T
            #print('lhs_dual')
            #print('  '+arr2str(lhs_dual.todense(), prefix='  '))
            f = 2**(-(np.linspace(-10,10,Nx-2))**2)
            rhs_dual = np.zeros(Nx-2)
            for k in range(0,Nx-3):
                rhs_dual[k] = f[k]*hxs[k]/2 + f[k+1]*hxs[k+1]/2
            #rhs_dual[0] = hxs[0] + hxs[1]/2
            #rhs_dual[-1] = hxs[-2]/2 + hxs[-1]
            zh = sparse.linalg.cg(lhs_dual, rhs_dual, tol=1e-14)
            assert zh[1]==0; zh = zh[0]
            zh = np.insert(zh, 0, zh[0])
            zh = np.insert(zh, -1, zh[-1])

            #append
            V1s.append(V1)
            V0s.append(V0)
            zhs.append(zh)
            hxss.append(hxs)
            Nxs.append(Nx)
            xss.append(xs)
        V1s = np.asarray(V1s, dtype=object)
        V0s = np.asarray(V0s, dtype=object)
        zhs = np.asarray(zhs, dtype=object)
        hxss = np.asarray(hxss, dtype=object)
        Nxs = np.asarray(Nxs)
        xss = np.asarray(xss, dtype=object)

        #compute error
        V1_exact = V1s[-1]
        e_trues = []
        e_ests = []
        for i in range(0, max_factor):
            e_true = compute_error(V1_exact, make_fine(V1s[i], xss[i], max_factor-1-i)[0], hxss[-1])
            e_est = abs(disc_error_est(V0s[i], V1s[i], zhs[i], xss[i]))
            e_trues.append(e_true)
            e_ests.append(e_est)
        e_trues=np.asarray(e_trues)
        e_ests=np.asarray(e_ests)
        
        #plot results
        print('true error list:',e_trues)
        print('est error list:',e_ests)
        plt.plot(Nxs, e_trues,'k--', label='true disc error')
        plt.plot(Nxs, e_ests,'orange', label='disc error (estimate)')
        plt.scatter(Nxs, e_trues,color='black')
        plt.scatter(Nxs, e_ests, color='orange')
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.xlabel('#nodes')
        plt.grid()
        plt.legend()
        plt.show()

def iter_error_convtest(tend):
    print('-------------------------------iter_error_convtest-------------------------------')
    initial_value_file = sys.argv[1]

    hts = float(sys.argv[3])
    if len(sys.argv) > 4:
        ht0 = hts / int(sys.argv[4])
        ht1 = hts / int(sys.argv[5])
    else:
        # strang splitting with one step for each component
        ht0 = hts / 2 # first and last step in strang splitting
        ht1 = hts / 1 # central step in strang splittig
    
    #create discretization
    Nx, xs, hxs, Vmhn0 = create_initial_fiber(initial_value_file, 1191, 0, 11.9)

    # system matrices for implicit euler
    @lru_cache(maxsize=8)
    def ie_sys_expl(ht, Nx):
        print(f"> build expl. IE matrix for ht={ht} and Nx={Nx}\033[90m\033[m")
        xs = np.linspace(0,11.9,Nx)
        hxs = xs[1:]-xs[:-1]
        return create_mass(Nx, hxs, inner=True)
    @lru_cache(maxsize=8)
    def ie_sys_impl(ht, Nx):
        print(f"> build impl. IE matrix for ht={ht} and Nx={Nx}\033[90m\033[m")
        xs = np.linspace(0,11.9,Nx)
        hxs = xs[1:]-xs[:-1]
        return create_mass(Nx, hxs, inner=True) + ht * create_stiffness(Nx,hxs, inner=True)

    def rhs_hodgkin_huxley(Vmhn, t):
        return rhs_hh(Vmhn)

    #solve the dual problem
    lhs_dual = ie_sys_impl(hts, Nx).T

    f = 2**(-(np.linspace(-10,10,Nx-2))**2)
    rhs_dual = np.zeros(Nx-2)
    for k in range(0,Nx-3):
        rhs_dual[k] = f[k]*hxs[k]/2 + f[k+1]*hxs[k+1]/2
    zh = sparse.linalg.cg(lhs_dual, rhs_dual, tol=1e-14)
    assert zh[1]==0; zh = zh[0]
    zh = np.insert(zh, 0, zh[0])
    zh = np.insert(zh, -1, zh[-1])

    # Solve the equation until t = 5 (tstep=4999)
    time_discretization=dict(
        t0=0, t1=tend, # start and end time of the simulation
        hts=hts, # time step width of the splitting method
        ht0=ht0, # time step width for the reaction term (Hodgkin-Huxley)
        ht1=ht1, # time step width for the diffusion term
        eps=1e-12, # stopping criterion for the linear solver in the diffusion step
        maxit=1000, # maximum number of iterations for the implicit solver
    )
    trajectory = strang_1H_1CN_FE(
        Vmhn0,
        rhs_hodgkin_huxley,
        (ie_sys_expl, ie_sys_impl),
        time_discretization['t0'],
        time_discretization['t1'],
        time_discretization['hts'],
        traj=True,
        eps=time_discretization['eps'],
        maxit=time_discretization['maxit'],
    )[0]
    Vmhn = trajectory[-1,:,:]
    V_5000 = trajectory[-1, :, 0]
    V_5000_1 = trajectory[-1, :, 1]
    V_5000_2 = trajectory[-1, :, 2]
    V_5000_3 = trajectory[-1, :, 3]
    xs_init = xs

    #compute half a timestep with heun scheme
    Vmhn = heun_step(Vmhn, rhs_hodgkin_huxley, tend, hts/2)
    Vmhn0 = np.array(Vmhn)
    V0 = Vmhn0[:,0]
    print(np.linalg.norm(V0))

    #exact solution of the discretized primal problem (tol=1e-14)
    lhs = ie_sys_impl(hts,Nx)
    rhs = ie_sys_expl(hts,Nx)
    V1_exact = sparse.linalg.cg(lhs, rhs*V0[1:-1],maxiter=None, tol=1e-14)
    assert V1_exact[1]==0; V1_exact=V1_exact[0]
    V1_exact = np.insert(V1_exact, 0, V1_exact[0])
    V1_exact = np.insert(V1_exact, -1, V1_exact[-1])

    #exact solution of the primal problem
    Nx_fine = numberofnodes(3,1191)
    xs_fine = np.linspace(0,11.9,Nx_fine)
    hxs_fine = xs_fine[1:] - xs_fine[:-1]
    lhs_fine = ie_sys_impl(hts, Nx_fine)
    rhs_fine = ie_sys_expl(hts, Nx_fine)
    V0_fine = make_fine(V0, xs, 3)[0]
    V1_exact_disc = sparse.linalg.cg(lhs_fine, rhs_fine*V0_fine[1:-1], maxiter=None, tol=1e-14)
    assert V1_exact_disc[1]==0; V1_exact_disc = V1_exact_disc[0]
    V1_exact_disc = np.insert(V1_exact_disc, 0, V1_exact_disc[0])
    V1_exact_disc = np.insert(V1_exact_disc, -1, V1_exact_disc[-1])

    maxit = 1
    e_trues = []
    e_ests = []
    e_disc_ests = []
    e_disc_trues = []
    iters = []
    while maxit <= 20:
        #solve primal problem for maxiter=maxit
        lhs = ie_sys_impl(hts,Nx)
        rhs = ie_sys_expl(hts,Nx)
        V1 = mycg(lhs, rhs,V0[1:-1],1,maxiter = maxit, tol=0)
        V1 = V1[0]
        V1 = np.insert(V1, 0, V1[0])
        V1 = np.insert(V1, -1, V1[-1])

        #compute error
        e_true = abs(compute_error(V1_exact, V1, hxs))
        e_est = abs(iter_error_est(V0, V1, zh, xs, hxs, hts, prefactor))
        e_disc_est = abs(disc_error_est(V0, V1, zh, xs))
        e_disc_true = compute_error(V1_exact_disc, make_fine(V1_exact,xs,3)[0], hxs_fine)

        #append
        e_trues.append(e_true)
        e_ests.append(e_est)
        e_disc_ests.append(e_disc_est)
        e_disc_trues.append(e_disc_true)
        iters.append(maxit)
        maxit += 1
    
    #plot results
    print('true errors:', e_trues)
    print('')
    print('est errors:', e_ests)
    print('')
    print('est disc errors: ', e_disc_ests)
    print('')
    print('true disc errors: ', e_disc_trues)
    plt.plot(np.asarray(iters), np.asarray(e_ests),'magenta', label='iter error (estimate)')
    plt.plot(np.asarray(iters), np.asarray(e_trues),'k--', label='true iteration error')
    plt.plot(np.asarray(iters), np.asarray(e_disc_ests), 'orange', label='disc error (estimate)')
    plt.plot(np.asarray(iters), np.asarray(e_disc_trues), color='gray', linestyle='dashed', label='true disc error')
    plt.title(f't = {tend}')
    plt.yscale('log', base=10)
    plt.xlabel('#iterations')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #without error estimation
    main(tolerance=1e-12, initial_value_file = sys.argv[1])
    #test setup for the discretization error
    disc_error_convtest(Nx_start=100, steps=8)
    #test setup for the discretization error
    iter_error_convtest(tend=5)
    #with error estimation for stopping criterion
    main(tolerance=None, initial_value_file = sys.argv[1])
    #with error estimation for stopping criterion (new initial values)
    main(tolerance=None, initial_value_file = '')