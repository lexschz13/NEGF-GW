import numpy as np
from numba import njit
from .linalg import eig
from .contour_funcs import Gmatrix, Vmatrix, matrix_matrix, matrix_matrix2, matrix_tensor, tensor_tensor
from .convolution import conv_mat_extern
from .utils import gdist_mat
from .printing import float_string, exp_string


@njit
def sgn(x):
    if x < 0:
        return -1
    elif x > 0:
        return +1
    else:
        return 0


@njit
def g_nonint_init(ntau, N, mu, H0, beta=1, particle=0, mu_jump=0.5, tolN=1e-6):
    assert H0.ndim==2, "Only constant hamiltonian"
    assert H0.shape[0]==H0.shape[1], "Hamiltonian is a squared matrix"
    n_orb = H0.shape[0]
    particle_sign = (-1)**particle
    tau = np.linspace(0, beta, ntau)
    print("Aceeded to g init")
    
    assert not np.any(np.isnan(H0))
    
    w, P = eig(H0)
    print("Eigen H0")
    
    last_sign = 2
    if N>0:
        print("Checking number of particle for non-interacting case")
        while True:
            N0 = 0.0
            for jj in range(n_orb):
                e = w[jj].real - mu
                N0 += 1/(particle_sign - np.exp(e * beta))
            
            DN = N - N0
            DNsign = sgn(DN)
            if abs(DN) < tolN:
                break
            if DNsign!=last_sign and last_sign!=2:
                mu_jump /= 2
            mu += DNsign * mu_jump
            last_sign = DNsign
    
    g = np.zeros((ntau, H0.shape[0], H0.shape[1]), dtype=np.complex128)
    print("g allocated")
    for ii in range(n_orb):
        for jj in range(n_orb):
            for kk in range(n_orb):
                e = w[jj] - mu
                g[...,ii,kk] += P[ii,jj] * np.exp(-e*tau)/(particle_sign * np.exp(e*beta) - 1) * P[kk,jj].conjugate()
    
    print("g initialized")
    return g, mu


@njit
def matsubara_branch_init(N, mu, H0, G, S, vP, W, E, v, interpol, beta=1, particle=0, mu_jump=0.5, max_iter=100000, tol=1e-6):
    print("Initilizing Matsubara branch")
    ntau = G.get_mat().shape[0]
    particle_sign = (-1)**particle
    
    ntau = G.get_mat().shape[0]
    htau = beta / (ntau-1)
    
    last_sign = 2
    while True:
        conv = 1e5
        loop_iterations = 0
        print("Starting Matsubara loop for mu="+float_string(mu, 5))
        while conv>=tol:
            for tt in range(ntau):
                vP.set_mat_loc(tt, -tensor_tensor(v, matrix_matrix2(G.get_mat()[tt], G.neg_imag_time_mat()[tt])))
                E.set_mat_loc(tt, tensor_tensor(vP[tt], v))
            W.set_mat(E.get_mat() + conv_mat_extern(vP.get_mat(), W.get_mat(), interpol, htau, tensor_tensor, 0))
            
            S.set_hf_loc(0, -particle_sign * matrix_tensor(G.get_mat()[-1], v-np.swapaxes(v, -1, -2)))
            for tt in range(ntau):
                S.set_mat_loc(tt, matrix_tensor(G.get_mat()[tt], W.get_mat()[tt]))
            
            g,_ = g_nonint_init(ntau, -1, mu, H0 - mu*np.eye(H0.shape[0]) + S.get_hf()[0], beta, particle)
            F = conv_mat_extern(g, S.get_mat(), interpol, htau, matrix_matrix, particle)
            newGM = g + conv_mat_extern(F, G.get_mat(), interpol, htau, matrix_matrix, particle)
            
            conv = gdist_mat(newGM, G.get_mat())
            G.set_mat(newGM)
            
            loop_iterations += 1
            assert loop_iterations <= max_iter
        
        print("Convergence for Matsubara branch and mu="+float_string(mu, 5))
        print("Norm "+exp_string(conv, 5)+"\n\n")
        
        N0 = -np.trace(G.get_mat()[-1])
        DN = N - N0.real
        DNsign = sgn(DN)
        if abs(DN) < tol:
            print("Reached convergence for number of particles "+float_string(N0.real, 5))
            print("----------------------------------------")
            break
        if DNsign!=last_sign and last_sign!=2:
            mu_jump /= 2
        mu += DNsign * mu_jump
        last_sign = DNsign
        
        print("Computed number of particles "+float_string(N0.real, 5))
        print("New mu="+float_string(mu, 5))
    
    return mu


@njit
def matsubara_branch_init_gw0(N, mu, H0, G, S, v, interpol, beta=1, particle=0, mu_jump=0.5, max_iter=100000, tol=1e-6):
    print("Initilizing Matsubara branch")
    ntau = G.get_mat().shape[0]
    particle_sign = (-1)**particle
    
    ntau = G.get_mat().shape[0]
    htau = beta / (ntau-1)
    
    last_sign = 2
    while True:
        conv = 1e5
        loop_iterations = 0
        print("Starting Matsubara loop for mu="+float_string(mu, 5))
        while conv>=tol:
            
            S.set_hf_loc(0, -particle_sign * matrix_tensor(G.get_mat()[-1], v-np.swapaxes(v, -1, -2)))
            print("Shf(t=0) computed")
            print(S.get_hf()[0])
            print("\n")
            assert not np.any(np.isnan(v))
            assert not np.any(np.isnan(S.get_hf()[0]))
            for tt in range(ntau):
                S.set_mat_loc(tt, -matrix_tensor(G.get_mat()[tt], tensor_tensor(tensor_tensor(v, matrix_matrix2(G.get_mat()[tt], G.neg_imag_time_mat()[tt])), v)))
            print("SM computed")
            
            g,no_use = g_nonint_init(ntau, -1, mu, H0 - mu*np.eye(H0.shape[0]) + S.get_hf()[0], beta, particle)
            print("Non-interactive g computed")
            F = conv_mat_extern(g, S.get_mat(), interpol, htau, matrix_matrix, particle)
            print("G and self-energy convoluted")
            newGM = g + conv_mat_extern(F, G.get_mat(), interpol, htau, matrix_matrix, particle)
            print("New GM convoluted")
            
            conv = gdist_mat(newGM, G.get_mat())
            print("Distance computed")
            G.set_mat(newGM)
            print("New GM set")
            
            loop_iterations += 1
            assert loop_iterations <= max_iter
        
        print("Convergence for Matsubara branch and mu="+float_string(mu, 5))
        print("Norm "+exp_string(conv, 5)+"\n\n")
        
        N0 = -np.trace(G.get_mat()[-1])
        DN = N - N0.real
        DNsign = sgn(DN)
        if abs(DN) < tol:
            print("Reached convergence for number of particles "+float_string(N0.real, 5))
            print("----------------------------------------")
            break
        if DNsign!=last_sign and last_sign!=2:
            mu_jump /= 2
        mu += DNsign * mu_jump
        last_sign = DNsign
        
        print("Computed number of particles "+float_string(N0.real, 5))
        print("New mu="+float_string(mu, 5))
    
    return mu