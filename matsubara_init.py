import numpy as np
from numba import njit
from .linalg import eig
from .contour_funcs import Gmatrix, Vmatrix, matrix_matrix, matrix_matrix2, matrix_tensor, tensor_tensor
from .convolution import conv_mat_extern
from .utils import gdist_mat


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
    assert H0.shape[1]==H0.shape[1], "Hamiltonian is a squared matrix"
    n_orb = H0.shape[0]
    particle_sign = (-1)**particle
    tau = np.linspace(0, beta, ntau)
    
    w, P = eig(H0)
    
    last_sign = 2
    if N>0:
        while True:
            N0 = 0
            for jj in n_orb:
                e = w[jj] - mu
                N0 += np.exp(-e*tau)/(particle_sign * np.exp(e*beta) - 1)
            
            DN = N - N0
            DNsign = sgn(DN)
            if abs(DN) < tolN:
                break
            if DNsign!=last_sign and last_sign!=2:
                mu_jump /= 2
            mu += DNsign * mu_jump
            last_sign = DNsign
    
    g = np.zeros((ntau, H0.shape[0], H0.shape[1]), dtype=np.complex128)
    for ii in range(n_orb):
        for jj in range(n_orb):
            for kk in range(n_orb):
                e = w[jj] - mu
                g[...,ii,kk] += P[ii,jj] * np.exp(-e*tau)/(particle_sign * np.exp(e*beta) - 1) * P[kk,jj].conjugate()
    
    return g, mu


@njit
def matsubara_branch_init(N, mu, H0, G, S, vP, W, v, interpol, beta=1, particle=0, mu_jump=0.5, max_iter=1000000, tol=1e-6):
    ntau = G.get_mat().shape[0]
    particle_sign = (-1)**particle
    
    last_sign = 2
    while True:
        conv = 1e5
        iterations = 0
        while conv>=tol:
            vPv = np.zeros_like(vP.get_mat())
            for tt in range(ntau):
                vP.set_mat_loc(tt, tensor_tensor(matrix_matrix2(G.get_mat()[tt], G.neg_imag_time_mat()[tt]), v))
                vPv[tt] = tensor_tensor(vP[tt], v)
            W.set_mat(vPv + conv_mat_extern(vP.get_mat(), W.get_mat(), interpol, tensor_tensor, 0))
            
            S.set_hf_loc(0, -particle_sign * matrix_tensor(G.get_mat()[-1], v-np.swapaxes(v, -1, -2)))
            for tt in range(ntau):
                S.set_mat_loc(tt, matrix_tensor(G.get_mat()[tt], W.get_mat()[tt]))
            
            g,_ = g_nonint_init(ntau, -1, mu, H0 - mu*np.eye(H0.shape[0]) + S.get_hf()[0], beta, particle)
            F = conv_mat_extern(g, S.get_mat(), interpol, matrix_matrix, particle)
            newGM = g + conv_mat_extern(F, G.get_mat(), interpol, matrix_matrix, particle)
            
            conv = gdist_mat(newGM, G.get_mat())
            G.set_mat(newGM)
            
            iterations += 1
            assert iterations <= max_iter
        
        N0 = -np.trace(G.get_mat()[-1])
        DN = N - N0
        DNsign = sgn(DN)
        if abs(DN) < tol:
            break
        if DNsign!=last_sign and last_sign!=2:
            mu_jump /= 2
        mu += DNsign * mu_jump
        last_sign = DNsign