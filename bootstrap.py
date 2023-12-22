import numpy as np
from numba import njit
from .vide import vide_startR, vide_start, vie_startR, vie_start
from .contour_funcs import matrix_matrix, matrix_matrix2, matrix_tensor, tensor_tensor, copy_gmatrix
from .convolution import conv_lmx_res, conv_les_res
from .utils import gdist_real_time



@njit
def boot_gret(G, S, H, mu, interpol):
    norb = G.get_ret().shape[-1]
    Gaux = G.get_ret_comp()[:interpol.k+1,:interpol.k+1]
    Hhf = H[:interpol.k+1] + S.get_hf()[:interpol.k+1] - mu*np.array([np.eye(norb),]*(interpol.k+1))
    for tt in range(interpol.k+1):
        for oo in range(norb):
            Gaux[tt,tt,oo,oo] = -1j
    
    for tt in range(interpol.k+1):
        Gaux[tt+1:,tt] = vide_startR(interpol, 1j*Hhf, np.zeros_like(Hhf), 1j*S.get_ret(), Gaux[:tt+1,tt])
        Gaux[tt,tt+1:] = -np.swapaxes(Gaux[tt+1:,tt].conjugate(), -1, -2)
        for ss in range(tt, interpol.k+1):
            G.set_ret_loc(ss, tt, Gaux[ss,tt])


@njit
def boot_glmx(G, S, Q, H, mu, interpol):
    ntau = G.get_mat().shape[0]
    Hhf = H[:interpol.k+1] + S.get_hf()[:interpol.k+1] - mu*np.array([np.eye(H.shape[-1]),]*(interpol.k+1))
    for tt in range(ntau):
        bootGI = vide_start(interpol, 1j*Hhf, -1j*Q.get_lmx()[:,tt], 1j*S.get_ret(), G.get_lmx()[0,tt])
        for ss in range(1,interpol.k+1):
            G.set_lmx_loc(ss, tt, bootGI[ss-1])


@njit
def boot_gles(G, S, Q, H, mu, interpol):
    Hhf = H[:interpol.k+1] + S.get_hf()[:interpol.k+1] - mu*np.array([np.eye(H.shape[-1]),]*(interpol.k+1))
    for tt in range(interpol.k+1):
        bootGL = vide_start(interpol, 1j*Hhf, -1j*Q.get_les()[:,tt], 1j*S.get_ret(), G.get_les()[0,tt])
        for ss in range(1, interpol.k+1):
            G.set_les_loc(ss, tt, bootGL[ss-1])



@njit
def boot_wret(W, vP, E, interpol):
    norb = W.get_ret().shape[-1]
    Waux = np.zeros((interpol.k+1,interpol.k+1,norb,norb,norb,norb))
    for tt in range(interpol.k+1):
        Waux[tt,tt] = E.get_ret()[tt,tt]
        Waux[tt+1:,tt] = vie_startR(interpol, E.get_ret()[:,tt], -vP.get_ret(), Waux[:tt+1,tt])
        Waux[tt,tt+1:] = -np.swapaxes(np.swapaxes(Waux[tt+1:,tt].conjugate(), -2, -3), -1, -4)
        for ss in range(tt, interpol.k+1):
            W.set_ret_loc(ss, tt, Waux[ss,tt])


@njit
def boot_wlmx(W, vP, E, interpol):
    ntau = W.get_mat().shape[0]
    for tt in range(ntau):
        bootWI = vie_start(interpol, E.get_lmx()[:,tt], -vP.get_ret(), W.get_lmx[0,tt])
        for ss in range(1,interpol.k+1):
            W.set_lmx_loc(ss, tt, bootWI[ss-1,tt])


@njit
def boot_wles(W, vP, E, interpol):
    for tt in range(interpol.k+1):
        bootWL = vie_start(interpol, E.get_les()[:,tt], -vP.get_ret(), W.get_les[0,tt])
        for ss in range(1,interpol.k+1):
            W.set_les_loc(ss, tt, bootWL[ss-1,tt])




@njit
def boot_loop_gw(G, S, vP, W, Q, E, H, v, mu, interpol, tol=1e-6):
    ntau, norb = G.get_mat().shape[:-1]
    particle_sign = (-1)**G.particle_type
    
    #G initial conditions
    for tt in range(interpol.k+1):
        G.set_ret_loc(tt, tt, -1j*np.eye(norb))
    for rr in range(ntau):
        G.set_lmx_loc(0, rr, 1j * particle_sign * G.get_mat()[-1-rr])
    
    conv = 1e5
    while conv>tol:
        # Update bubbles
        for tt in range(interpol.k+1):
            for ss in range(interpol.k+1):
                vP.set_ret_loc(tt, ss, -1j*tensor_tensor(v, matrix_matrix2(G.get_ret()[tt,ss], G.get_les()[ss,tt]) +
                                                            matrix_matrix2(G.get_les()[tt,ss], G.get_adv()[ss,tt])))
                vP.set_les_loc(tt, ss, -1j*tensor_tensor(v, matrix_matrix2(G.get_les()[tt,ss], G.get_gtr()[ss,tt])))
            for rr in range(ntau):
                vP.set_lmx_loc(tt, rr, -1j*tensor_tensor(v, matrix_matrix2(G.get_lmx()[tt,rr], G.get_rmx()[rr,tt])))
        
        # Boot W
        for tt in range(interpol.k+1):
            conv_lmx_res(tt, vP, W, E, tensor_tensor)
            for rr in range(ntau):
                E.add_to_lmx_loc(tensor_tensor(v, vP.get_lmx()[tt,rr]))
                if tt==0:
                    W.set_lmx_loc(0, rr, E.get_lmx()[0,rr])
            for ss in range(interpol.k+1):
                E.set_ret_loc(tt, ss, tensor_tensor(vP.get_ret()[tt,ss], v))
                if ss==tt:
                    W.set_ret_loc(ss, ss, E.get_ret()[ss,ss])
                conv_les_res(tt, ss, vP, W, E, interpol, tensor_tensor)
                E.add_to_les_loc(tensor_tensor(v, vP.get_les()[tt,ss]))
                if tt==0:
                    W.set_les_loc(0, ss, E.get_les()[0, ss])
        boot_wret(W, vP, E, interpol)
        boot_wlmx(W, vP, E, interpol)
        boot_wles(W, vP, E, interpol)
        
        # Update S
        for tt in range(interpol.k+1):
            S.set_hf_loc(tt, matrix_tensor(G.get_les()[tt,tt], v - np.swapaxes(v, -1, -2)))
            for ss in range(interpol.k+1):
                S.set_ret_loc(tt, ss, 1j*matrix_tensor(G.get_ret()[tt,ss], W.get_gtr()[tt,ss]) +
                                      1j*matrix_tensor(G.get_les()[tt,ss], W.get_ret()[tt,ss]))
                S.set_les_loc(tt, ss, 1j*matrix_tensor(G.get_les()[tt,ss], W.get_les()[tt,ss]))
            for rr in range(ntau):
                S.set_lmx_loc(tt, rr, 1j*matrix_tensor(G.get_lmx()[tt,rr], W.get_lmx()[tt,rr]))
        
        # Boot G
        lastG = copy_gmatrix(G)
        for tt in range(interpol.k+1):
            conv_lmx_res(tt, S, G, Q, interpol, matrix_matrix)
            for ss in range(interpol.k+1):
                conv_les_res(tt, ss, S, G, Q, interpol, matrix_matrix)
        boot_gret(G, S, H, mu, interpol)
        boot_glmx(G, S, Q, H, mu, interpol)
        for tt in range(interpol.k+1):
            G.set_les_loc(0, tt, G.get_rmx()[0,tt])
        boot_gles(G, S, Q, H, mu, interpol)
        
        # Convergence
        conv = gdist_real_time(interpol.k, lastG, G)