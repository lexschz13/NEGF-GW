import numpy as np
from numba import njit
from .vide import vide_step, vie_step
from .contour_funcs import matrix_matrix, matrix_matrix2, matrix_tensor, tensor_tensor, copy_gmatrix
from .convolution import conv_lmx_res, conv_les_res
from .utils import gdist_real_time
from .printing import float_string, exp_string



@njit
def step_gret(t, G, S, Q, H, mu, interpol):
    norb = G.get_mat()[-1]
    Gaux = G.get_ret_comp()[:t+1,:t+1]
    Hhf = H[:t+1] + S.get_hf()[:t+1] - mu * np.array([np.eye(norb),]*(t+1))
    for jj in range(t):
        G.set_ret_loc(t, jj, vide_step(interpol, 1j*Hhf, -1j*Q.get_ret()[:,jj], 1j*S.get_ret(), Gaux[:t,jj]))
    G.set_ret_loc(t, t, -1j*np.eye(norb))


@njit
def step_glmx(t, G, S, Q, H, mu, interpol):
    ntau, norb = G.get_mat()[:-1]
    Hhf = H[:t+1] + S.get_hf()[:t+1] - mu * np.array([np.eye(norb),]*(t+1))
    for jj in range(ntau):
        G.set_lmx_loc(t, jj, vide_step(interpol, 1j*Hhf, -1j*Q.get_lmx[:,jj], 1j*S.get_ret(), G.get_lmx()[:t,jj]))


@njit
def step_gles(t, G, S, Q, H, mu, interpol):
    norb = G.get_mat()[-1]
    Hhf = H[:t+1] + S.get_hf()[:t+1] - mu * np.array([np.eye(norb),]*(t+1))
    for jj in range(t+1):
        G.set_les_loc(t, jj, vide_step(interpol, 1j*Hhf, -1j*Q.get_les()[:,jj], 1j*S.get_ret(), G.get_les()[:t,jj]))
        if jj < t:
            G.set_les_loc(jj, t, -G.get_les()[t, jj].T.conjugate())


@njit
def step_wret(t, W, vP, E, interpol):
    Waux = W.get_ret_comp()[:t+1,:t+1]
    for jj in range(t):
        W.set_ret_loc(t, jj, vie_step(interpol, E.get_ret()[:,jj], -vP.get_ret(), Waux[:t,jj]))
    W.set_ret_loc(t, t, E.get_ret()[t,t])


@njit
def step_wlmx(t, W, vP, E, interpol):
    ntau = W.get_mat().shape[0]
    for jj in range(ntau):
        W.set_lmx_loc(t, jj, vide_step(interpol, E.get_lmx[:,jj], -vP.get_ret(), W.get_lmx()[:t,jj]))


@njit
def step_wles(t, W, vP, E, interpol):
    for jj in range(t+1):
        W.set_les_loc(t, jj, vie_step(interpol, E.get_les()[:,jj], -vP.get_ret(), W.get_les()[:t,jj]))
        if jj < t:
            W.set_les_loc(jj, t, -np.swapaxes(np.swapaxes(W.get_les()[t,jj].conjugate(), 0, 3), 1, 2))



@njit
def step_loop_gw(t, G, S, vP, W, Q, E, H, v, mu, interpol, h, tol=1e-6, max_iter=100000):
    print("Starting convergence loop for time step "+str(t))
    ntau, norb = G.get_mat().shape[:-1]
    
    #G initial conditions
    G.set_ret_loc(t, t, -1j*np.eye(norb))
    #Init for GI already set
    
    conv = 1e5
    loop_iterations = 0
    while conv > tol:
        #Update bubbles
        for ss in range(t+1):
            vP.set_ret_loc(t, ss, -1j*tensor_tensor(v, matrix_matrix2(G.get_ret()[t,ss], G.get_les()[ss,t]) +
                                                       matrix_matrix2(G.get_les()[t,ss], G.get_adv()[ss,t])))
            vP.set_les_loc(t, ss, -1j*tensor_tensor(v, matrix_matrix2(G.get_les()[t,ss], G.get_gtr()[ss,t])))
            if ss < t:
                vP.set_les_loc(ss, t, -np.swapaxes(np.swapaxes(W.get_les()[t,ss].conjugate(), 0, 3), 1, 2))
        for kk in range(ntau):
            vP.set_lmx(t, kk, -1j*tensor_tensor(v, matrix_matrix2(G.get_lmx()[t,kk], G.get_rmx()[kk,t])))
        
        #Step W
        conv_lmx_res(t, vP, W, E, interpol, h, tensor_tensor)
        for kk in range(ntau):
            E.add_to_lmx_loc(t, kk, tensor_tensor(vP.get_lmx()[t,kk], v))
            #Init for WI already set
        for ss in range(t+1):
            E.set_ret_loc(t, ss, tensor_tensor(vP.get_ret()[t, ss], v))
            if ss==t:
                W.set_ret_loc(t, t, E.get_ret()[t,t])
            conv_les_res(t, ss, vP, W, E, interpol, h, tensor_tensor)
            E.add_to_les_loc(t, ss, tensor_tensor(vP.get_les()[t,ss], v))
            #Init for WL not necessary because of c.h.
            if ss < t:
                E.set_les_loc(ss, t, -np.swapaxes(np.swapaxes(E.get_les()[t,ss].conjugate(), 0, 3), 1, 2))
        step_wret(t, W, vP, E, interpol)
        step_wlmx(t, W, vP, E, interpol)
        step_wles(t, W, vP, E, interpol)
        
        #Update S
        S.set_hf_loc(t, matrix_tensor(G.get_les()[t,t], v - np.swapaxes(v, -1, -2)))
        for ss in range(interpol.k+1):
            S.set_ret_loc(t, ss, 1j*matrix_tensor(G.get_ret()[t,ss], W.get_gtr()[t,ss]) +
                                 1j*matrix_tensor(G.get_les()[t,ss], W.get_ret()[t,ss]))
            S.set_les_loc(t, ss, 1j*matrix_tensor(G.get_les()[t,ss], W.get_les()[t,ss]))
            if ss < t:
                S.set_les_loc(ss, t, -S.get_les()[t,ss].T.conjugate())
        for kk in range(ntau):
            S.set_lmx_loc(t, kk, 1j*matrix_tensor(G.get_lmx()[t,kk], W.get_lmx()[t,kk]))
        
        #Step G
        lastG = copy_gmatrix(G)
        conv_lmx_res(t, S, G, Q, interpol, h, matrix_matrix)
        for ss in range(interpol.k+1):
            conv_les_res(t, ss, S, G, Q, interpol, h, matrix_matrix)
        step_gret(G, S, Q, H, mu, interpol)
        step_glmx(G, S, Q, H, mu, interpol)
        #Init for GL not necessary because of c.h.
        step_gles(G, S, Q, H, mu, interpol)
        
        # Convergence
        conv = gdist_real_time(t, lastG, G)
        loop_iterations += 1
        
        # Reched max iterations
        assert loop_iterations < max_iter
    
    print("Convergence for time step "+str(t)+" reached")
    print("Norm "+exp_string(conv, 5)+"\n\n")


@njit
def step_loop_gw0(t, G, S, Q, H, v, mu, interpol, h, tol=1e-6, max_iter=100000):
    print("Starting convergence loop for time step "+str(t))
    ntau, norb = G.get_mat().shape[:-1]
    
    #G initial conditions
    G.set_ret_loc(t, t, -1j*np.eye(norb))
    #Init for GI already set
    
    conv = 1e5
    loop_iterations = 0
    while conv > tol:
        
        #Update S
        S.set_hf_loc(t, matrix_tensor(G.get_les()[t,t], v - np.swapaxes(v, -1, -2)))
        for ss in range(interpol.k+1):
            S.set_ret_loc(t, ss, matrix_tensor(G.get_ret()[t,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_gtr()[t,ss], G.get_les()[ss,t]), v))) +
                                  matrix_tensor(G.get_les()[t,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_ret()[t,ss], G.get_les()[ss,t]), v))) +
                                  matrix_tensor(G.get_les()[t,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_les()[t,ss], G.get_adv()[ss,t]), v))) )
            S.set_les_loc(t, ss, matrix_tensor(G.get_les()[t,ss], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_les()[t,ss], G.get_gtr()[ss,t]), v))) )
            if ss < t:
                S.set_les_loc(ss, t, -S.get_les()[t,ss].T.conjugate())
        for kk in range(ntau):
            S.set_lmx_loc(t, kk, matrix_tensor(G.get_lmx()[t,kk], tensor_tensor(v, tensor_tensor(matrix_matrix2(G.get_lmx()[t,kk], G.get_rmx()[kk,t]), v))))
        
        #Step G
        lastG = copy_gmatrix(G)
        conv_lmx_res(t, S, G, Q, interpol, h, matrix_matrix)
        for ss in range(interpol.k+1):
            conv_les_res(t, ss, S, G, Q, interpol, h, matrix_matrix)
        step_gret(G, S, Q, H, mu, interpol)
        step_glmx(G, S, Q, H, mu, interpol)
        #Init for GL not necessary because of c.h.
        step_gles(G, S, Q, H, mu, interpol)
        
        # Convergence
        conv = gdist_real_time(t, lastG, G)
        loop_iterations += 1
        
        # Reched max iterations
        assert loop_iterations < max_iter
    
    print("Convergence for time step "+str(t)+" reached")
    print("Norm "+exp_string(conv, 5)+"\n\n")