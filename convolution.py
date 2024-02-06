import numpy as np
from numba import njit
from .utils import xor_particle



@njit
def conv_mat_extern(a, b, interpol, h, mat_prod, particle=0):
    assert a.shape[0] == b.shape[0], "Arrays must be same imaginary time steps"
    
    N = a.shape[0]-1
    c1 = np.zeros_like(a)
    c2 = np.zeros_like(a)
    w = interpol.gregory_weights(N)
    
    for mm in range(N+1):
        if mm <= interpol.k:
            for jj in range(interpol.k+1):
                for ll in range(interpol.k+1):
                    c1[mm] += interpol.R[mm,jj,ll] * mat_prod(a[jj], b[ll])
        else:
            for ll in range(mm+1):
                c1[mm] += w[mm,ll] * mat_prod(a[mm-ll], b[ll])
        
        if mm >= N-interpol.k:
            for jj in range(interpol.k+1):
                for kk in range(interpol.k+1):
                    c2[mm] += (-1)**particle * interpol.R[N-mm,jj,kk] * mat_prod(a[N-jj], b[N-kk])
        else:
            for kk in range(N-mm+1):
                c2[mm] += (-1)**particle * w[N-mm,kk] * mat_prod(a[mm+kk], b[N-kk])
    
    return h * (c1 + c2)


@njit
def conv_mat(a, b, c, interpol, h, mat_prod, particle=0):
    assert a.get_mat().shape[0] == b.get_mat().shape[0], "Arrays must be same imaginary time steps"
    
    particle = xor_particle(a.particle_type, b.particle_type)
    assert particle == c.particle_type
    
    N = a.get_mat().shape[0]-1
    w = interpol.gregory_weights(N)
    
    for mm in range(N+1):
        c1 = np.zeros_like(a.get_mat()[0])
        c2 = np.zeros_like(c1)
        if mm <= interpol.k:
            for jj in range(interpol.k+1):
                for ll in range(interpol.k+1):
                    c1 += interpol.R[mm,jj,ll] * mat_prod(a.get_mat()[jj], b.get_mat()[ll])
        else:
            for ll in range(mm+1):
                c1 += w[mm,ll] * mat_prod(a.get_mat()[mm-ll], b.get_mat()[ll])
        
        if mm >= N-interpol.k:
            for jj in range(interpol.k+1):
                for kk in range(interpol.k+1):
                    c2 += (-1)**particle * interpol.R[N-mm,jj,kk] * mat_prod(a.get_mat()[N-jj], b.get_mat()[N-kk])
        else:
            for kk in range(N-mm+1):
                c2 += (-1)**particle * w[N-mm,kk] * mat_prod(a.get_mat()[mm+kk], b.get_mat()[N-kk])
                
        c.set_mat_loc(mm, h * (c1 + c2))


@njit
def conv_lmx_res(t, a, b, c, interpol, h, mat_prod):
    assert a.get_lmx().shape[1] == b.get_mat().shape[0], "Arrays must be same imaginary time steps"
    
    particle = xor_particle(a.particle_type, b.particle_type)
    assert particle == c.particle_type
    
    N = b.get_mat().shape[0]-1
    w = interpol.gregory_weights(N)
    
    for mm in range(a.shape[1]):
        c2 = np.zeros_like(b.get_mat()[0])
        c3 = np.zeros_like(c2)
        if mm <= interpol.k:
            for jj in range(interpol.k+1):
                for ll in range(interpol.k+1):
                    c2 += (-1)**particle * interpol.R[mm,jj,ll] * mat_prod(a.get_lmx()[t,ll], b.get_mat()[N-jj])
        else:
            for ll in range(mm+1):
                c2 += (-1)**particle * w[mm,ll] * mat_prod(a.get_lmx()[t,mm-ll], b.get_mat()[N-ll])
        
        if mm >= N-interpol.k:
            for jj in range(interpol.k+1):
                for ll in range(interpol.k+1):
                    c3 += interpol.R[N-mm,jj,ll] * mat_prod(a.get_lmx()[t,N-ll], b.get_mat()[jj])
        else:
            for ll in range(N-mm+1):
                c3 += w[N-mm,ll] * mat_prod(a.get_lmx()[t,mm+ll], b.get_mat()[ll])

        c.set_lmx_loc(t, mm, h * (c2 + c3))


@njit
def conv_les_res(t, tp_max, a, b, c, interpol, h, mat_prod):
    assert a.get_les().shape[1] == b.get_adv().shape[0], "Arrays must be same time steps"
    assert a.get_lmx().shape[1] == b.get_rmx().shape[0], "Arrays must be same imaginary time steps"
    
    particle = xor_particle(a.particle_type, b.particle_type)
    assert particle == c.particle_type
    
    w = interpol.gregory_weights(a.shape[1]-1)
    N = a.get_mat().shape[0] - 1
    
    for mm in range(tp_max+1):
        cla = np.zeros_like(a.get_mat()[0])
        cij = np.zeros_like(cla)
        for jj in range(max(interpol.k,mm)+1):
            cla += w[mm,jj] * mat_prod(a.get_les()[t,jj], b.get_adv()[jj,mm])
        for kk in range(N+1):
            cij += w[N,kk] * mat_prod(a.get_lmx()[t,kk], b.get_rmx()[jj,tp_max])
    
    c.set_les_loc(t, tp_max, h * (cla - 1j*cij))