import numpy as np
from numba import njit
from .printing import exp_string



@njit
def wpped_matrix_inv(A):
    assert A.ndim==2, "Inversion is for 2D arrays"
    assert A.shape[0] == A.shape[1], "Only squared matrices have inverse"
    
    n = A.shape[0]
    auxiliar = np.empty((n,2*n), dtype=A.dtype)
    auxiliar[:,:n] = A
    auxiliar[:,n:] = np.eye(n, dtype=A.dtype)
    
    # Gauss elimination
    for i in range(n-1):
        if auxiliar[i,i] == 0:
            for l in range(i+1,n):
                if auxiliar[l,i] != 0:
                    new_i_row = np.copy(auxiliar[l,:])
                    auxiliar[l,:] = auxiliar[i,:]
                    auxiliar[i,:] = new_i_row
                    break
                assert l<n-1, "Non invertible matrix"
        if abs(auxiliar[i,i]) > 1e4 or abs(auxiliar[i,i]) < 1e-4: # Normalization avoids over/underflow
            auxiliar[i,:] /= abs(auxiliar[i,i])
        for j in range(i+1,n):
            auxiliar[j,:] = auxiliar[j,:]*auxiliar[i,i] - auxiliar[i,:]*auxiliar[j,i]
    
    # Jordan elimination
    for i in range(n-1,0,-1):
        if auxiliar[i,i] == 0:
            for l in range(i-1,-1,-1):
                if auxiliar[l,i] != 0:
                    new_i_row = auxiliar[l,:]
                    auxiliar[l,:] = auxiliar[i,:]
                    auxiliar[i,:] = new_i_row
                    break
                assert l>0, "Non invertible matrix"
        if abs(auxiliar[i,i]) > 1e4 or abs(auxiliar[i,i]) < 1e-4: # Normalization avoids over/underflow
            auxiliar[i,:] /= abs(auxiliar[i,i])
        for j in range(i-1,-1,-1):
            auxiliar[j,:] = auxiliar[j,:]*auxiliar[i,i] - auxiliar[i,:]*auxiliar[j,i]
    
    # Normalization
    for i in range(n):
        assert auxiliar[i,i] != 0, "Non invertible matrix"
        auxiliar[i,:] /= auxiliar[i,i]
    return auxiliar[:,n:]



@njit
def matrix_inv(arr):
    assert arr.ndim>1, "Inversion is for 2D arrays"
    
    if arr.ndim == 2:
        return wpped_matrix_inv(arr)
    
    else:
        sh = arr.shape
        nmats = 1
        for k in range(arr.ndim-2):
            nmats *= arr.shape[k]
        arr = np.ascontiguousarray(arr.reshape((nmats,arr.shape[-2],arr.shape[-1])))
        ret = np.empty_like(arr)
        for n in range(nmats):
            ret[n,:,:] = wpped_matrix_inv(arr[n,:,:])
        return np.ascontiguousarray(ret.reshape(sh))


@njit
def inner(v, u):
    assert v.ndim==1 and u.ndim==1, "1D arrays are vectors"
    assert v.size == u.size, "Inner product only for same size vectors"
    
    if v.dtype is np.dtype(np.complex128) or u.dtype is np.dtype(np.complex128):
        w = 0j
        for i in range(v.size):
            w += v[i].conjugate() * u[i]
    else:
        w = 0.
        for i in range(v.size):
            w += v[i] * u[i]
    return w



@njit
def QRdecomp(A):
    assert A.ndim==2 and A.shape[0]==A.shape[1], "QR decomposition only admits 2D squared matrices"
    
    U = np.empty_like(A)
    E = np.empty_like(A)
    
    n = A.shape[0]
    for i in range(n):
        U[:,i] = A[:,i]
        for j in range(i):
            U[:,i] -= inner(U[:,j], A[:,i]) / inner(U[:,j], U[:,j]) * U[:,j]
        E[:,i] = U[:,i] / np.sqrt(inner(U[:,i], U[:,i]))
    
    R = np.zeros_like(A)
    for i in range(n):
        for j in range(i,n):
            R[i,j] = inner(E[:,i], A[:,j])
    
    return E, R


@njit
def wpped_eig(A, tol=1e-8):
    assert A.ndim==2, "Eigen is for 2D arrays"
    assert A.shape[0] == A.shape[1], "Only squared matrices have eigen"
    
    Atf = np.copy(A)
    n = A.shape[0]
    num_iter = 0
    while True:
        num_iter += 1
        Q, R = QRdecomp(Atf)
        Atf = np.zeros_like(Atf)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    Atf[i,k] += R[i,j] * Q[j,k]
        check_conv = True
        conv_sum = 0.0
        for i in range(n):
            for j in range(i):
                check_conv *= np.abs(Atf[i,j]) < tol
                conv_sum += np.abs(Atf[i,j])
        if num_iter%100==0:
            print(num_iter//100, conv_sum)
        if check_conv:
            break
    
    evals = np.empty_like(A[0])
    evecs = np.empty_like(A)
    
    for i in range(n):
        evals[i] = Atf[i,i]
    for i in range(n):
        rdvec = np.random.random((n,)) + 1j*np.random.random((n,))
        v = rdvec / np.sqrt(inner(rdvec,rdvec))
        for j in range(n):
            if j != i:
                new_v = np.zeros_like(v)
                for k in range(n):
                    for l in range(n):
                        new_v[k] += (A[k,l] - evals[j]*(k==l)) * v[l]
                v = new_v / np.sqrt(inner(new_v,new_v))
        evecs[:,i] = np.copy(v)
    
    return evals, evecs


@njit
def eig(arr):
    assert arr.ndim>1, "Eigen is for 2D arrays"
    
    if arr.ndim == 2:
        return wpped_eig(arr)
    
    else:
        sh = arr.shape
        nmats = 1
        for k in range(arr.ndim-2):
            nmats *= arr.shape[k]
        arr = np.ascontiguousarray(arr.reshape((nmats,arr.shape[-2],arr.shape[-1])))
        ret_vecs = np.empty_like(arr)
        ret_vals = np.empty((nmats,arr.shape[-2]), dtype=np.complex128)
        for n in range(nmats):
            valsn, vecsn = wpped_eig(arr[n,:,:])
            ret_vals[n,:] = valsn
            ret_vecs[n,:,:] = vecsn
        return np.ascontiguousarray(ret_vals.reshape(sh[:-1])), np.ascontiguousarray(ret_vecs.reshape(sh))


@njit
def norm2_matrix(arr):
    assert arr.ndim==2, "Norm 2 for matrices"
    
    arrsq = np.zeros((arr.shape[0],arr.shape[0]), dtype=np.complex128)
    for ii in range(arr.shape[0]):
        for jj in range(arr.shape[0]):
            for kk in range(arr.shape[1]):
                arrsq[ii,jj] += arr[ii,kk] * arr[jj,kk].conjugate()
    
    w,_ = eig(arrsq)
    return np.sqrt(w.real.max())