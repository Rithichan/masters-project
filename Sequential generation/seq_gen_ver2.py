# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:18:55 2024

Sequential generation 2

@author: rithi
"""

import numpy as np

a = np.array([0,0,1])
b1 = np.array([0,1,0])
b2 = np.array([1,0,0])

mapAB = np.array([[0,0,0], [1,0,0], [0,0,1],[0,1,0],[0,0,0],[0,0,0]])

def U_mkl(m,k,l,phi,theta):
    return (np.cos(theta)*np.outer(k,k)) + (np.cos(theta)*np.outer(l,l)) + (np.exp(1j*phi)*np.sin(theta)*np.outer(k,l)) - (np.exp(-1j*phi)*np.sin(theta)*np.outer(l,k)) + np.outer(m,m)

def partial_trace(rho, dims, axis=0):
    """
    Takes partial trace over the subsystem defined by 'axis'
    rho: a matrix
    dims: a list containing the dimension of each subsystem
    axis: the index of the subsytem to be traced out
    (We assume that each subsystem is square)
    """
    dims_ = np.array(dims)
    # Reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # Each subsystem gets one index for its row and another one for its column
    reshaped_rho = rho.reshape(np.concatenate((dims_, dims_), axis=None))

    # Move the subsystems to be traced towards the end
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims)+axis-1, -1)

    # Trace over the very last row and column indices
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    # traced_out_rho is still in the shape of a tensor
    # Reshape back to a matrix
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])

state_initial = a
phi_init = 0
theta_init = np.pi/4

len_states = 8

#create states
for i in range(len_states):
    if i == 0:
        state = np.matmul(mapAB,np.matmul(U_mkl(b1,a,b2,phi_init,theta_init),state_initial))
    elif i == len_states-1:
        U_i = np.kron(np.matmul(U_mkl(a,b1,b2,0,np.pi/2),U_mkl(b2, a, b1, 0, np.pi/2)),np.eye(2**i))
        mapAB_i = np.kron(mapAB,np.eye(2**i))
        state = np.matmul(mapAB_i,np.matmul(U_i,state))
    else:
        U_i = np.kron(U_mkl(b2,a,b1,0,np.pi/2),np.eye(2**i))
        mapAB_i = np.kron(mapAB,np.eye(2**i))
        state = np.matmul(mapAB_i,np.matmul(U_i,state))
        
    dens_op = np.outer(state,state)
    traced_out_rho = partial_trace(dens_op, [3,2**(i+1)])
    
    