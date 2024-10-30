# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:20:47 2024

@author: rithi
"""

import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

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

#pauli matrices
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0,1],[1,0]])
pauli_z = np.array([[1,0],[0,-1]])

#initial conditions
dt = 0.2
U = np.cos(dt)**2 * pauli_y - np.sin(dt)**2 * pauli_z
sys_state = np.array([1,0])
state_initial = np.kron(sys_state,a)
total_operator = np.kron(np.kron(U,np.eye(3)),U) #U x I(3) x U
phi_init = 0
theta_init = np.pi/4
len_states = 10
threshold = 10e-5
D = 5

#data
data = []
steps = np.arange(1,len_states+1,step=1)

#create states
for i in range(len_states):
    if i == 0:
        #generation
        U_i = np.kron(np.eye(2),U_mkl(b1,a,b2,phi_init,theta_init))
        mapAB_i = np.kron(np.eye(2),mapAB)
        state = np.matmul(mapAB_i,np.matmul(U_i,state_initial))
        #apply unitary on system and nth env state
        state = np.matmul(total_operator,state)
        memory = state
        #Take SVD
        state = state.reshape(2, 6)
        U, S, Vh = np.linalg.svd(state, full_matrices=False)
        U_dagger = np.conjugate(U).T
        # Reconstruct + uncompute
        left = np.dot(U_dagger,U)
        right = np.dot(np.diag(S), Vh)
        state = np.dot(left,right)  
        state = state.flatten()
        state = np.where(np.abs(state) < threshold, 0, state)
        
        dens_op = np.outer(state,state)
        traced_out_rho = partial_trace(dens_op, [6,2**(i+1)]) #env
        sys = partial_trace(dens_op, [2,3*2**(i+1)],axis=1) #sys
        exp_sig = np.matrix.trace((np.matmul(sys,pauli_z)))
        data.append(exp_sig.item())

    elif i == len_states-1:
        U_i = np.kron(np.eye(2),np.kron(np.matmul(U_mkl(a,b1,b2,0,np.pi/2),U_mkl(b2, a, b1, 0, np.pi/2)),np.eye(2**D)))
        mapAB_i = np.kron(np.eye(2),np.kron(mapAB,np.eye(2**D)))
        state = np.matmul(mapAB_i,np.matmul(U_i,state))
        #apply
        total_operator_i = np.kron(total_operator,np.eye(2**D))
        state = np.matmul(total_operator_i,state)
        #Take SVD
        state = state.reshape(2, 3*2**(D+1))
        U, S, Vh = np.linalg.svd(state, full_matrices=False)
        U_dagger = np.conjugate(U).T
        # Reconstruct + uncompute
        left = np.dot(U_dagger,U)
        right = np.dot(np.diag(S), Vh)
        state = np.dot(left,right)  
        state = state.flatten()
        state = np.where(np.abs(state) < threshold, 0, state)
        
        #SVD on state
        state = state.reshape(2**(D+1),6)
        Uv, Sv, Vhv = np.linalg.svd(state, full_matrices=False)
        Uv = Uv[0:2**D,:]
        state = np.dot(Uv,np.dot(np.diag(Sv),Vhv))
        state = state.flatten()
        state = np.where(np.abs(state) < threshold, 0, state)
        
        dens_op = np.outer(state,state)
        sys = partial_trace(dens_op, [2,3*2**D],axis=1) #sys
        exp_sig = np.matrix.trace((np.matmul(sys,pauli_z)))
        data.append(exp_sig.item())

    else:
        if i < D:
            U_i = np.kron(np.eye(2),np.kron(U_mkl(b2,a,b1,0,np.pi/2),np.eye(2**i)))
            mapAB_i = np.kron(np.eye(2),np.kron(mapAB,np.eye(2**i)))
            state = np.matmul(mapAB_i,np.matmul(U_i,state))
            #apply
            total_operator_i = np.kron(total_operator,np.eye(2**i))
            state = np.matmul(total_operator_i,state)
            #Take SVD
            state = state.reshape(2, 3*2**(i+1))
            U, S, Vh = np.linalg.svd(state, full_matrices=False)
            U_dagger = np.conjugate(U).T
            # Reconstruct + uncompute
            left = np.dot(U_dagger,U)
            right = np.dot(np.diag(S), Vh)
            state = np.dot(left,right)
            state = state.flatten()
            state = np.where(np.abs(state) < threshold, 0, state)
            
            dens_op = np.outer(state,state)
            traced_out_rho = partial_trace(dens_op, [6,2**(i+1)]) #env
            sys = partial_trace(dens_op, [2,3*2**(i+1)],axis=1) #sys
            exp_sig = np.matrix.trace((np.matmul(sys,pauli_z)))
            data.append(exp_sig.item())
            
        elif i == D:
            U_i = np.kron(np.eye(2),np.kron(U_mkl(b2,a,b1,0,np.pi/2),np.eye(2**i)))
            mapAB_i = np.kron(np.eye(2),np.kron(mapAB,np.eye(2**i)))
            state = np.matmul(mapAB_i,np.matmul(U_i,state))
            #apply
            total_operator_i = np.kron(total_operator,np.eye(2**i))
            state = np.matmul(total_operator_i,state)
            #Take SVD
            state = state.reshape(2, 3*2**(i+1))
            U, S, Vh = np.linalg.svd(state, full_matrices=False)
            U_dagger = np.conjugate(U).T
            # Reconstruct + uncompute
            left = np.dot(U_dagger,U)
            right = np.dot(np.diag(S), Vh)
            state = np.dot(left,right)
            state = state.flatten()
            state = np.where(np.abs(state) < threshold, 0, state)
            
            #SVD on state
            state = state.reshape(2**(D+1),6)
            Uv, Sv, Vhv = np.linalg.svd(state, full_matrices=False)
            Uv = Uv[0:2**D,:]
            state = np.dot(Uv,np.dot(np.diag(Sv),Vhv))
            state = state.flatten()
            state = np.where(np.abs(state) < threshold, 0, state)
            
            dens_op = np.outer(state,state)
            sys = partial_trace(dens_op, [2,3*2**(D)],axis=1) #sys
            exp_sig = np.matrix.trace((np.matmul(sys,pauli_z)))
            data.append(exp_sig.item())
            
        else:
            U_i = np.kron(np.eye(2),np.kron(U_mkl(b2,a,b1,0,np.pi/2),np.eye(2**D)))
            mapAB_i = np.kron(np.eye(2),np.kron(mapAB,np.eye(2**D)))
            state = np.matmul(mapAB_i,np.matmul(U_i,state))
            #apply
            total_operator_i = np.kron(total_operator,np.eye(2**D))
            state = np.matmul(total_operator_i,state)
            #Take SVD
            state = state.reshape(2, 3*2**(D+1))
            U, S, Vh = np.linalg.svd(state, full_matrices=False)
            U_dagger = np.conjugate(U).T
            # Reconstruct + uncompute
            left = np.dot(U_dagger,U)
            right = np.dot(np.diag(S), Vh)
            state = np.dot(left,right)
            state = state.flatten()
            state = np.where(np.abs(state) < threshold, 0, state)
            
            #SVD on state
            state = state.reshape(2**(D+1),6)
            Uv, Sv, Vhv = np.linalg.svd(state, full_matrices=False)
            Uv = Uv[0:2**D,:]
            state = np.dot(Uv,np.dot(np.diag(Sv),Vhv))
            state = state.flatten()
            state = np.where(np.abs(state) < threshold, 0, state)
            
            dens_op = np.outer(state,state)
            sys = partial_trace(dens_op, [2,3*2**(D)],axis=1) #sys
            exp_sig = np.matrix.trace((np.matmul(sys,pauli_z)))
            data.append(exp_sig.item())
        
        print(sys)


end = time.time()
print(end - start)

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data
ax.plot(steps, data, label="With Uncompute", color="blue", linewidth=2)

# Add title and labels with proper font size
ax.set_title("Evolution of system", fontsize=16, pad=15)
ax.set_xlabel("Steps (collision with ith env state)", fontsize=14)
ax.set_ylabel("Expectation value", fontsize=14)

# Add a grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend with a frame
ax.legend(loc='upper right', fontsize=12, frameon=True)

# Increase the ticks font size for better visibility
ax.tick_params(axis='both', which='major', labelsize=12)

# Set tighter layout for padding
plt.tight_layout()

#plt.savefig("Evolution of expectation pauli z.jpg",dpi=200)
# Show the plot
plt.show()