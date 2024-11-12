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

def random_unitary(seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    # Generate a random complex matrix
    random_matrix = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    
    # Perform QR decomposition and normalize to get a unitary matrix
    q, _ = np.linalg.qr(random_matrix)
    return q

#states
zs = np.array([0,1])
os = np.array([1,0])

#pauli matrices
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0,1],[1,0]])
pauli_z = np.array([[1,0],[0,-1]])

#initial condition
dt = 0.1
U = random_unitary(1)
sys_state = np.array([1,0])
state_initial = np.kron(sys_state,a)
total_operator = np.kron(np.kron(U,np.eye(3)),U) #U x I(3) x U x I(2**i)
phi_init = 0
theta_init = np.pi/4
len_states = 10

#data
data = []
steps = np.arange(1,len_states+1,step=1)

for i in range(len_states):
    if i == 0:
        #generation
        U_i = np.kron(np.eye(2),U_mkl(b1,a,b2,phi_init,theta_init))
        mapAB_i = np.kron(np.eye(2),mapAB)
        state = np.matmul(mapAB_i,np.matmul(U_i,state_initial))

        state = np.matmul(total_operator,state)
        
        memory = state

        #Uncompute
        state = state.reshape(6,2)
        U,S,Vh = np.linalg.svd(state,full_matrices=False)
        
        Vh_dag = np.conjugate(Vh).T
        U_dag = np.conjugate(U).T
        
        state = np.dot(np.dot(np.dot(U,np.diag(S)),Vh),Vh_dag)
        state = state.flatten()
        
    elif i == len_states-1:
        U_i = np.kron(np.eye(2),np.kron(np.matmul(U_mkl(a,b1,b2,0,np.pi/2),U_mkl(b2, a, b1, 0, np.pi/2)),np.eye(6)))
        mapAB_i = np.kron(np.eye(2),np.kron(mapAB,np.eye(6)))
        state = np.matmul(mapAB_i,np.matmul(U_i,state))
        
        total_operator_i = np.kron(total_operator,np.eye(6))
        state = np.matmul(total_operator_i,state)
        
        memory = state
    
        #Uncompute
        state = state.reshape(6,2*6)
        U,S,Vh = np.linalg.svd(state,full_matrices=False)
                
        Vh_dag = np.conjugate(Vh).T

        state = np.dot(np.dot(U,np.diag(S)),np.dot(Vh,Vh_dag))
        state = state.flatten()

    else:
        if len(state) == 36:

            U_i = np.kron(np.eye(2),np.kron(U_mkl(b2,a,b1,0,np.pi/2),np.eye(6)))
            mapAB_i = np.kron(np.eye(2),np.kron(mapAB,np.eye(6)))
            state = np.matmul(mapAB_i,np.matmul(U_i,state))
    
            total_operator_i = np.kron(total_operator,np.eye(6))
            state = np.matmul(total_operator_i,state)
            
            memory = state
    
            #Uncompute
            state = state.reshape(6,2*6)
            U,S,Vh = np.linalg.svd(state,full_matrices=False)
                
            Vh_dag = np.conjugate(Vh).T
                
            state = np.dot(np.dot(U,np.diag(S)),np.dot(Vh,Vh_dag))
            state = state.flatten()

        else:
            U_i = np.kron(np.eye(2),np.kron(U_mkl(b2,a,b1,0,np.pi/2),np.eye(2**i)))
            mapAB_i = np.kron(np.eye(2),np.kron(mapAB,np.eye(2**i)))
            state = np.matmul(mapAB_i,np.matmul(U_i,state))
            
            memory = state
    
            total_operator_i = np.kron(total_operator,np.eye(2**i))
            state = np.matmul(total_operator_i,state)
    
            #Uncompute
            state = state.reshape(6,2**(i+1))
            U,S,Vh = np.linalg.svd(state,full_matrices=False)
                
            Vh_dag = np.conjugate(Vh).T
                
            state = np.dot(np.dot(U,np.diag(S)),np.dot(Vh,Vh_dag))
            state = state.flatten()
             
    if len(state) == 36:
        rho = np.outer(state,state)
        sys = partial_trace(rho, [2,18],axis=1) #sys
        exp_sig = np.matrix.trace((np.matmul(sys,pauli_z)))
        data.append(exp_sig.item())
    else:
        rho = np.outer(state,state)
        sys = partial_trace(rho, [2,3*2**(i+1)],axis=1) #sys
        exp_sig = np.matrix.trace((np.matmul(sys,pauli_z)))
        data.append(exp_sig.item())


# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data
ax.plot(steps, data, label="Expectation pauli z", color="blue", linewidth=2)

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