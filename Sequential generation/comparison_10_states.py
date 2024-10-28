# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:02:59 2024

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
total_operator = np.kron(np.kron(U,np.eye(3)),U) #U x I(3) x U x I(2**i)
phi_init = 0
theta_init = np.pi/4
len_states = 10

#data
data1 = []
steps1 = np.arange(1,len_states+1,step=1)

#create states
for i in range(len_states):
    if i == 0:
        #generation
        U_i = np.kron(np.eye(2),U_mkl(b1,a,b2,phi_init,theta_init))
        mapAB_i = np.kron(np.eye(2),mapAB)
        state = np.matmul(mapAB_i,np.matmul(U_i,state_initial))
        #apply unitary on system and nth env state
        state = np.matmul(total_operator,state)
    elif i == len_states-1:
        U_i = np.kron(np.eye(2),np.kron(np.matmul(U_mkl(a,b1,b2,0,np.pi/2),U_mkl(b2, a, b1, 0, np.pi/2)),np.eye(2**i)))
        mapAB_i = np.kron(np.eye(2),np.kron(mapAB,np.eye(2**i)))
        state = np.matmul(mapAB_i,np.matmul(U_i,state))
        #apply
        total_operator_i = np.kron(total_operator,np.eye(2**i))
        state = np.matmul(total_operator_i,state)
    else:
        U_i = np.kron(np.eye(2),np.kron(U_mkl(b2,a,b1,0,np.pi/2),np.eye(2**i)))
        mapAB_i = np.kron(np.eye(2),np.kron(mapAB,np.eye(2**i)))
        state = np.matmul(mapAB_i,np.matmul(U_i,state))
        #apply
        total_operator_i = np.kron(total_operator,np.eye(2**i))
        state = np.matmul(total_operator_i,state)

    dens_op = np.outer(state,state)
    traced_out_rho = partial_trace(dens_op, [6,2**(i+1)]) #env
    sys = partial_trace(dens_op, [2,3*2**(i+1)],axis=1) #sys
    exp_sig = np.matrix.trace((np.matmul(sys,pauli_z)))
    data1.append(exp_sig.item())


#pauli matrices
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_x = np.array([[0,1],[1,0]])
pauli_z = np.array([[1,0],[0,-1]])

zs = np.array([0,1])
os = np.array([1,0])

def tp(state,N):
    temp = state
    for _ in range(N-1):
        state = np.kron(state,temp)
    return state

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

dt = 0.2
zs_10 = tp(zs,10)
os_10 = tp(os,10)

GHZ = 1/(np.sqrt(2)) * (zs_10 - os_10)

total = np.kron(os,GHZ)
U = np.cos(dt)**2 * pauli_y - np.sin(dt)**2 * pauli_z

data = []
steps = np.arange(1,11,step=1)

for i in range(10):

    if i == 0:
        U_i = np.kron(np.kron(U,U),np.eye(2**9))
        state = np.matmul(U_i,total)
    elif i == 9:
        U_i = np.kron(np.eye(2**9),np.kron(U,U))
        state = np.matmul(U_i,state)
    else:
        U_i = np.kron(U,np.kron(np.eye(2**i),np.kron(U,np.eye(2**(9-i)))))
        state = np.matmul(U_i,state)

    dens_op = np.outer(state,state)
    sys = partial_trace(dens_op, [2,2**(10)],axis=1) #sys
    exp_sig = np.matrix.trace((np.matmul(sys,pauli_z)))
    data.append(exp_sig.item())

end = time.time()
print(end - start)

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data
ax.plot(steps, data, label="Expectation pauli z (Direct)", color="blue", linewidth=2)
ax.plot(steps1, data1, label="Expectation pauli z (Seq Gen)", color="red", linewidth=2)


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

plt.savefig("Comparison_seq_gen_vs_direct.jpg",dpi=200)
# Show the plot
plt.show()