
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:30:50 2024

@author: rithi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from qutip import ptrace, Qobj, tensor, ket2dm, sigmaz, sigmax, sigmay, qeye, basis



'''
---Parameters--
'''

len_states = 10
dt = 0.01
p = 0.001 # Example value, adjust as needed
zs = basis(2, 0)  # |0⟩ in computational basis
os = basis(2, 1)  # |1⟩ in computational basis

# Define the system and environment states
sys_state = zs
env_state = np.sqrt(1 / 2) * (tensor(zs, zs) + tensor(os, os))  # Bell state
# Combined initial state [[2, 2], [1, 1]]
state = tensor(env_state,sys_state)
state.dims = [[2,2,2],[1,1,1]]


'''
---Sequential Generation Tensor---
'''

# Define the tensor_q operator
tensor_q = np.zeros([2, 2, 2], dtype=complex)

tensor_q[0,0,0] = np.sqrt(p)
tensor_q[0,1,0] = 0
tensor_q[0,0,1] = np.sqrt(1-p)
tensor_q[0,1,1] = 0

tensor_q[1,0,0] = 0
tensor_q[1,1,0] = -np.sqrt(1-p)
tensor_q[1,0,1] = 0
tensor_q[1,1,1] = np.sqrt(p)

# Reshape the tensor to match operator dimensions [[4,1],[1,2]]
tensor_q = tensor_q.reshape(4,2)
tensor_qobj = Qobj(tensor_q)
tensor_qobj.dims = [[4], [2]]

# Assign the operator to U_matrix
U_matrix = tensor_qobj

'''
---Operators---
'''
def random_unitary(seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    # Generate a random complex matrix
    random_matrix = np.random.randn(2, 2) + 1j * np.random.randn(2,2)
    
    # Perform QR decomposition and normalize to get a unitary matrix
    q, _ = np.linalg.qr(random_matrix)
    return q


#operation
theta = 1.30 #defines entanglement
H_op = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, np.cos(theta), -np.sin(theta)],
                 [0, 0, np.sin(theta), np.cos(theta)]])

H_op = Qobj(H_op)
H_op.dims = [[2,2],[2,2]]

#swap
swap_operator = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

swap_operator = Qobj(swap_operator)
swap_operator.dims = [[2,2],[2,2]]

'''
--Models--
'''

def build_sites(state,len_states):
    i = 0
    sites = []
    while i < len_states:
        if i == 0:
            U_extended = tensor(U_matrix,qeye(2),qeye(2))  # Extend with identity operator
        else:
            U_extended = tensor(qeye(2**(i)),U_matrix,qeye(2),qeye(2))  # Extend with identity operator

        # Apply the extended operator
        state_next = U_extended * state
        array_of_2s = list(np.full(i+4, 2))
        array_of_1s = list(np.full(i+4, 1))
        state_next.dims = [array_of_2s, array_of_1s]
        state = state_next
        state.dims = [[2**(i+1),2,2,2],[1,1,1,1]]

        rho = ket2dm(state_next)
        sites.append(rho)
        i += 1
    return sites

def unpack_sites(sites):
    i = 0
    data_sites = []
    for element in sites:
        array_of_2s = list(np.full(i+4, 2))
        element.dims = [array_of_2s,array_of_2s]
    
        j = 0
        element_sites = []
        while j < len(array_of_2s):
            site_j = ptrace(element,j)
            site_j = site_j.full()
            element_sites.append(site_j)
            j += 1
    
        data_sites.append(element_sites)
        i += 1
    return data_sites

def run_model_uncorr(sys_state,len_sites,dt,measurement,H_op):
    i = 0
    steps = np.arange(1,len_sites+1,step=1)
    sys = ket2dm(sys_state)
    operator = H_op
    operator_dag = H_op.dag()
    exp_data = []
    while i < len_sites:
        comb = tensor(sys,Qobj(0.5*qeye(2)))
        result = operator * comb * operator_dag
        sys = ptrace(result,0)
        exp_data.append((sys * measurement).tr())
        i += 1
    return exp_data,steps

def run_model_full_new(state,len_states,dt,measurement,H_op):
    i = 0
    exp_data = []
    sites = []
    steps = np.arange(1,len_states+1,step=1)
    while i < len_states:
        if i == 0:
            U_extended = tensor(qeye(2),U_matrix,qeye(2))  # Extend with identity operator
            U_op_extend = tensor(qeye(2),qeye(2),H_op)
            swap_extend = tensor(qeye(2),qeye(2),swap_operator)
        else:
            U_extended = tensor(qeye(2),U_matrix,qeye(2),qeye(2**i))  # Extend with identity operator
            U_op_extend = tensor(qeye(2),qeye(2),H_op,qeye(2**i))
            swap_extend = tensor(qeye(2),qeye(2),swap_operator,qeye(2**i))
        # Apply the extended operator
        state_next = U_extended * state
        if i == 0:
            state_next.dims = [[2,2,2,2],[1,1,1,1]]
        else:
            state_next.dims = [[2,2,2,2,2**i],[1,1,1,1,1]]
        state_next = U_op_extend * state_next
        state_next = swap_extend * state_next
        array_of_2s = list(np.full(i+4, 2))
        array_of_1s = list(np.full(i+4, 1))
        state_next.dims = [array_of_2s, array_of_1s]
        state = state_next.copy()
        state.dims = [[2,2,2,2**(i+1)],[1,1,1,1]]
    
        rho = ket2dm(state_next)
        sites.append(rho)
        sys = ptrace(rho,2)
        exp_data.append((sys * measurement).tr())
        i += 1
    return steps,exp_data,sites

def run_model_uncompute(state,len_states,dt,measurement,H_op):
    i = 0
    exp_data = []
    sites = []
    steps = np.arange(1,len_states+1,step=1)
    while i < len_states:
        if i == 0:
            U_extended = tensor(qeye(2),U_matrix,qeye(2))  # Extend with identity operator
            U_op_extend = tensor(qeye(2),qeye(2),H_op)
            swap_extend = tensor(qeye(2),qeye(2),swap_operator)
            
            state_next = U_extended * state
            state_next.dims = [[2,2,2,2],[1,1,1,1]]
            state_next = U_op_extend * state_next
            state_next = swap_extend * state_next
            state = state_next

            #uncompute
            state = state.full().flatten()
            state = state.reshape(8,2)
            U,S,Vh = np.linalg.svd(state,full_matrices=False)
            
            Vh_dag = np.conjugate(Vh).T

            state = np.dot(np.dot(U,np.diag(S)),np.dot(Vh,Vh_dag))
            state = state.flatten()
            
            state = Qobj(state)
            array_of_2s = list(np.full(i+4, 2))
            array_of_1s = list(np.full(i+4, 1))
            state.dims = [array_of_2s,array_of_1s]

            rho = ket2dm(state)
            sites.append(rho)
            sys = ptrace(rho,2)
            exp_data.append((sys * measurement).tr())

            state.dims = [[2,2,2,2**(i+1)],[1,1,1,1]]

        elif state.shape[0] != 64:
            U_extended = tensor(qeye(2),U_matrix,qeye(2),qeye(2**i))  # Extend with identity operator
            U_op_extend = tensor(qeye(2),qeye(2),H_op,qeye(2**i))
            swap_extend = tensor(qeye(2),qeye(2),swap_operator,qeye(2**i))
            
            state_next = U_extended * state
            state_next.dims = [[2,2,2,2,2**i],[1,1,1,1,1]]
            state_next = U_op_extend * state_next
            state_next = swap_extend * state_next
            state = state_next

            #uncompute
            state = state.full().flatten()
            state = state.reshape(8,2**(i+1))
            U,S,Vh = np.linalg.svd(state,full_matrices=False)
            
            Vh_dag = np.conjugate(Vh).T

            state = np.dot(np.dot(U,np.diag(S)),np.dot(Vh,Vh_dag))
            state = state.flatten()
            
            state = Qobj(state)
            array_of_2s = list(np.full(i+4, 2))
            array_of_1s = list(np.full(i+4, 1))
            state.dims = [array_of_2s,array_of_1s]

            rho = ket2dm(state)
            sites.append(rho)
            sys = ptrace(rho,2)
            exp_data.append((sys * measurement).tr())

            state.dims = [[2,2,2,2**(i+1)],[1,1,1,1]]
        else:
            state.dims = [[2,2,2,2,2,2],[1,1,1,1,1,1]]
            U_extended = tensor(qeye(2),U_matrix,qeye(2),qeye(2),qeye(2),qeye(2))
            U_op_extend = tensor(qeye(2),qeye(2),H_op,qeye(2),qeye(2),qeye(2))
            swap_extend = tensor(qeye(2),qeye(2),swap_operator,qeye(2),qeye(2),qeye(2))

            state_next = U_extended * state
            state_next.dims = [[2,2,2,2,2,2,2],[1,1,1,1,1,1,1]]
            state_next = U_op_extend * state_next
            state_next = swap_extend * state_next
            state = state_next
            
            state = state.full().flatten()
            state = state.reshape(8,16)
            U,S,Vh = np.linalg.svd(state,full_matrices=False)
            
            Vh_dag = np.conjugate(Vh).T

            state = np.dot(np.dot(U,np.diag(S)),np.dot(Vh,Vh_dag))
            state = state.flatten()
            
            state = Qobj(state)
            array_of_2s = list(np.full(6, 2))
            array_of_1s = list(np.full(6, 1))
            state.dims = [array_of_2s,array_of_1s]
            
            rho = ket2dm(state)
            sites.append(rho)
            sys = ptrace(rho,2)
            exp_data.append((sys * measurement).tr())
        i += 1
    return exp_data,steps,sites

'''
---simulate---
'''

steps_corr,exp_data_corr,sites_full = run_model_full_new(state, 10,dt,sigmax(),H_op)
exp_data_uncomp,steps_uncomp,sites_reduced = run_model_uncompute(state, 50, dt, sigmax(), H_op)
exp_data_uncor,steps_uncor = run_model_uncorr(sys_state, 50, dt, sigmax(), H_op)
#data_sites_full = unpack_sites(sites_full)

'''
---Plot---
'''

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data
ax.plot(steps_uncor, exp_data_uncor, label="Expectation pauli x uncorrelated", color="blue", linewidth=2)
ax.plot(steps_uncomp, exp_data_uncomp, label="Expectation pauli x with uncomp", color="red", linewidth=2, linestyle='-.')
ax.plot(steps_corr, exp_data_corr, label="Expectation pauli x with full state", color="black", linewidth=2, linestyle='--')

# Add title and labels with proper font size
ax.set_title("Evolution of system", fontsize=16, pad=15)
ax.set_xlabel("Steps (collision with ith env state)", fontsize=14)
ax.set_ylabel("Expectation value", fontsize=14)

# Add a grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend with a frame
ax.legend(loc='lower right', fontsize=10)


# Increase the ticks font size for better visibility
ax.tick_params(axis='both', which='major', labelsize=12)

# Set tighter layout for padding
plt.tight_layout(pad=2)

#plt.savefig("Evolution of expectation pauli z.jpg",dpi=200)
# Show the plot
plt.show()   