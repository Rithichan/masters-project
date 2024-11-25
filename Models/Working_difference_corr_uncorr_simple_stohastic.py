# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:30:50 2024

@author: rithi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from qutip import ptrace, Qobj, tensor, ket2dm, sigmaz, sigmax, sigmay, qeye, basis

len_states = 10

# Parameters
dt = 0.3
p = 0.1 # Example value, adjust as needed
zs = basis(2, 0)  # |0⟩ in computational basis
os = basis(2, 1)  # |1⟩ in computational basis

# Define the tensor_q operator
tensor_q = np.zeros([2, 2, 2], dtype=complex)

tensor_q[0,0,0] = np.sqrt(p)
tensor_q[0,1,0] = 0
tensor_q[0,0,1] = np.sqrt(1-p)
tensor_q[0,1,1] = 0

tensor_q[1,0,0] = 0
tensor_q[1,1,0] = np.sqrt(1-p)
tensor_q[1,0,1] = 0
tensor_q[1,1,1] = np.sqrt(p)


# Reshape the tensor to match operator dimensions [[4,1],[1,2]]
tensor_q = tensor_q.reshape(4,2)
tensor_qobj = Qobj(tensor_q)
tensor_qobj.dims = [[4], [2]]

# Assign the operator to U_matrix
U_matrix = tensor_qobj

# Define the system and environment states
sys_state = zs
env_state = np.sqrt(1 / 2) * (tensor(zs, zs) + tensor(os, os))  # Bell state

blank_state = np.sqrt(1/2)*(zs+os)

# Combined initial state [[2, 2], [1, 1]]
state = tensor(sys_state,env_state)
state.dims = [[2,2,2],[1,1,1]]

def build_sites(state,len_states):
    i = 0
    sites = []
    while i < len_states:
        if i == 0:
            U_extended = tensor(qeye(2),qeye(2),U_matrix)  # Extend with identity operator
        else:
            U_extended = tensor(qeye(2),qeye(2**(i+1)),U_matrix)  # Extend with identity operator

        # Apply the extended operator
        state_next = U_extended * state
        state_next.dims = [[2,2**(i+1),2,2],[1,1,1,1]]
        array_of_2s = list(np.full(i+4, 2))
        array_of_1s = list(np.full(i+4, 1))
        state_next.dims = [array_of_2s, array_of_1s]
        state = state_next
        state.dims = [[2,2**(i+2),2],[1,1]]
    
        rho = ket2dm(state_next)
        sites.append(rho)
        i += 1
    return sites

def run_model_full_new(state,len_states,dt):
    i = 0
    exp_data = []
    sites = []
    steps = np.arange(1,len_states+1,step=1)
    while i < len_states:
        if i == 0:
            U_extended = tensor(qeye(2),qeye(2),U_matrix)  # Extend with identity operator
            U_op_extend = (np.cos(dt) * tensor(qeye(2),qeye(2),qeye(2),qeye(2)) - 1j*np.sin(dt)*tensor(sigmay(),qeye(2),sigmay(),qeye(2)))
        else:
            U_extended = tensor(qeye(2),qeye(2**(i+1)),U_matrix)  # Extend with identity operator
            U_op_extend = (np.cos(dt) * tensor(qeye(2),qeye(2**(i+1)),qeye(2),qeye(2)) - 1j*np.sin(dt)*tensor(sigmay(),qeye(2**(i+1)),sigmay(),qeye(2)))

        # Apply the extended operator
        state_next = U_extended * state
        state_next.dims = [[2,2**(i+1),2,2],[1,1,1,1]]
        state_next = U_op_extend * state_next
        array_of_2s = list(np.full(i+4, 2))
        array_of_1s = list(np.full(i+4, 1))
        state_next.dims = [array_of_2s, array_of_1s]
        state = state_next
        state.dims = [[2,2**(i+2),2],[1,1]]
    
        rho = ket2dm(state_next)
        sites.append(rho)
        sys = ptrace(rho,0)
        exp_data.append((sys * sigmay()).tr())
        i += 1
    return steps,exp_data,sites

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

def run_model_uncorr(sites,sys_state,len_sites,dt):
    i = 0
    steps = np.arange(1,len_sites+1,step=1)
    sys = ket2dm(sys_state)
    operator = (np.cos(dt)*tensor(qeye(2),qeye(2)) - 1j*np.sin(dt)*tensor(sigmay(),sigmay()))
    operator_dag = operator.dag()
    exp_data = []
    while i < len_sites:
        comb = tensor(sys,Qobj(sites[i+2]))
        result = operator * comb * operator_dag
        sys = ptrace(result,0)
        exp_data.append((sys * sigmay()).tr())
        i += 1
    return exp_data,steps

sites = build_sites(state,len_states)
data_sites = unpack_sites(sites)

exp_data_uncor,steps_uncor = run_model_uncorr(data_sites[-1],sys_state,len_states,dt)
steps_corr,exp_data_corr,_ = run_model_full_new(state, len_states,dt)

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data
ax.plot(steps_uncor, exp_data_uncor, label="Expectation pauli x with no correlation", color="blue", linewidth=2)
ax.plot(steps_corr, exp_data_corr, label="Expectation pauli x with full state", color="black", linewidth=2, linestyle='--')

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

