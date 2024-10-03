# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:27:23 2024

@author: Rithichan

Task 1: Evolution of some toy system given by H = pauli_y (x) pauli_y
system_initial = ket(0)bra(0)
environment_initial = ket(0)bra(0)
"""

import numpy as np
import qutip
import matplotlib.pyplot as plt

#Define Pauli Matrices
pauli_y = np.matrix([[0,-1j],[1j,0]])
pauli_x = np.matrix([[0,1],[1,0]])
pauli_z = np.matrix([[1,0],[0,-1]])

identity = np.matrix([[1,0],[0,1]])

H = np.kron(pauli_y, pauli_y)
identity_x4 = np.kron(identity,identity)

zero_basis = np.array([1,0]).T
one_basis = np.array([0,1]).T

set_basis = [zero_basis,one_basis]

sys_initial = np.outer(zero_basis,zero_basis)
env_initial = np.outer(zero_basis,zero_basis)

density_matrix  = np.kron(sys_initial,env_initial)

t = 0
dt = 0.1

data = []
time = []

while t < 2*np.pi:

    propagator = np.cos(t)*identity_x4 - 1j*np.sin(t)*H
    propagator_dagger = propagator.conj().T
    
    sys_final = np.matrix([[0,0],[0,0]]) #slate
    
    for basis in set_basis:
        #I (tensor product) basis_vectors
        # env initialised as ket(zero).bra(zero)
        right_env = np.kron(identity,zero_basis).T
        left_env = np.kron(identity,zero_basis)
        right = np.kron(identity,basis).T
        left = np.kron(identity,basis)
        
        #Kraus Operators
        K = np.matmul(left,(np.matmul(propagator,right_env)))
        K_dagger = np.matmul(left_env,(np.matmul(propagator_dagger,right)))
        
        sys_t_i = np.matmul(K,(np.matmul(sys_initial,K_dagger)))
        sys_final = sys_final + sys_t_i
    
    exp_sigx = np.matrix.trace((np.matmul(sys_final,pauli_z)))
    data.append(exp_sigx.item())
    time.append(t)
    
    t += dt
    
# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data
ax.plot(time, data, label="Expectation pauli z", color="blue", linewidth=2)

# Add title and labels with proper font size
ax.set_title("Evolution of system", fontsize=16, pad=15)
ax.set_xlabel("Time", fontsize=14)
ax.set_ylabel("Expectation value", fontsize=14)

# Add a grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend with a frame
ax.legend(loc='upper right', fontsize=12, frameon=True)

# Increase the ticks font size for better visibility
ax.tick_params(axis='both', which='major', labelsize=12)

# Set tighter layout for padding
plt.tight_layout()

plt.savefig("Evolution of expectation pauli z.jpg",dpi=200)
# Show the plot
plt.show()
