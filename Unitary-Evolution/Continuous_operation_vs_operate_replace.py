# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:39:06 2024

@author: rithi
"""

import numpy as np
import qutip
import matplotlib.pyplot as plt

#Define Pauli Matrices
pauli_y = np.matrix([[0,-1j],[1j,0]])
pauli_x = np.matrix([[0,1],[1,0]])
pauli_z = np.matrix([[1,0],[0,-1]])

zero_basis = np.array([1,0]).T
one_basis = np.array([0,1]).T

H = np.kron(pauli_y, pauli_y)
identity = np.matrix([[1,0],[0,1]])
identity_x4 = np.kron(identity,identity)


'''
Continuosly acting on rho
'''

t = 0
dt = 0.1

data = []
time = []

psi_env = np.cos(t)*zero_basis + np.sin(t)*one_basis
state_env = np.outer(psi_env,psi_env)

sys_initial = np.outer(zero_basis,zero_basis)
    
density_matrix = np.kron(sys_initial,state_env)

propagator = np.cos(dt)*identity_x4 - 1j*np.sin(dt)*H
propagator_dagger = propagator.conj().T

while t < 4*np.pi:

    density_matrix_dt = np.matmul(propagator,(np.matmul(density_matrix,propagator_dagger)))
    density_matrix_dt = np.asarray(density_matrix_dt)
    density_matrix = density_matrix_dt
    
    density_matrix_dt = density_matrix_dt.reshape([2,2,2,2])
    rho_s = np.trace(density_matrix_dt, axis1=1, axis2=3)
    rho_e = np.trace(density_matrix_dt, axis1=0, axis2=2)
    
    exp_sigz = np.matrix.trace((np.matmul(rho_s,pauli_z)))
    data.append(exp_sigz.item())
    time.append(t)

    t += dt

'''
Replace and separate sys and env at each step
'''

t = 0
dt = 0.1

data1 = []
time1 = []

rho_s = sys_initial = np.outer(zero_basis,zero_basis)

while t < 4*np.pi:

    psi_env = np.cos(t)*zero_basis + np.sin(t)*one_basis
    state_env = np.outer(psi_env,psi_env)

    density_matrix = np.kron(rho_s,state_env)

    propagator = np.cos(dt)*identity_x4 - 1j*np.sin(dt)*H
    propagator_dagger = propagator.conj().T

    density_matrix_dt = np.matmul(propagator,(np.matmul(density_matrix,propagator_dagger)))
    density_matrix_dt = np.asarray(density_matrix_dt)

    density_matrix_dt = density_matrix_dt.reshape([2,2,2,2])
    rho_s = np.trace(density_matrix_dt, axis1=1, axis2=3)
    
    exp_sigz = np.matrix.trace((np.matmul(rho_s,pauli_z)))
    data1.append(exp_sigz.item())
    time1.append(t)

    t += dt

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data
ax.plot(time, data, label="Expectation pauli z", color="blue", linewidth=2)
ax.plot(time1, data1, label=f"Expectation pauli z replace (dt={dt})", color="red", linewidth=2)

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

plt.savefig("Evolution_replace_vs_cont.jpg",dpi=200)
# Show the plot
plt.show()



