# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:28:05 2024

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

set_basis = [zero_basis,one_basis]

H = np.kron(pauli_y, pauli_y)
identity = np.matrix([[1,0],[0,1]])
identity_x4 = np.kron(identity,identity)

'''
Continuosly acting on rho
'''

t = 0
dt = 0.01

data = []
time = []

def theta_t(t):
    return t

psi_env = np.cos(theta_t(t))*zero_basis + np.sin(theta_t(t))*one_basis

sys = sys_initial = np.outer(zero_basis,zero_basis)
env = env_initial = np.outer(psi_env,psi_env)
rho = rho_initial = np.kron(sys,env)

propagator = np.cos(dt)*identity_x4 - 1j*np.sin(dt)*H
propagator_dagger = propagator.conj().T

while t < 2*np.pi:
    
    

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data
ax.plot(time, data, label="Expectation pauli z", color="blue", linewidth=2)

# Add title and labels with proper font size
ax.set_title("Evolution of system (Using Kraus Operators)", fontsize=16, pad=15)
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

plt.savefig("Evolution of expectation pauli z(Kraus Operators).jpg",dpi=200)
# Show the plot
plt.show()