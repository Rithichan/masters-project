# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:27:23 2024

@author: ASUS
"""

import numpy as np

#Define Pauli Matrices
pauli_y = np.matrix([[0,-1j],[1j,0]])
pauli_x = np.matrix([[0,1],[1,0]])
pauli_z = np.matrix([[1,0],[0,-1]])

identity = np.matrix([[1,0],[0,1]])

H = np.kron(pauli_y, pauli_y)
identity_x4 = np.kron(identity,identity)

zero_basis = np.array([1,0])
one_basis = np.array([0,1])


sys_initial = np.outer(zero_basis,zero_basis)
env_initial = np.outer(zero_basis,zero_basis)

density_matrix  = np.kron(sys_initial,env_initial)

t = (np.pi)/2
propagator = np.cos(t)*identity_x4 - 1j*np.sin(t)*H
propagator_dagger = propagator.conj().T

density_t = np.matmul(np.matmul(propagator,density_matrix),propagator_dagger)

density_t = density_t.reshape((2,2,2,2))