# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:30:50 2024

@author: rithi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from qutip import ptrace, Qobj, tensor, ket2dm, sigmaz, sigmax, sigmay

#pauli matrices
pauli_y = np.array([[0, -1j], [1j, 0]],dtype=complex)
pauli_x = np.array([[0,1],[1,0]],dtype=complex)
pauli_z = np.array([[1,0],[0,-1]],dtype=complex)

#initial condition
len_states = 10
dt = 0.1
U_op = (np.cos(dt)**2 * pauli_y - np.sin(dt)**2 * pauli_y)**(0.01)
p = 0.03

zs = np.array([1,0],dtype=complex)
os = np.array([0,1],dtype=complex)

A0 = np.array([[np.sqrt(1-p),np.sqrt(p)],[0,0]])
A1 = np.array([[0,0],[np.sqrt(p),np.sqrt(1-p)]])

A = np.array([[np.sqrt(1-p),np.sqrt(p)]
              ,[0,0]
              ,[0,0]
              ,[np.sqrt(p),np.sqrt(1-p)]],dtype=complex)

U_matrix = np.array([
    [np.sqrt(1 - p), 0, 0, np.sqrt(p)],
    [0, np.sqrt(1 - p), np.sqrt(p), 0],
    [0, np.sqrt(p), -np.sqrt(1 - p), 0],
    [np.sqrt(p), 0, 0, -np.sqrt(1 - p)]
])

U_matrix = A


system_state = zs
env_state = np.sqrt(1/2) * (np.kron(zs,zs) + np.kron(os,os))
state = np.kron(system_state,env_state)


'''
def run_model_no_uncompute(state):
    #data
    data = []
    steps = np.arange(1,len_states+1,step=1)
    
    for i in range(len_states):
        if i == 0:
            state_next = np.matmul(np.kron(np.eye(4),U_matrix),state)
            state = state_next

        else:
            state_next = np.matmul(np.kron(np.eye(2**(i+2)),U_matrix),state)
            state = state_next

        state_obj = Qobj(state)
        array_of_2s = list(np.full(i+4, 2))
        state_obj.dims = [array_of_2s,[1]]
        rho = ket2dm(state_obj)
        rho = rho.full()
        data.append(rho)
    return data,steps
'''


def run_model_no_uncompute(state):
    #data
    data = []
    steps = np.arange(1,len_states+1,step=1)
    
    for i in range(len_states):
        if i == 0:
            state_next = np.dot(np.kron(A,np.eye(4)),state)
            state = state_next
        else:
            state_next = np.dot(np.kron(np.kron(np.eye(2**i),A),np.eye(4)),state)
            state = state_next

        state_obj = Qobj(state)
        array_of_2s = list(np.full(i+4, 2))
        state_obj.dims = [array_of_2s,[1]]
        rho = ket2dm(state_obj)
        rho = rho.full()
        data.append(rho)
    return data,steps

data,_ = run_model_no_uncompute(state)

i = 0
data_sites = []
for element in data:
    rho_element = Qobj(element)
    array_of_2s = list(np.full(i+4, 2))
    rho_element.dims = [array_of_2s,array_of_2s]

    j = 0
    element_sites = []
    while j < len(array_of_2s):
        site_j = ptrace(rho_element,j)
        site_j = site_j.full()
        element_sites.append(site_j)
        j += 1

    data_sites.append(element_sites)
    i += 1
