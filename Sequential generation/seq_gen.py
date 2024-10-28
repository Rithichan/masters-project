# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:01:46 2024

@author: rithi
"""

import numpy as np

a = np.array([0,0,1])
b1 = np.array([0,1,0])
b2 = np.array([1,0,0])

zero_state = np.array([0,1])
one_state = np.array([1,0])

def small(num):
    if num < 0.0000001:
        return 0
    else:
        return num

def map_AB(state):
    mapA = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
    resA = np.matmul(mapA,state)
    resA[0] = small(resA[0])
    resA[1] = small(resA[1])
    resA[1] = small(resA[1])
    l1 = small(resA[0])
    l2 = small(resA[1])
    l3 = small(resA[2])
    state_B = l1*one_state + l2*zero_state + l3*zero_state
    return [resA,state_B]
    
    
def U_mkl(m,k,l,phi,theta):
    return (np.cos(theta)*np.outer(k,k)) + (np.cos(theta)*np.outer(l,l)) + (np.exp(1j*phi)*np.sin(theta)*np.outer(k,l)) - (np.exp(-1j*phi)*np.sin(theta)*np.outer(l,k)) + np.outer(m,m)

phi_initial = b2

n = 8

phi_A = phi_initial
for i in range(n):
    if i == 0:
        phi_AB = map_AB(np.matmul(U_mkl(b1,a,b2,0,np.pi/6),phi_A))
        phi_A = phi_AB[0]
        phi_B = phi_AB[1]
    if i == n-1:
        phi_AB = map_AB(np.matmul(U_mkl(b1, a, b2, 0, np.pi/2),np.matmul(U_mkl(b2,a,b1,0,np.pi/2),phi_A)))
        phi_A = phi_AB[0]
        phi_B = phi_AB[1]
    else:
        phi_AB = map_AB(np.matmul(U_mkl(b2,a,b1,0,np.pi/2),phi_A))
        phi_A = phi_AB[0]
        phi_B = phi_AB[1]
    if i == 0:
        state = phi_B
    else:
        state = np.kron(state,phi_B)
