# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 20:59:00 2025

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp,factorial,log2
from scipy.integrate import quad
from numpy import kron,tensordot
from numpy import identity
from qutip import destroy, sigmax, sigmaz, basis,expect, sigmay,sigmap,sigmam
from scipy.linalg import expm
from qutip import tensor, Qobj, ket2dm, ptrace, qeye, create
from scipy.stats import unitary_group

#Params
omega_c = 1
s = 0
alpha = 0.0001
eta = 1
epsilon = 1
env_sim_step = 200

N_levels = 6
len_chain = 6
dt = 0.1
T = 30
steps = int(T/dt)

def spectral_dens(omega):
    return 2*np.pi*alpha*(omega_c**(1-s))*(omega**s)*exp((-1*omega)/omega_c)

def omega_n(n):
    return omega_c*(2*n + 1 + s)

def t_n(n):
    return omega_c*sqrt((n+1)*(n+s+1))

eta0, err = quad(spectral_dens,0,np.inf)
c0 = np.sqrt(eta0)
print(c0)

def normalize(state):
    """Normalize the state vector."""
    norm = np.linalg.norm(state)
    if norm != 0:
        state /= norm
    return state

def ten_mul(t1,t2):
    #(A_ij x B_kl)(C_ab x D_cd)
    # (s_ja A_ijC_ab x s_lc B_klD_cd)
    # ({AC}_ib x {BD}_kd)
    tensor_mul = np.tensordot(t1, t2,axes=([1,3],[0,2])) 
    # s_jalc AB_ijkl CD_abcd
    # {ABCD}_ikbd (i->0,k->2,b->1,d->3)
    # == ({AC}_ib x {BD}_kd)
    tensor_mul = np.transpose(tensor_mul,axes=[0,2,1,3])
    return tensor_mul

def exp_mat(A,n_terms=100):
    result = tensordot(identity(N_levels,dtype=complex),identity(N_levels,dtype=complex),axes=0)  # Initialize with the identity matrix
    term = tensordot(identity(N_levels,dtype=complex),identity(N_levels,dtype=complex),axes=0)   # First term (A^0 / 0!)

    for n in range(1, n_terms):
        term = (ten_mul(term,A))/n  # Compute A^n * x^n / n!
        result += term
    return result

"""
Hamiltonians
"""

b = destroy(N_levels)
b_dag = b.dag()

sz = sigmaz()
sx = sigmax()

b = b.full()
b_dag = b_dag.full()

sz = sz.full()
sx = sx.full()

H_sys = 0.5*eta*sx + 0.5*epsilon*sz

H_sites = []
for n in range(len_chain):
    H_sites.append(omega_n(n)*tensordot(b_dag@b,identity(N_levels),axes=0) + t_n(n)*tensordot(b,b_dag,axes=0) + t_n(n)*tensordot(b_dag,b,axes=0))

def H_check(n):
    return omega_n(n)*b_dag@b

"""
Unitaries
"""

U_sys = expm(-1j*H_sys*dt) #unitary

#unitary
U_int = np.cos(dt*0.5*c0)*tensordot(identity(2),identity(N_levels),axes=0) - 1j*np.sin(dt*0.5*c0)*tensordot(sz,b+b_dag,axes=0)

#unitary
U_sites = []
for H in H_sites:
    U_sites.append(exp_mat(-1j*H*dt))

#non_unitary
U_sites_im = []
for H in H_sites:
    U_sites_im.append(exp_mat(-1*H*dt))

"""
Env evo
"""
env_energy_vals = []
env = []
anc = basis(N_levels,0).full()
for _ in range(len_chain):
    env.append(anc)

sys = basis(2,0).full()

comb = env[0]
i = 1
while i < len_chain:
    comb = tensordot(comb,env[i],axes=(0))
    i += 1
comb = comb.flatten()
comb_dag = np.conj(comb).T
norm = comb @ comb_dag
print(norm)

#Start off to get to general ground state
for i in range(env_sim_step):
    if i == 0: #first loop
        #EVEN
        for n in range(len_chain):
            if n%2 == 0:
                op = U_sites[n]
                state = tensordot(env[n],env[n+1],axes=0)
                temp = tensordot(op,state,axes=([1,3], [0,2]))
                temp = temp[:,:,0,0]

                U,S,Vh = np.linalg.svd(temp)
                L = U@np.diag(np.sqrt(S))/sum(S)
                R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                env[n] = L
                env[n+1] = R

        #ODD
        for n in range(len_chain - 1):
            if n%2 != 0:
                op = U_sites[n]
                state = tensordot(env[n],env[n+1],axes=0)
                temp = tensordot(op,state,axes=([1,3],[1,2]))
                temp = np.transpose(temp, axes=[2,0,1,3]) #to correct indices
                
                temp = temp.reshape(N_levels*N_levels,N_levels*N_levels)
                U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                L = U@np.diag(np.sqrt(S))/sum(S)
                L = L.reshape(N_levels,N_levels,N_levels*N_levels)
                R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                R = R.reshape(N_levels*N_levels,N_levels,N_levels)
                env[n] = L
                env[n+1] = R

    #After 1 loop
    else:
        #EVEN
        for n in range(len_chain):
            if n%2 == 0:
                op = U_sites[n]
                state = tensordot(env[n],env[n+1],axes=(-1,0))
                if n == 0: #right
                    temp = tensordot(op,state,axes=([1,3],[0,1]))
                    shape = temp.shape
                    temp = temp.reshape(shape[0],shape[1]*shape[2])
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R
                elif n == (len_chain - 2): #left
                    temp = tensordot(op,state,axes=([1,3],[1,2]))
                    temp = np.transpose(temp, axes=[2,0,1])
                    shape = temp.shape
                    temp = temp.reshape(shape[0]*shape[1],shape[2])
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels)
                    env[n] = L
                    env[n+1] = R
                else:
                    temp = tensordot(op,state,axes=([1,3],[1,2]))
                    temp = np.transpose(temp, axes=[2,0,1,3])
                    shape = temp.shape
                    temp = temp.reshape(shape[0]*shape[1],shape[2]*shape[3])
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R

        #ODD
        for n in range(len_chain - 1):
            if n%2 != 0: #always middle
                op = U_sites[n]
                state = tensordot(env[n],env[n+1],axes=(-1,0))
                temp = tensordot(op,state,axes=([1,3],[1,2]))
                temp = np.transpose(temp, axes=[2,0,1,3])
                shape = temp.shape
                temp = temp.reshape(shape[0]*shape[1],shape[2]*shape[3])
                if n == 1:
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R
                elif n == len_chain - 3:
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R
                else:   
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R

    comb = env[0]
    i = 1
    while i < len_chain:
        comb = tensordot(comb,env[i],axes=(-1,0))
        i += 1
    comb = comb.flatten()
    comb_dag = np.conj(comb).T
    norm = comb @ comb_dag
    norm = norm*len_chain

'''
Evolve
'''
result = []
systems = []
entropy_over_time = []
for i in range(steps):
    if i == 0:
        #SYS
        sys = np.matmul(U_sys,sys)

        #INT
        comb = tensordot(sys,env[0],axes=0)
        comb = tensordot(U_int, comb,axes=([1,3],[0,2]))
        comb = comb.reshape(2,N_levels*N_levels)
        U,S,Vh = np.linalg.svd(comb,full_matrices=False)
        lambda_sq = sum(x**2 for x in S)
        shannon_entropy = -1*(lambda_sq*log2(lambda_sq))
        entropy_over_time.append(shannon_entropy)
        L = U@np.diag(np.sqrt(S))/sum(S)
        R = (np.diag(np.sqrt(S))@Vh)/sum(S)
        R = R.reshape(2,N_levels,N_levels)
        sys = L
        env[0] = R

        #Even
        for n in range(len_chain):
            if n%2 == 0:
                op = U_sites[n]
                state = tensordot(env[n],env[n+1],axes=(-1,0))
                if n == 0: #right
                    temp = tensordot(op,state,axes=([1,3],[1,2]))
                    temp = np.transpose(temp, axes=[2,0,1,3])
                    shape = temp.shape
                    temp = temp.reshape(shape[0]*shape[1],shape[2]*shape[3])
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(2,N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R
                elif n == (len_chain - 2): #left
                    temp = tensordot(op,state,axes=([1,3],[1,2]))
                    temp = np.transpose(temp, axes=[2,0,1])
                    shape = temp.shape
                    temp = temp.reshape(shape[0]*shape[1],shape[2])
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels)
                    env[n] = L
                    env[n+1] = R
                else:
                    temp = tensordot(op,state,axes=([1,3],[1,2]))
                    temp = np.transpose(temp, axes=[2,0,1,3])
                    shape = temp.shape
                    temp = temp.reshape(shape[0]*shape[1],shape[2]*shape[3])
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R
        #ODD
        for n in range(len_chain - 1):
            if n%2 != 0: #always middle
                op = U_sites[n]
                state = tensordot(env[n],env[n+1],axes=(-1,0))
                temp = tensordot(op,state,axes=([1,3],[1,2]))
                temp = np.transpose(temp, axes=[2,0,1,3])
                shape = temp.shape
                temp = temp.reshape(shape[0]*shape[1],shape[2]*shape[3])
                if n == 1:
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R
                elif n == len_chain - 3:
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R
                else:   
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R

    else:
        #sys
        sys = tensordot(U_sys,sys,axes=(1,0))

        #INT
        comb = tensordot(sys,env[0],axes=(-1,0))
        comb = tensordot(U_int, comb,axes=([1,3],[0,1]))
        shape = comb.shape
        comb = comb.reshape(shape[0],shape[1]*shape[2])
        U,S,Vh = np.linalg.svd(comb,full_matrices=False)
        lambda_sq = sum(x**2 for x in S)
        shannon_entropy = -1*(lambda_sq*log2(lambda_sq))
        entropy_over_time.append(shannon_entropy)
        L = U@np.diag(np.sqrt(S))/sum(S)
        L = L.reshape(2,2)
        R = (np.diag(np.sqrt(S))@Vh)/sum(S)
        R = R.reshape(2,N_levels,int((R.shape[1])/N_levels))
        sys = L
        env[0] = R

        for n in range(len_chain):
            if n%2 == 0:
                op = U_sites[n]
                state = tensordot(env[n],env[n+1],axes=(-1,0))
                if n == 0: #right
                    temp = tensordot(op,state,axes=([1,3],[1,2]))
                    temp = np.transpose(temp, axes=[2,0,1,3])
                    shape = temp.shape
                    temp = temp.reshape(shape[0]*shape[1],shape[2]*shape[3])
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(2,N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R
                elif n == (len_chain - 2): #left
                    temp = tensordot(op,state,axes=([1,3],[1,2]))
                    temp = np.transpose(temp, axes=[2,0,1])
                    shape = temp.shape
                    temp = temp.reshape(shape[0]*shape[1],shape[2])
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels)
                    env[n] = L
                    env[n+1] = R
                else:
                    temp = tensordot(op,state,axes=([1,3],[1,2]))
                    temp = np.transpose(temp, axes=[2,0,1,3])
                    shape = temp.shape
                    temp = temp.reshape(shape[0]*shape[1],shape[2]*shape[3])
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R
        #ODD
        for n in range(len_chain - 1):
            if n%2 != 0: #always middle
                op = U_sites[n]
                state = tensordot(env[n],env[n+1],axes=(-1,0))
                temp = tensordot(op,state,axes=([1,3],[1,2]))
                temp = np.transpose(temp, axes=[2,0,1,3])
                shape = temp.shape
                temp = temp.reshape(shape[0]*shape[1],shape[2]*shape[3])
                if n == 1:
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R
                elif n == len_chain - 3:
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R
                else:   
                    U,S,Vh = np.linalg.svd(temp,full_matrices=False)
                    L = U@np.diag(np.sqrt(S))/sum(S)
                    R = (np.diag(np.sqrt(S))@Vh)/sum(S)
                    L = L.reshape(shape[0],N_levels,(L.shape)[1])
                    R = R.reshape((R.shape)[0],N_levels,int((R.shape[1])/N_levels))
                    env[n] = L
                    env[n+1] = R

    sys_dag = np.conjugate(sys).T
    rho = sys @ sys_dag
    result.append(np.trace(rho @ sz))
    systems.append(rho)

result_markov = []
anc = ket2dm(basis(N_levels,0))

sys = ket2dm(basis(2,0))
U_int_qutip = np.cos(dt*0.5*c0)*tensor(qeye(2),qeye(N_levels)) - 1j*np.sin(dt*0.5*c0)*tensor(sigmaz(),destroy(N_levels) + create(N_levels))
U_int_qutip_dag = U_int_qutip.dag()
U_sys_qutip = Qobj(U_sys)
U_sys_qutip_dag = U_sys_qutip.dag()
for i in range(steps):
    sys = U_sys_qutip * sys * U_sys_qutip_dag
    comb = tensor(sys,anc)
    comb = U_int_qutip * comb * U_int_qutip_dag
    sys = comb.ptrace(0)
    result_markov.append(expect(sigmaz(),sys))

test_sys = basis(2,0).full()
things = []
for _ in range(steps):
    test_sys = U_sys@test_sys
    resulting = np.conj(test_sys).T @ sigmaz().full() @ test_sys
    things.append(resulting[0])

plt.figure(figsize=(10,8))
plt.plot(np.arange(0,T,step=dt),result_markov,'-.',label='markov')
plt.plot(np.arange(0,T,step=dt),result,label='TEDOPA')
plt.plot(np.arange(0,T,step=dt),things,'--',label='No env')
plt.legend()
plt.show()
print(result[-1])

plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(10,10))
plt.plot(np.arange(0,T,step=dt),result,label='TEDOPA')
plt.plot(np.arange(0,T,step=dt),result_markov,'-.',label='Markov')
plt.title("Expectation of sz of spin", fontsize=18)
plt.ylabel("Ex(sz)", fontsize=18)
plt.xlabel("Time t", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.grid()
plt.legend(fontsize=15)
#plt.savefig("exp_sz_tedopa.jpg",dpi=200)
plt.show()


plt.figure(figsize=(10,10))
plt.plot(np.arange(0,T,step=dt),entropy_over_time,'-',label='Entropy over time')
plt.title("Entanglement entropy between S and E over time", fontsize=18)
plt.ylabel("Entanglement entropy", fontsize=18)
plt.xlabel("Time t", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.grid()
plt.legend(fontsize=15)
#plt.savefig("entanglement_ent.jpg",dpi=200)
plt.show()


