# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:55:49 2025

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from qutip import ptrace, Qobj, tensor, ket2dm, sigmaz, sigmax, sigmay, qeye, basis, sigmam,sigmap,identity,destroy,swap,rand_unitary,expect,num,rand_herm
from scipy.linalg import expm
from scipy.stats import unitary_group
import random
from math import sqrt,exp

"""
Parameters
"""
def run_simulation(g,G,beta,V_comp,N_qubit=4,N_levels=12,tau=0.1,time=15,omega=1):
    sim_step = int(time/tau)
 
    '''
    Base Operators
    '''
    A = tensor(*[identity(2) for i in range(N_qubit)],destroy(N_levels)) #lowering op
    A_dag = A.dag()
    
    #boson-boson interaction
    A_small = destroy(N_levels)
    
    sigma_plus_operators = []
    sigma_minus_operators = []
    
    i = 0
    for i in range(N_qubit):
        if i == 0:
            sm = tensor(destroy(2),*[identity(2) for j in range(N_qubit - 1)],identity(N_levels))
            sp = sm.dag()
            sigma_minus_operators.append(sm)
            sigma_plus_operators.append(sp)
        else:
            sm = tensor(*[identity(2) for j in range(i)],destroy(2),*[identity(2) for j in range(N_qubit - (1 + i))],identity(N_levels))
            sp = sm.dag()
            sigma_minus_operators.append(sm)
            sigma_plus_operators.append(sp)
            
    '''
    Useful functions
    '''
    
    def calc_fidelity(rho,sigma):
        return ((((rho.sqrtm() * sigma * rho.sqrtm())).sqrtm()).tr())**2
    
    '''
    ---Define Operators---
    '''
    
    def quantum_computer(N):
        '''
        Initialises our system given by s x s x s ... N times
        where each s is a qubit initialized in state 0
        '''
        return tensor(*[ket2dm(basis(2,0)) for i in range(N)])
    
    H_harm = omega*A.dag()*A #environment boson part, A is lowering operator
    
    #sys-env interaction hamiltonian to be determined during simulation
    
    # Boson - boson interaction
    H_anc_anc = G*(tensor(A_small,A_small.dag()) + tensor(A_small.dag(),A_small))
    U_anc_anc = (-1j * H_anc_anc * tau).expm() #Ancilla ancilla interaction
    
    '''
    Simulation Functions
    '''
           
    def create_env_mode(omega):
        energy_eigenstates = [omega * (n + 0.5) for n in range(N_levels)]
        Z = sum([exp(-1*beta*E_n) for E_n in energy_eigenstates])
        p_n = [exp(-1*beta*E_n)/Z for E_n in energy_eigenstates]
        print(p_n)
    
        env_ancilla_initial = sum(p_n[n]*ket2dm(basis(N_levels,n)) for n in range(N_levels))
        return env_ancilla_initial/env_ancilla_initial.tr()
    
    env_ancilla_initial = create_env_mode(omega)
    
    
    def simulate_markov_env(initial_state,measurement):
        combined_state = initial_state
        results = []
        rhoS = []
        i = 0
        for i in range(sim_step):
            #draw random qubit for env to interact with
            H_int = g*(sigma_minus_operators[i%N_qubit] * A_dag + sigma_plus_operators[i%N_qubit] * A)
            #Total interaction
            H_tot = H_harm + H_int
            U = (-1j * H_tot * tau).expm()
            
            new_state = V_comp * combined_state * V_comp.dag()
            new_state = U * new_state * U.dag()
            
            #for now avg expection of pauli z
            j = 0
            result = 0
            for j in range(N_qubit):
                meas = expect(measurement,new_state.ptrace(j))
                result += meas
        
            result = result/N_qubit
            results.append(result)
        
            system = new_state.ptrace(list(range(N_qubit)))
            rhoS.append(system)
            
            combined_state = tensor(system,env_ancilla_initial)
        
        return results,rhoS
    
    def simulate_no_noise_computer(initial_state,measurement):
        results = []
        rhoS = []
        fresh_sys = initial_state
        i = 0
        for i in range(sim_step):
            fresh_sys = V_comp * fresh_sys * V_comp.dag()
            #for now avg expection of pauli z
            j = 0
            result = 0
            for j in range(N_qubit):
                meas = expect(measurement,fresh_sys.ptrace(j))
                result += meas
            
            result = result/N_qubit
            results.append(result)
            
            rhoS.append(fresh_sys.ptrace(list(range(N_qubit))))
        
        return results,rhoS
    
    def simulate_nonmarkov_env(initial_state,measurement):
        combined_state = initial_state
        results = []
        rhoS = []
        i = 0
        for i in range(sim_step):
            H_int = g*(sigma_minus_operators[i%N_qubit] * A_dag + sigma_plus_operators[i%N_qubit] * A)
            
            #Total interaction
            H_tot = H_harm + H_int
            U = (-1j * H_tot * tau).expm()
            
            new_state = V_comp * combined_state * V_comp.dag()
            new_state = U * new_state * U.dag()
            
            #for now avg expection of pauli z
            j = 0
            result = 0
            for j in range(N_qubit):
                meas = expect(measurement,new_state.ptrace(j))
                result += meas
        
            result = result/N_qubit
            results.append(result)
        
            system = new_state.ptrace(list(range(N_qubit)))
            rhoS.append(system)
            
            old_anc = new_state.ptrace(N_qubit)
            combined_anc = tensor(old_anc,env_ancilla_initial)
            combined_anc = U_anc_anc * combined_anc * U_anc_anc.dag()
            new_anc = combined_anc.ptrace(1)
            
            combined_state = tensor(system,new_anc)
            combined_state = combined_state / combined_state.tr()
        
        return results,rhoS
    
    '''
    Run Simulation
    '''
    initial_state = tensor(quantum_computer(N_qubit),env_ancilla_initial)

    result_markov,rho_markov = simulate_markov_env(initial_state, ket2dm(basis(2,1)))
    result_nm,rho_nm = simulate_nonmarkov_env(initial_state, ket2dm(basis(2,1)))
    result_c,rho_c = simulate_no_noise_computer(initial_state, ket2dm(basis(2,1)))

    '''
    Plot
    '''

    i = 0
    result_fid_m = []
    result_fid_nm = []
    for i in range(len(rho_c)):
        fid_c_m = calc_fidelity(rho_markov[i], rho_c[i])
        fid_c_nm = calc_fidelity(rho_nm[i], rho_c[i])
        result_fid_m.append(fid_c_m)
        result_fid_nm.append(fid_c_nm)
    
    ratio = np.array(result_fid_nm)/np.array(result_fid_m)
    
    return result_fid_m,result_fid_nm,result_markov,result_nm,result_c,ratio