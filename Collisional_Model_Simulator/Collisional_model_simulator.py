# -*- coding: utf-8 -*-
"""
---Masters Project---
Non-Markovian Collisional Models for open quantum dynamics
@author: rithi

Last updated: 12/02/2025

This program can simulate system dynamics for some NMCM, 
for different interactions.

Simulates:
    - Basic Markovian CM
    - NMCM with initially correlated ancillas
    - NMCM with initially correlated ancillas and AA-collisions
Features:
    - Sequentially generated ancillas and efficient representation through MPS
    - Tunable and general system-ancilla and ancilla-ancilla interactions
    - Functions to check and debug environment

Uses a new proposed method with collisional model in the state vector
formulation of QM, which is quadratically more efficient than the density
operator formulation and yields identical system dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from qutip import ptrace, Qobj, tensor, ket2dm, sigmaz, sigmax, sigmay, qeye, basis
from scipy.linalg import expm

'''
---Global Parameters--
'''

#Useful
zs = basis(2, 0)  # |0⟩ in computational basis
os = basis(2, 1)  # |1⟩ in computational basis
# Pauli matrices
sx = np.array([[0, 1], [1, 0]]) / 2  # S^x = (1/2) sigma^x
sy = np.array([[0, -1j], [1j, 0]]) / 2  # S^y = (1/2) sigma^y
sz = np.array([[1, 0], [0, -1]]) / 2  # S^z = (1/2) sigma^z

len_states = 300 #defines the timesteps and size of environment
dt = 0.1 # timesteps
measurement = sigmay() #defines the observable we are measuring

# Define initial the system and environment states
sys_state = zs
env_state = np.sqrt(1 / 2) * (tensor(zs, zs) + tensor(os, os))  # Bell state
# Combined initial state [[2, 2], [1, 1]]
state = tensor(env_state,sys_state)
state.dims = [[2,2,2],[1,1,1]]

'''
---Sequential Generation Tensors---

Functions here form the operators that generate the next Ancilla states.
Assign a function to gen_operator ([4x4] matrix) by changing the variable
gen_operator.
'''

def basic_stochastic_process(p):
    '''
    Generates the next ancilla either 0 or 1 based on probability p
    p is probability to switch states.
    '''
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
    return tensor_qobj

# Assign the generation operator that generates the next ancilla state
gen_operator = basic_stochastic_process(0.3)


'''
---Utility Operators---
'''
def random_unitary(seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    # Generate a random complex matrix
    random_matrix = np.random.randn(4, 4) + 1j * np.random.randn(4,4)
    
    # Perform QR decomposition and normalize to get a unitary matrix
    q, _ = np.linalg.qr(random_matrix)
    return q

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
--- System-Ancilla Interactions ---

Functions here give the system and ancilla interaction.
Change the variable H_op ([4x4] Matrix) to change the interaction.
'''

def one_control_ex(theta):
    '''
    One qubit control on ancilla and a rotation on system
    '''
    return np.array([[0,0,0,0],
                     [0,0,0,0],
                     [0, 0, np.cos(theta), -np.sin(theta)],
                     [0, 0, np.sin(theta), np.cos(theta)]])

def two_control_ex(theta):
    '''
    Control on both 1 and 0 on ancilla and two rotations on system
    controlled by ancilla amplitudes.
    '''
    return np.array([[np.cos(theta) + 1j*np.sin(theta), 0, 0, 0],
                 [0, np.cos(theta) - 1j*np.sin(theta), 0, 0],
                 [0, 0, np.cos(theta), -np.sin(theta)],
                 [0, 0, np.sin(theta), np.cos(theta)]])

def heisenberg_exchange(J, t):
    """
    Computes the time evolution operator for the Heisenberg exchange interaction.
    """
    
    # Tensor products for two-qubit system
    Sx1Sx2 = np.kron(sx, sx)
    Sy1Sy2 = np.kron(sy, sy)
    Sz1Sz2 = np.kron(sz, sz)
    
    # Heisenberg Hamiltonian
    H = J * (Sx1Sx2 + Sy1Sy2 + Sz1Sz2)
    
    # Time evolution operator U = exp(-i H t)
    U = expm(-1j * H * t)
    
    return U

def ising_interaction(J, t, pauli1, pauli2):
    """
    Computes the time evolution operator for the Ising interaction.
    
    Parameters:
        J (float): Coupling strength.
        t (float): Evolution time.
    
    Returns:
        numpy.ndarray: 4x4 unitary evolution matrix.
    """
    
    # Tensor product for two-qubit system
    Sz1Sz2 = np.kron(pauli1, pauli2)
    
    # Ising Hamiltonian
    H = J * Sz1Sz2
    
    # Time evolution operator U = exp(-i H t)
    U = expm(-1j * H * t)
    
    return U

def general_evolution(interaction_hamiltonian,t):
    return expm(-1j * interaction_hamiltonian * t)

#Define the interaction between system and ancilla here
H_op = ising_interaction(1,dt,sx,sz)
H_op = Qobj(H_op)
H_op.dims = [[2,2],[2,2]]


'''
---Ancilla-Ancilla Interactions---

Functions here give the interactions between ancilla and ancilla.
Change variable AA_col to change the AA interaction.
'''

def partial_swap(theta):
    """
    Computes the partial swap matrix for a given swap angle θ.
    """
    I = np.eye(4)  # 4x4 Identity matrix for 2 qubits
    S = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]])  # Standard Swap Operator
    
    U_n = np.cos(theta) * I - 1j * np.sin(theta) * S  # Partial swap formula
    return U_n

AA_col = Qobj(partial_swap(np.pi/3))
AA_col.dims = [[2,2],[2,2]]


'''
--Models--
Various collisional models  built for this project.
'''

def build_sites(state,len_states):
    '''
    Testing function to check each site in an environment being generated.
    Builds environement state.
    '''
    i = 0
    sites = []
    while i < len_states:
        if i == 0:
            gen_operator_extend = tensor(gen_operator,qeye(2),qeye(2))  # Extend with identity operator
        else:
            gen_operator_extend = tensor(qeye(2**(i)),gen_operator,qeye(2),qeye(2))  # Extend with identity operator

        # Apply the extended operator
        state_next = gen_operator_extend * state
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
    '''
    Testing function to unpack sites from function build sites.
    Unpacks each site to be examined.
    '''
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
    '''
    Runs a memoryless CM where the ancilla is renewed at everystep.
    '''
    i = 0
    steps = np.arange(1,len_sites+1,step=1)
    sys = ket2dm(sys_state)
    operator = H_op
    operator_dag = H_op.dag()
    exp_data = []
    while i < len_sites:
        comb = tensor(Qobj(0.5*qeye(2)),sys)
        result = operator * comb * operator_dag
        sys = ptrace(result,1)
        exp_data.append((sys * measurement).tr())
        i += 1
    return exp_data,steps

def run_model_full_new(state,len_states,dt,measurement,H_op):
    '''
    Runs a non-markovian CM witht the full correlated environment state.
    Is very memory inefficient, used for benchmark and control.
    '''
    i = 0
    exp_data = []
    sites = []
    steps = np.arange(1,len_states+1,step=1)
    while i < len_states:
        if i == 0:
            gen_operator_extend = tensor(qeye(2),gen_operator,qeye(2))  # Extend with identity operator
            U_op_extend = tensor(qeye(2),qeye(2),H_op)
            swap_extend = tensor(qeye(2),qeye(2),swap_operator)
        else:
            gen_operator_extend = tensor(qeye(2),gen_operator,qeye(2),qeye(2**i))  # Extend with identity operator
            U_op_extend = tensor(qeye(2),qeye(2),H_op,qeye(2**i))
            swap_extend = tensor(qeye(2),qeye(2),swap_operator,qeye(2**i))
        # Apply the extended operator
        state_next = gen_operator_extend * state
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

def run_model_uncompute_AA(state,len_states,dt,measurement,H_op):
    '''
    Runs a non-markovian CM with initially correlated ancillas and
    ancilla-ancilla interactions. Saves memory via proposed method.
    '''
    i = 0
    exp_data = []
    sites = []
    steps = np.arange(1,len_states+1,step=1)
    while i < len_states:
        if i == 0:
            gen_operator_extend = tensor(qeye(2),gen_operator,qeye(2))  # Extend with identity operator
            U_op_extend = tensor(qeye(2),qeye(2),H_op)
            AA_extend = tensor(qeye(2),AA_col,qeye(2))
            swap_extend = tensor(qeye(2),qeye(2),swap_operator)
            
            state_next = gen_operator_extend * state
            state_next.dims = [[2,2,2,2],[1,1,1,1]]
            state_next = U_op_extend * state_next
            state_next = AA_extend * state_next
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
            gen_operator_extend = tensor(qeye(2),gen_operator,qeye(2),qeye(2**i))  # Extend with identity operator
            U_op_extend = tensor(qeye(2),qeye(2),H_op,qeye(2**i))
            AA_extend = tensor(qeye(2),AA_col,qeye(2),qeye(2**i))
            swap_extend = tensor(qeye(2),qeye(2),swap_operator,qeye(2**i))
            
            state_next = gen_operator_extend * state
            state_next.dims = [[2,2,2,2,2**i],[1,1,1,1,1]]
            state_next = U_op_extend * state_next
            state_next = AA_extend * state_next
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
            gen_operator_extend = tensor(qeye(2),gen_operator,qeye(2),qeye(2),qeye(2),qeye(2))
            U_op_extend = tensor(qeye(2),qeye(2),H_op,qeye(2),qeye(2),qeye(2))
            AA_extend = tensor(qeye(2),AA_col,qeye(2),qeye(2),qeye(2),qeye(2))
            swap_extend = tensor(qeye(2),qeye(2),swap_operator,qeye(2),qeye(2),qeye(2))

            state_next = gen_operator_extend * state
            state_next.dims = [[2,2,2,2,2,2,2],[1,1,1,1,1,1,1]]
            state_next = U_op_extend * state_next
            state_next = AA_extend * state_next
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

def run_model_uncompute(state,len_states,dt,measurement,H_op):
    '''
    Runs a non-markovian CM without AA-collisions.
    Saves memory by proposed method.
    '''
    i = 0
    exp_data = []
    sites = []
    steps = np.arange(1,len_states+1,step=1)
    while i < len_states:
        if i == 0:
            gen_operator_extend = tensor(qeye(2),gen_operator,qeye(2))  # Extend with identity operator
            U_op_extend = tensor(qeye(2),qeye(2),H_op)
            swap_extend = tensor(qeye(2),qeye(2),swap_operator)
            
            state_next = gen_operator_extend * state
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
            gen_operator_extend = tensor(qeye(2),gen_operator,qeye(2),qeye(2**i))  # Extend with identity operator
            U_op_extend = tensor(qeye(2),qeye(2),H_op,qeye(2**i))
            swap_extend = tensor(qeye(2),qeye(2),swap_operator,qeye(2**i))
            
            state_next = gen_operator_extend * state
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
            gen_operator_extend = tensor(qeye(2),gen_operator,qeye(2),qeye(2),qeye(2),qeye(2))
            U_op_extend = tensor(qeye(2),qeye(2),H_op,qeye(2),qeye(2),qeye(2))
            swap_extend = tensor(qeye(2),qeye(2),swap_operator,qeye(2),qeye(2),qeye(2))

            state_next = gen_operator_extend * state
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
---Simulate---

Environment to simulate CM and obtain results.
'''

exp_data_uncomp,steps_uncomp,sites_red = run_model_uncompute(state, len_states, dt, measurement, H_op)
exp_data_uncomp_AA,steps_uncomp_AA,sites_reduced_AA = run_model_uncompute_AA(state, len_states, dt, measurement, H_op)
data_uncor,steps_uncor = run_model_uncorr(sys_state, len_states, dt, measurement, H_op)

'''
---Plot---
'''

# Create the figure and axis objects
plt.rc('font', family='Times New Roman')
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the data
ax.plot(steps_uncomp, exp_data_uncomp, label="Expectation-z no AA", color="blue", linewidth=2)
ax.plot(steps_uncomp_AA, exp_data_uncomp_AA, label="Expectation-z with AA collision", color="red", linewidth=2, linestyle='-.')
ax.plot(steps_uncor, data_uncor, label="Expectation-z uncorrelated", color="black", linewidth=2, linestyle='--')

# Add title and labels with proper font size
ax.set_title("Evolution of system over N collisions", fontsize=20, pad=15)
ax.set_xlabel("Steps (collision with N-th env state)", fontsize=16)
ax.set_ylabel("Expectation value", fontsize=16)

# Add a grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend with a frame

# Get existing handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

ax.legend(handles, labels, loc='upper right', fontsize=16)


# Increase the ticks font size for better visibility
ax.tick_params(axis='both', which='major', labelsize=15)

# Set tighter layout for padding
plt.tight_layout(pad=2)

# Show the plot
plt.show()