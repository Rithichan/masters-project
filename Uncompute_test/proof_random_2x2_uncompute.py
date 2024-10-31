# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:45:35 2024

@author: rithi
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import unitary_group

def partial_trace(rho, dims, axis=0):
    """
    Takes partial trace over the subsystem defined by 'axis'
    rho: a matrix
    dims: a list containing the dimension of each subsystem
    axis: the index of the subsytem to be traced out
    (We assume that each subsystem is square)
    """
    dims_ = np.array(dims)
    # Reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # Each subsystem gets one index for its row and another one for its column
    reshaped_rho = rho.reshape(np.concatenate((dims_, dims_), axis=None))

    # Move the subsystems to be traced towards the end
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims)+axis-1, -1)

    # Trace over the very last row and column indices
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    # traced_out_rho is still in the shape of a tensor
    # Reshape back to a matrix
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])

# Reinitialize system and environment states
random_sys_state = np.random.rand(2)
random_env_state = np.random.rand(2)

# Construct density matrices for each state
dens_sys = np.outer(random_sys_state, random_sys_state)
dens_env = np.outer(random_env_state, random_env_state)

# Generate a random unitary and apply it to both system and environment
unitary_matrix = unitary_group.rvs(2)
operator = np.kron(unitary_matrix, unitary_matrix)
operator_dagger = np.conjugate(operator).T

# Form the total density matrix in the density matrix formalism
total_rho = np.kron(dens_sys, dens_env)
total_rho = np.matmul(operator, np.matmul(total_rho, operator_dagger))

# Calculate the density matrix using the pure state formalism
# Construct a total state by applying the unitary to the initial system-environment vector
total_vector = np.kron(random_sys_state, random_env_state)
transformed_vector = np.matmul(operator, total_vector)

transformed_vector = transformed_vector.reshape(2,2)
U,S,Vh = np.linalg.svd(transformed_vector)

U_dagger = np.conjugate(U).T

transformed_vector = np.matmul(np.matmul(U_dagger,U),np.matmul(np.diag(S),Vh))
transformed_vector = transformed_vector.reshape(4)

density_matrix_from_vector = np.outer(transformed_vector, np.conjugate(transformed_vector))

# Check if they are close
are_equivalent = np.allclose(density_matrix_from_vector, total_rho)
print(are_equivalent)

sys_1 = partial_trace(density_matrix_from_vector, [2,2])
sys_2 = partial_trace(total_rho, [2,2])

print(np.allclose(sys_1,sys_2))

