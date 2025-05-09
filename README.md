# Masters project: Collisional model methods for modelling memoryful open quantum dynamics

## Semester1: Modelling Non-Markovian Collisional models in state vector formulation of QM

1. Unitary Operation
- Shows some unitary evolution of the combined system over time.
- Comparison between joint system evolution vs separating system and operating.

2. Sequential Generation of entangled multiqubit states with unitary operation
- Computational sequential generation of GHZ state while operating with a unitary operator at every step.
- Comparison with direct operation on joint state and system to show how the method works

3. Uncompute
- Using the adjoint of the left/right unitary from SVD, can reduce the size of the tensor and reduce it to an identity-type object, while leaving the forward states intact.

Code:
- Test Models
- Collisional_model_simulator.py does a non-Markovian collisional model with sequential generation of correlated ancillas.

## Semester2: Simulation of the spin-boson model

1. Spin-boson interaction with TEDOPA
- Simulates the dynamics between a two-level system with a bosonic reservoir
- Transforming bosonic reservoir to discrete chain of harmonic modes via TEDOPA
- Breaking up interactions via trotterization

2. Jaynes–Cummings model type interaction for the simulation of thermal noise on a quantum computer
- Simulates how a theoretical quantum computer loses fidelity due to interactions with a thermal bath.

