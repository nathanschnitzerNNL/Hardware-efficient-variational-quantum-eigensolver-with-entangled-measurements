import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from VQE import VQE
from GroupingAlgorithm import *
from utils import get_backend_connectivity
# Importing standard Qiskit libraries
from qiskit import IBMQ, QuantumCircuit, Aer, qasm
from qiskit.providers.aer import AerSimulator, QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.circuit.library import EfficientSU2
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
# from qiskit_nature.drivers import PyQuanteDriver
from qiskit_nature.drivers import Molecule#, PySCFDriver
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import NumPyMinimumEigensolverFactory
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber
from qiskit.opflow.primitive_ops import Z2Symmetries
from qiskit.opflow import converters
from qiskit.algorithms.optimizers import SPSA
from qiskit.test.mock import FakeMontreal, FakeBrooklyn
from IPython.display import display, clear_output

IBMQ.load_account()
#provider      = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
# backend_exp   = provider.get_backend('ibmq_paris')
backend_exp   = provider.get_backend('ibm_nairobi')
backend = FakeMontreal()
#backend = FakeBrooklyn()
#backend_exp   = provider.get_backend('ibm_oslo')
#WC_exp        = backend_exp.configuration().coupling_map
WC_exp = get_backend_connectivity(backend)
NUM_SHOTS = 2**13  # Number of shots for each circuit

#quantum_instance = QuantumInstance( backend_exp, shots = NUM_SHOTS )

backend_hpc = provider.get_backend('ibmq_qasm_simulator')
backed_simulation = provider.get_backend('simulator_statevector')
simulator = Aer.get_backend('aer_simulator')  # Backend for simulation

device = QasmSimulator.from_backend(backend)
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)
basis_gates = noise_model.basis_gates

backed_calculations = simulator
quantum_instance = QuantumInstance(backend=backed_calculations, coupling_map=coupling_map, noise_model=noise_model, basis_gates=basis_gates, shots=NUM_SHOTS)

molecule = 'Li 0.0 0.0 0.0; H 0.0 0.0 1.5474'
#driver = PySCFDriver(molecule)
driver = PySCFDriver(atom=molecule)
qmolecule = driver.run()
freezeCoreTransfomer = FreezeCoreTransformer( freeze_core=True, remove_orbitals= [3,4] )
problem = ElectronicStructureProblem(driver, transformers=[freezeCoreTransfomer])

# Generate the second-quantized operators
second_q_ops = problem.second_q_ops()

# Hamiltonian
main_op = second_q_ops[0]

# Setup the mapper and qubit converter
mapper_type = 'ParityMapper'
mapper = ParityMapper()

converter = QubitConverter( mapper=mapper, two_qubit_reduction=True, z2symmetry_reduction=[1,1],) #1] 
particle_number = qmolecule.get_property(ParticleNumber)
# The fermionic operators are mapped to qubit operators
#num_particles = (problem.molecule_data_transformed.num_alpha, problem.molecule_data_transformed.num_beta)
num_particles = particle_number.num_particles
num_spin_orbitals = problem.num_spin_orbitals

qubit_op = converter.convert(main_op, num_particles=num_particles)

num_qubits = qubit_op.num_qubits
WC = list(range(num_qubits))
WC = list(permutations(list(range(num_qubits)),2))

init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
print(num_spin_orbitals)
print(num_particles)
print( num_qubits )
print( qubit_op )


entangled_layer = []
for qbs in WC_exp:
    if qbs[0] < qbs[1] and qbs[1] < num_qubits:
        entangled_layer.append(qbs)

ansatz = init_state.compose(EfficientSU2(num_qubits, ['ry', 'rz'], entanglement=entangled_layer, reps=1))
print("Ansatz qubits: {}".format(ansatz.num_qubits))
def callback(evals, params):
    display("{}, {}".format(len(evaluations), evals))
    clear_output(wait=True)
    parameters.append(params)
    evaluations.append(evals)

parameters = []
evaluations = []

optimizer = SPSA(maxiter=250, last_avg=1)

num_var = ansatz.num_parameters
pars = [0.01] * num_var

solver = VQE(ansatz, optimizer, pars, grouping='Entangled', connectivity=WC_exp, callback=None, quantum_instance=quantum_instance)
results = solver.compute_minimum_eigenvalue(qubit_op)
print("Ground State Energy via HEEM VQE: {}".format(results.eigenvalue))

