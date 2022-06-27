import sys
import warnings
sys.path.append('../')
warnings.filterwarnings('ignore')


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.auto import tqdm
from os.path import isfile, exists
from os import mkdir

from qiskit import IBMQ, Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, execute
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.ignis.verification import get_ghz_simple
from qiskit.compiler import transpile
from qiskit.tools.monitor import job_monitor
from qiskit.opflow.state_fns import CircuitStateFn
from qiskit.providers.aer.extensions.snapshot_statevector import *
from qiskit.test.mock import FakeMontreal

from utils import get_backend_connectivity, molecules, Label2Chain, from_string_to_numbers, load_grouping_data, current_time, load_data_IBMQ, send_ibmq_parallel
from GroupingAlgorithm import groupingWithOrder, TPBgrouping
from HEEM_VQE_Functions import measure_circuit_factor, probability2expected_binary, post_process_results, probability2expected_parallel, post_process_results_2, post_process_results_3


NUM_SHOTS = 2 ** 14
# NUM_SHOTS = 2 ** 16

def compute_energy_sparse(Groups, Measurements, method, n_chunks=10, layout=None, return_energy_temp=False, noisy=True, shots=NUM_SHOTS):

    circuits = [measure_circuit_factor(measurement, n_qubits, measure_all=False).compose(state_0, front=True) for measurement
                 in Measurements]

    if noisy:
        kwards_run = {'coupling_map': coupling_map, 'noise_model': noise_model, 'basis_gates': basis_gates}
    else:
        kwards_run = {}

    if shots == -1:
        shots = 20000 * len(Groups)  # Max allowed shots

    kwards_run['shots'] = shots // len(Groups)

    if layout is not None:
        kwards_run['initial_layout'] = layout[::-1]

    jobs = send_ibmq_parallel(backed_calculations, n_chunks, circuits, job_tag=[molecule_name, method], verbose=False,
                              progress_bar=True, kwards_run=kwards_run)

    counts = []

    for job in jobs:
        counts += job.result().get_counts()

    energy_temp = []
    pbar = tqdm(range(len(counts)), desc='Computing energy')
    for j in pbar:
        counts_indices, counts_values = post_process_results_3(counts[j])

        diagonals, factors = probability2expected_binary(coeffs, labels, [Groups[j]], [Measurements[j]], shift=False)
        diagonals = [(~diagonal * 2 - 1).astype('int8') for diagonal in diagonals[0][:, counts_indices]]

        energy_temp.append(np.sum((diagonals * np.array(factors[0])[:, None]) * counts_values[None, :]) / kwards_run['shots'])

    returns = [jobs, sum(energy_temp)]

    if return_energy_temp:
        returns.append(energy_temp)

    return returns

# 1: Luciano, 2: Guillermo, 3: Gabriel, 4: Fran, 5: David
#index = None
#if index is None:
#    raise Exception('Use you index!')


def check_folder_create(path):
    if exists(path):
        pass
    else:
        mkdir(path)

def save_data(energy, method):
    subdics = ['data', 'energies', initial_state, 'partial']

    folder = '../'
    for dic in subdics:
        folder += dic + '/'
        check_folder_create(folder)

    file = folder + molecule_name + '_' + method +'_' + str(index) + '.npy'
    if isfile(file):
        data = np.load(file)
    else:
        data = np.array([])

    data = np.append(data, energy)
    np.save(file, data)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    IBMQ.load_account()

provider_main = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
# provider_CSIC = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')

# name_backend = 'ibmq_montreal'
# backend = provider_CSIC.get_backend(name_backend)
backend = FakeMontreal()
WC_device = get_backend_connectivity(backend)

G_device = nx.Graph()
G_device.add_edges_from(WC_device)

backend_hpc = provider_main.get_backend('ibmq_qasm_simulator')
backed_simulation = provider_main.get_backend('simulator_statevector')
simulator = Aer.get_backend('aer_simulator')  # Backend for simulation

device = QasmSimulator.from_backend(backend)
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)
# noise_model = None
basis_gates = noise_model.basis_gates if noise_model is not None else None

backed_calculations = backend_hpc

shots_TPB = {'H2': 300, 'LiH': 300, 'BeH2': 300, 'H2O': 300, 'CH4': 300,
             'C2H2': 300, 'CH3OH': 300, 'C2H6': 100}

shots_EM = {'H2': 300, 'LiH': 300, 'BeH2': 300, 'H2O': 300, 'CH4': 300,
            'C2H2': 100, 'CH3OH': 50, 'C2H6': 25}

shots_HEEM = {'H2': 300, 'LiH': 300, 'BeH2': 300, 'H2O': 300, 'CH4': 300,
              'C2H2': 300, 'CH3OH': 300, 'C2H6': 50}


molecule_name = 'H2'
initial_state = '0xN'

try:
    qubit_op = molecules(molecule_name)
    paulis, coeffs, labels = Label2Chain(qubit_op)
except AttributeError:
    paulis, coeffs, labels = np.load('../data/molecules_qubitop_list.npy', allow_pickle=True).item()[molecule_name]

n_qubits = len(paulis[0])
state_0 = QuantumCircuit(n_qubits)
# state_0 = state_0.compose(get_ghz_simple(n_qubits, measure=False))  # Initialized in the GHZ state

print(f'{len(paulis)} total Pauli strings')
print(f'{n_qubits} qubits')





