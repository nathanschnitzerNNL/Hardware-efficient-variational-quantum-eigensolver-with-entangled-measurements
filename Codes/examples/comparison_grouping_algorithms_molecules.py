import sys

sys.path.append('../')

from utils import save_object, get_backend_connectivity, n_groups_shuffle, \
	molecules, Label2Chain
import numpy as np
import getopt
import os
from qiskit import IBMQ
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm
import networkx as nx
from time import time

# ------------  Default parameters calculation  --------------------
clear_start = False

molecule_name = 'LiH'
n_jobs = -1
time_save = 60  # (min)
total_time = 48 * 60  # (min)
N_test = 100

name_backend = 'ibmq_montreal'
backend_parallel = 'multiprocessing'
file_name = 'comparison_grouping_algorithms_molecules'

# ---------------------------------------------------------
message_help = file_name + '.py -m <molecule ({})> -j <#JOBS ({})> -t <time save (min) ({})> ' \
                           '-T <time total (min) ({})> -N <# tests ({})>, -c <clear start ({})>'.format(molecule_name,
                                                                                                        n_jobs,
                                                                                                        time_save,
                                                                                                        total_time,
                                                                                                        N_test,
                                                                                                        clear_start)

try:
	argv = sys.argv[1:]
	opts, args = getopt.getopt(argv, "hm:j:t:T:N:c:")
except getopt.GetoptError:
	print(message_help)
	sys.exit(2)

for opt, arg in opts:
	if opt == '-h':
		print(message_help)
		sys.exit(2)
	elif opt == '-m':
		molecule_name = str(arg)
	elif opt == '-j':
		n_jobs = int(arg)
	elif opt == '-t':
		time_save = int(arg)
	elif opt == '-T':
		total_time = int(arg)
	elif opt == '-N':
		N_test = int(arg)
	elif opt == '-c':
		clear_start = bool(arg)

if n_jobs == -1:
	n_jobs = os.cpu_count()

N_test = int(np.ceil(N_test / n_jobs) * n_jobs)

# Start calculation
if __name__ == '__main__':
	qubit_op = molecules(molecule_name)
	n_qubits = qubit_op.num_qubits
	paulis, _, _ = Label2Chain(qubit_op)
	print(
		'Grouping the {} molecule, with {} Pauli strings of {} qubits.'.format(molecule_name, len(qubit_op), n_qubits))

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		IBMQ.load_account()

	provider = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')
	backend = provider.get_backend(name_backend)
	WC_device = get_backend_connectivity(backend)

	G_device = nx.Graph()
	G_device.add_nodes_from(range(n_qubits))
	G_device.add_edges_from(WC_device)

	labels = ['naive', 'order_disconnected', 'order_connected']

	if clear_start:
		try:
			os.remove('../data/' + file_name + '.npy')
		except OSError:
			pass

	start = time()
	pbar = tqdm(range(N_test), desc='Test grouping', file=sys.stdout, ncols=90)
	results = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
		delayed(n_groups_shuffle)(paulis, G_device, None, order=False) for _ in pbar)
	total_time_test = time() - start

	average_time = total_time_test / N_test / 60 * 3  # (min)
	batch_size = int(np.ceil(int(time_save / average_time) / n_jobs) * n_jobs)
	N_total = int(np.ceil(int(total_time / average_time) / n_jobs) * n_jobs)
	N_batches = int(np.ceil(N_total / batch_size))
	print('There will be a total of {} shots in {} batches'.format(N_total, N_batches))

	pbar1 = tqdm(range(N_batches), desc='Computing batches', ncols=90, file=sys.stdout)
	for _ in pbar1:
		n_groups = {}
		times = {}

		start = time()
		pbar = tqdm(range(batch_size), desc='  Grouping', ncols=90, file=sys.stdout, leave=False)
		results = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
			delayed(n_groups_shuffle)(paulis, G_device, None, order=False) for _ in pbar)
		n_groups['naive'] = [result[0] for result in results]
		times['naive'] = time() - start

		start = time()
		pbar = tqdm(range(batch_size), desc='  Grouping + order', ncols=90, file=sys.stdout, leave=False)
		results = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
			delayed(n_groups_shuffle)(paulis, G_device, None, order=True, connected=False) for _ in pbar)
		n_groups['order_disconnected'] = [result[0] for result in results]
		times['order_disconnected'] = time() - start

		start = time()
		pbar = tqdm(range(batch_size), desc='  Grouping + order + connected ', ncols=90, file=sys.stdout, leave=False)
		results = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
			delayed(n_groups_shuffle)(paulis, G_device, None, order=True, connected=True) for _ in pbar)
		n_groups['order_connected'] = [result[0] for result in results]
		times['order_connected'] = time() - start

		try:
			data = np.load('../data/' + file_name + '.npy', allow_pickle=True).item()
		except OSError:
			data = {}

		if molecule_name not in data:
			parameters = {'name_backend': name_backend}
			data[molecule_name] = {'times': {'naive': 0, 'order_disconnected': 0, 'order_connected': 0},
			                       'parameters': parameters}

		for label in labels:
			data[molecule_name][label] = np.append(data[molecule_name].setdefault(label, []), n_groups[label])
			data[molecule_name]['times'][label] = data[molecule_name]['times'].setdefault(label, 0) + times[label]

		save_object(data, file_name, overwrite=True, silent=True)
