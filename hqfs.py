import numpy as np
import multiprocessing as mp
from util import generate_dataset, _register_switcher
from QVAR import QVAR
from qiskit import QuantumCircuit, QuantumRegister

def HQFS_worker(to_drop, dataset, threshold, correction_factor, features, version, eval_qubits, post_processing):
    for f in features:
        
        values = dataset[f]
        N = len(values)
        n = int(np.ceil(np.log2(N)))

        r = QuantumRegister(1, 'r') 
        i = QuantumRegister(n, 'i') 
        U = QuantumCircuit(i, r)
        U.h(i)
        
        # data encoding 
        for index, val in zip(range(N), values):
            _register_switcher(U, index, i)
            U.mcry(np.arcsin(val)*2, i[0:], r) 
            _register_switcher(U, index, i)

        variance =  QVAR(
            U, 
            var_index=list(range(n)),
            ps_index=[U.num_qubits-1], 
            version=version,
            eval_qbits=eval_qubits,
            n_h_gates=n,
            postprocessing=post_processing 
        )*correction_factor

        print("featrue: " + str(f) + " - var: " + str(variance))

        if variance < threshold:
            to_drop.put(f)

def HQFS(dataset, threshold=1e-02, sample_size=None, correction_factor=1, n_processes=1, version="FAE", eval_qubits=5, post_processing=True):
    print("variance threshold: " + str(threshold))
    features = list(dataset.columns)
    to_drop = mp.Queue() 
    if n_processes > 1:
        if sample_size is not None:
            dataset = dataset.sample(n=sample_size, random_state=123)
    
        chunks = np.array_split(dataset.columns, n_processes)
        processes = [mp.Process(target=HQFS_worker, args=(to_drop, dataset, threshold, correction_factor, features, version, eval_qubits, post_processing))  for _, features in enumerate(chunks)]
        for p in processes:
                p.start()
        for p in processes:
            p.join()

    else:
        if sample_size is not None:
            dataset = dataset.sample(n=sample_size, random_state=123)
        HQFS_worker(to_drop, dataset, threshold, correction_factor, features, version, eval_qubits, post_processing)

    while not to_drop.empty():
        features.remove(to_drop.get())

    return features

if __name__ == "__main__":

    dataset = generate_dataset(4, 10, 7, mu=0.05, std=0.2)
    print("List of features: "+ str(list(dataset.columns)))

    features = HQFS(dataset, threshold=0.08, n_processes=1)
    print("Selected features = ", features)
