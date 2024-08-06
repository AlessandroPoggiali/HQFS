import numpy as np
import multiprocessing as mp
from util import generate_dataset, _register_switcher
from QVAR import QVAR
from qiskit import QuantumCircuit, QuantumRegister

def HQFS_worker(to_drop, chunk_idx, dataset, threshold, correction_factor, features, version, eval_qubits, post_processing):
    filename = 'result/chunk_' + str(chunk_idx) + '.csv'
    f = open(filename, 'w')
    
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
        cs_var = np.var(values)*correction_factor
        print("featrue: " + str(f) + " - var: " + str(variance))
        
        f.write(f+','+str(cs_var)+","+str(variance)+'\n')
        
        if variance < threshold:
            to_drop.put(f)

def HQFS(dataset, dataset_name="dataset", threshold=1e-02, sample_size=None, correction_factor=1, n_processes=1, version="FAE", eval_qubits=5, post_processing=True):
    print("variance threshold: " + str(threshold))
    features = list(dataset.columns)
    to_drop = mp.Queue() 
    if n_processes > 1:
        if sample_size is not None:
            dataset = dataset.sample(n=sample_size, random_state=123)
    
        chunks = np.array_split(dataset.columns, n_processes)
        processes = [mp.Process(target=HQFS_worker, args=(to_drop, idx, dataset, threshold, correction_factor, features, version, eval_qubits, post_processing))  for idx, features in enumerate(chunks)]
        for p in processes:
                p.start()
        for p in processes:
            p.join()

    else:
        if sample_size is not None:
            dataset = dataset.sample(n=sample_size, random_state=123)
        HQFS_worker(to_drop, 0, dataset, threshold, correction_factor, features, version, eval_qubits, post_processing)

    filename = 'result/' + dataset_name + "_m" + str(eval_qubits) + ".csv"
    f = open(filename, 'w')
    f.write("idx,feature,classical,qvar\n")
    idx_col = 0
    for idx_chunk in range(len(chunks)):
        f1_name  = "result/chunk_" + str(idx_chunk) + ".csv"
        with open(f1_name) as f1:
            lines = [line.rstrip('\n') for line in f1]
            for line in lines:
                f.write(str(idx_col)+","+line+'\n')
                idx_col = idx_col + 1
        f1.close()
    f.close()

    while not to_drop.empty():
        features.remove(to_drop.get())

    return features

if __name__ == "__main__":

    dataset = generate_dataset(4, 10, 7, mu=0.05, std=0.2)
    print("List of features: "+ str(list(dataset.columns)))

    features = HQFS(dataset, threshold=0.08, n_processes=1)
    print("Selected features = ", features)
