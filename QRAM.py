import numpy as np

def _register_switcher(circuit, value, qubit_index):
    bin_str_pattern = '{:0%sb}' % len(qubit_index)
    value = bin_str_pattern.format(value)[::-1]
    for idx, bit in enumerate(value):
        if not int(bit):
            circuit.x(qubit_index[idx])
            
def _encode_vector(circuit, vector, k, controls, len_controls, rotation, ancillaRotation):
    for idx in range(len(vector)):
        _register_switcher(circuit, idx, k)
        data = np.arcsin(vector[idx])*2
        #circuit.append(MCMT(RYGate(data), num_ctrl_qubits=len(controls), num_target_qubits=1 ), controls+rotation)
        circuit.mcry(data, controls, rotation, ancillaRotation)
        _register_switcher(circuit, idx, k)
        #circuit.barrier()
        
def _encode_vectors(circuit, vectors, d, k, i, r, ancillaRotation):
    ctrl_qubits = d[:]+k[:]+i[:]
    len_ctrl_qubits = d.size + k.size + i.size
    for idx, vector in enumerate(vectors):
        _register_switcher(circuit, idx, i)
        _encode_vector(circuit, vector, k, ctrl_qubits, len_ctrl_qubits, r[0], ancillaRotation)
        _register_switcher(circuit, idx, i)
        #circuit.barrier()