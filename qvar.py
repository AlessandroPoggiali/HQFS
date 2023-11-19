import math
import numpy as np
import sys
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from qiskit.circuit.library import MCMT, RYGate
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, execute, BasicAer
from qiskit.algorithms import FasterAmplitudeEstimation
from qiskit.providers.aer import AerSimulator
from qiskit.utils import QuantumInstance

from qiskit.algorithms import EstimationProblem
from qiskit.algorithms import AmplitudeEstimation
from qiskit.utils import QuantumInstance


def _register_switcher(circuit, value, qubit_index):
    bin_str_pattern = '{:0%sb}' % len(qubit_index)
    value = bin_str_pattern.format(value)[::-1]
    for idx, bit in enumerate(value):
        if not int(bit):
            circuit.x(qubit_index[idx])

class S_QVAR():
    def __init__(self, shots) -> None:
        self.shots = min(shots, 65536)
        self.plot_filename = 'plot/S_QVAR.png'
        
    def compute_variance(self, values):
        i_qbits = int(math.ceil(math.log2(len(values))))
        q_qbits = e_qbits = i_qbits

        a = QuantumRegister(1,'a')
        e = QuantumRegister(e_qbits,'e')
        q = QuantumRegister(q_qbits,'q')
        i = QuantumRegister(i_qbits, 'i')
        r = QuantumRegister(1, 'r')

        ca = ClassicalRegister(1,'ca')
        cq = ClassicalRegister(q_qbits,'cq')
        cr = ClassicalRegister(1,'cr')
        ce = ClassicalRegister(e_qbits,'cae')


        qc = QuantumCircuit(a, e, q, i, r, ca, cq, cr, ce)

        qc.h(a)
        
        qc.cx(a,e)
        qc.x(e)
        
        qc.h(i)

        qc.barrier()
        
        for index, val in zip(range(len(values)), values):
            _register_switcher(qc, index, i)
            qc.append(MCMT(RYGate(val*2), num_ctrl_qubits=len(i), num_target_qubits=1 ), i[0:]+[r]) 
            _register_switcher(qc, index, i)
            qc.barrier()
        
        for t in range(i_qbits):
            qc.cswap(a,q[t],i[t])
            
        qc.ch(a,i)
        qc.ch(a,e)
        qc.h(a)
        qc.barrier()

        qc.h(q)

        qc.measure(a, ca) #1
        qc.measure(q, cq) #0
        qc.measure(r, cr) #1
        qc.measure(e, ce) #1

        shots = self.shots
        backend = BasicAer.get_backend('qasm_simulator')
        counts = execute(qc, backend, shots=shots).result().get_counts(qc)

        target_conf = '1'*e_qbits + ' 1 ' + '0'*q_qbits + " 1"
        
        norm_factor = 4*len(values)

        try: 
            var = (counts[target_conf])/shots
        except:
            var = 0
        return var*norm_factor

class QVAR():
    def __init__(self, eval_qubits) -> None:
        self.eval_qubits = eval_qubits
        self.plot_filename = 'plot/QVAR.png'
        
    def compute_variance(self, values):
        i_qbits = int(math.ceil(math.log2(len(values))))
        q_qbits = e_qbits = i_qbits

        a = QuantumRegister(1,'a')
        e = QuantumRegister(e_qbits,'e')
        q = QuantumRegister(q_qbits,'q')
        r = QuantumRegister(1,'r')
        i = QuantumRegister(i_qbits, 'i')

        qc = QuantumCircuit(a, e, q, r, i)

        qc.h(a)
        qc.cx(a,e)
        qc.x(e)
        qc.h(i)

        for index, val in zip(range(len(values)), values):
            _register_switcher(qc, index, i)
            qc.mcry(val*2, i[0:], r) 
            _register_switcher(qc, index, i)

        for t in range(i_qbits):
            qc.cswap(a,q[t],i[t])
        qc.ch(a,i)
        qc.ch(a,e)
        qc.h(a)

        qc.h(q)
        qc.x(q) # Classic AE consider |11..1> state as target conf

        backend = Aer.get_backend("qasm_simulator")
        quantum_instance = QuantumInstance(backend)
            
        ae = AmplitudeEstimation(
            num_eval_qubits=self.eval_qubits,  # the number of evaluation qubits specifies circuit width and accuracy
            quantum_instance=quantum_instance
        )
        
        problem = EstimationProblem(
            state_preparation=qc,  # A operator
            objective_qubits=[x for x in range(qc.num_qubits-i_qbits)],
        )
        ae_result = ae.estimate(problem)

        norm_factor = 4*len(values)

        return ae_result.estimation*norm_factor

class ML_QVAR():
    def __init__(self, eval_qubits) -> None:
        self.eval_qubits = eval_qubits
        self.plot_filename = 'plot/ML_QVAR.png'
        
    def compute_variance(self, values):
        i_qbits = int(math.ceil(math.log2(len(values))))
        q_qbits = e_qbits = i_qbits

        a = QuantumRegister(1,'a')
        e = QuantumRegister(e_qbits,'e')
        q = QuantumRegister(q_qbits,'q')
        r = QuantumRegister(1,'r')
        i = QuantumRegister(i_qbits, 'i')

        qc = QuantumCircuit(a, e, q, r, i)

        qc.h(a)
        qc.cx(a,e)
        qc.x(e)
        qc.h(i)

        for index, val in zip(range(len(values)), values):
            _register_switcher(qc, index, i)
            qc.mcry(val*2, i[0:], r) 
            _register_switcher(qc, index, i)

        for t in range(i_qbits):
            qc.cswap(a,q[t],i[t])
        qc.ch(a,i)
        qc.ch(a,e)
        qc.h(a)

        qc.h(q)
        qc.x(q) # Classic AE consider |11..1> state as target conf

        #backend = Aer.get_backend("qasm_simulator")
        backend = AerSimulator(method='statevector', device='GPU',  cuStateVec_enable=True)
        #backend.set_options(device='GPU')
        quantum_instance = QuantumInstance(backend)
            
        ae = AmplitudeEstimation(
            num_eval_qubits=self.eval_qubits,  # the number of evaluation qubits specifies circuit width and accuracy
            quantum_instance=quantum_instance
        )
        
        problem = EstimationProblem(
            state_preparation=qc,  # A operator
            objective_qubits=[x for x in range(qc.num_qubits-i_qbits)],
        )
        ae_result = ae.estimate(problem)

        norm_factor = 4*len(values)

        return ae_result.mle*norm_factor


class FAE_QVAR():
    def __init__(self, accuracy, max_iter) -> None:
        self.plot_filename = 'plot/FAE_QVAR.png'
        self.accuracy = accuracy
        self.max_iter = max_iter
        
    def compute_variance(self, values):
        i_qbits = int(math.ceil(math.log2(len(values))))
        q_qbits = e_qbits = i_qbits

        a = QuantumRegister(1,'a')
        e = QuantumRegister(e_qbits,'e')
        q = QuantumRegister(q_qbits,'q')
        r = QuantumRegister(1,'r')
        i = QuantumRegister(i_qbits, 'i')

        qc = QuantumCircuit(a, e, q, r, i)

        qc.h(a)
        qc.cx(a,e)
        qc.x(e)
        qc.h(i)

        for index, val in zip(range(len(values)), values):
            _register_switcher(qc, index, i)
            qc.mcry(val*2, i[0:], r) 
            _register_switcher(qc, index, i)

        for t in range(i_qbits):
            qc.cswap(a,q[t],i[t])
        qc.ch(a,i)
        qc.ch(a,e)
        qc.h(a)

        qc.h(q)
        qc.x(q) # Classic AE consider |11..1> state as target conf

        backend = Aer.get_backend("qasm_simulator")
        quantum_instance = QuantumInstance(backend)
            
        fae = FasterAmplitudeEstimation(
            delta=self.accuracy,  # target accuracy
            maxiter=self.max_iter,  # determines the maximal power of the Grover operator
            quantum_instance=quantum_instance,
        )
        
        problem = EstimationProblem(
            state_preparation=qc,  # A operator
            objective_qubits=[x for x in range(qc.num_qubits-i_qbits)],
        )
        fae_result = fae.estimate(problem)

        norm_factor = 4*len(values)

        return fae_result.estimation*norm_factor


    
