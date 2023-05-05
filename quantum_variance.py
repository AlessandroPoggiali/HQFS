import random
import math
import numpy as np
import sys
import pandas as pd

import multiprocessing as mp

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, normalize

from qiskit.circuit.library import MCMT, RYGate, RZGate, CRYGate, XGate
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, execute, BasicAer
from qiskit.visualization import plot_bloch_vector, plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.algorithms import MaximumLikelihoodAmplitudeEstimation, IterativeAmplitudeEstimation, FasterAmplitudeEstimation
from qiskit.utils import QuantumInstance

from qiskit.algorithms import EstimationProblem
from qiskit.algorithms import AmplitudeEstimation
from qiskit.quantum_info import Statevector
from qiskit.utils import QuantumInstance


def _register_switcher(circuit, value, qubit_index):
    bin_str_pattern = '{:0%sb}' % len(qubit_index)
    value = bin_str_pattern.format(value)[::-1]
    for idx, bit in enumerate(value):
        if not int(bit):
            circuit.x(qubit_index[idx])

class qvar_measure_all():
    def __init__(self, shots) -> None:
        self.shots = min(shots, 65536)
        self.plot_filename = 'plot/qvar_measure_all.png'
        
    def compute_variance(self, values):
        i_qbits = int(math.ceil(math.log2(len(values))))
        q_qbits = ae_qbits = i_qbits

        a = QuantumRegister(1,'a')
        ae = QuantumRegister(ae_qbits,'ae')
        q = QuantumRegister(q_qbits,'q')
        i = QuantumRegister(i_qbits, 'i')
        r = QuantumRegister(1, 'r')

        ca = ClassicalRegister(1,'ca')
        cq = ClassicalRegister(q_qbits,'cq')
        cr = ClassicalRegister(1,'cr')
        cae = ClassicalRegister(ae_qbits,'cae')


        qc = QuantumCircuit(a, ae, q, i, r, ca, cq, cr, cae)

        qc.h(a)
        
        qc.cx(a,ae)
        qc.x(ae)
        
        qc.h(i)

        qc.barrier()
        
        for index, val in zip(range(len(values)), values):
            _register_switcher(qc, index, i)
            qc.append(MCMT(RYGate(val*2), num_ctrl_qubits=len(i), num_target_qubits=1 ), i[0:]+[r]) #togliendo il *2 va meglio?
            _register_switcher(qc, index, i)
            qc.barrier()
        
        for t in range(i_qbits):
            qc.cswap(a,q[t],i[t])
            
        qc.ch(a,i)
        qc.ch(a,ae)
        qc.h(a)
        qc.barrier()

        qc.h(q)

        qc.measure(a, ca) #1
        qc.measure(q, cq) #0
        qc.measure(r, cr) #1
        qc.measure(ae, cae) #1

        shots = self.shots
        backend = BasicAer.get_backend('qasm_simulator')
        counts = execute(qc, backend, shots=shots).result().get_counts(qc)

        target_conf = '1'*ae_qbits + ' 1 ' + '0'*q_qbits + " 1"
        
        n_hadamard = 2+2*i_qbits
        norm_factor = 2**n_hadamard/len(values)

        try: 
            var = (counts[target_conf])/shots
        except:
            var = 0
        return var*norm_factor

class qvar_AE():
    def __init__(self, eval_qubits) -> None:
        self.eval_qubits = eval_qubits
        self.plot_filename = 'plot/qvar_AE.png'
        
    def compute_variance(self, values):
        i_qbits = int(math.ceil(math.log2(len(values))))
        q_qbits = ae_qbits = i_qbits

        a = QuantumRegister(1,'a')
        ae = QuantumRegister(ae_qbits,'ae')
        q = QuantumRegister(q_qbits,'q')
        r = QuantumRegister(1,'r')
        i = QuantumRegister(i_qbits, 'i')

        qc = QuantumCircuit(a, ae, q, r, i)

        qc.h(a)
        qc.cx(a,ae)
        qc.x(ae)
        qc.h(i)

        for index, val in zip(range(len(values)), values):
            _register_switcher(qc, index, i)
            qc.mcry(val*2, i[0:], r) 
            _register_switcher(qc, index, i)

        for t in range(i_qbits):
            qc.cswap(a,q[t],i[t])
        qc.ch(a,i)
        qc.ch(a,ae)
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

        #n_hadamard = 2+2*i_qbits
        #norm_factor = 2**n_hadamard/len(values)
        norm_factor = 4*len(values)

        return ae_result.estimation*norm_factor, ae_result.mle*norm_factor

class qvar_FAE():
    def __init__(self, accuracy, max_iter) -> None:
        self.plot_filename = 'plot/qvar_FAE.png'
        self.accuracy = accuracy
        self.max_iter = max_iter
        
    def compute_variance(self, values):
        i_qbits = int(math.ceil(math.log2(len(values))))
        q_qbits = ae_qbits = i_qbits

        a = QuantumRegister(1,'a')
        ae = QuantumRegister(ae_qbits,'ae')
        q = QuantumRegister(q_qbits,'q')
        r = QuantumRegister(1,'r')
        i = QuantumRegister(i_qbits, 'i')

        qc = QuantumCircuit(a, ae, q, r, i)

        qc.h(a)
        qc.cx(a,ae)
        qc.x(ae)
        qc.h(i)

        for index, val in zip(range(len(values)), values):
            _register_switcher(qc, index, i)
            qc.mcry(val*2, i[0:], r) 
            _register_switcher(qc, index, i)

        for t in range(i_qbits):
            qc.cswap(a,q[t],i[t])
        qc.ch(a,i)
        qc.ch(a,ae)
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

        n_hadamard = 2+2*i_qbits
        norm_factor = 2**n_hadamard/len(values)

        return fae_result.estimation*norm_factor


def run(qvar, entries=2, n_variances=10, seed=123):
    np.random.seed(seed)

    quantum_variances = []
    classic_variances = []

    for _ in range(n_variances):
        ran_values = [np.random.uniform(-1,1) for _ in range(entries)]
        classic_variances.append(np.var(ran_values))
        ran_values = np.arcsin(ran_values)
        quantum_variances.append(qvar.compute_variance(ran_values))

    plt.plot(quantum_variances, label='Quantum')
    plt.plot(classic_variances, label='Classic')
    plt.legend(loc="upper left")

    filename = qvar.plot_filename

    if filename is not None:
            plt.savefig(filename)
    else:
        plt.show()

    differences = [(q-c)**2 for q,c in zip(quantum_variances, classic_variances)]

    print(quantum_variances[0]-classic_variances[0])
    print(np.mean(differences))
    print(np.var(differences))


def test_comparison():
    Nlist = [2,4,8,16]
    n_variances = 50
    np.random.seed(123)

    f_mse = open('result/mse.csv', 'w')

    f_mse.write("N,qvar_s,qvar_ae,qvar_fae\n")
    
    for N in Nlist:
        
        f_result = open('result/N=' + str(N) + '.csv', 'w')
        f_result.write("qvar_s,qvar_ae,qvar_fae\n")
        classic_variances = []
        qvars_1 = []
        qvars_2 = []
        qvars_3 = []
        qvar_1 = qvar_measure_all(2048*N)
        qvar_2 = qvar_AE(3)
        qvar_3 = qvar_FAE()
        for i in range(n_variances):
            print("N = " + str(N) + " - v" + str(i))
            ran_values = [np.random.uniform(-1,1) for _ in range(N)]
            classic_variances.append(np.var(ran_values))
            ran_values = np.arcsin(ran_values)
            qvars_1.append(qvar_1.compute_variance(ran_values))
            qvars_2.append(qvar_2.compute_variance(ran_values))
            qvars_3.append(qvar_3.compute_variance(ran_values))

            f_result.write(str(qvars_1[i]) + ',' + str(qvars_2[i]) + ',' + str(qvars_3[i]) + '\n')

        plt.plot(qvars_1, label='qvar_s')
        plt.plot(qvars_2, label='qvar_ae')
        plt.plot(qvars_3, label='qvar_fae')
        plt.plot(classic_variances, label='classic')
        plt.legend(loc="upper left")
        plt.savefig("plot/N=" + str(N) + ".png")

        plt.cla() 
        plt.clf() 
        plt.close('all')

        differences_qvars_1 = [(q-c)**2 for q,c in zip(qvars_1, classic_variances)]
        differences_qvars_2 = [(q-c)**2 for q,c in zip(qvars_2, classic_variances)]
        differences_qvars_3 = [(q-c)**2 for q,c in zip(qvars_3, classic_variances)]

        plt.plot(differences_qvars_1, label='qvar_s')
        plt.plot(differences_qvars_2, label='qvar_ae')
        plt.plot(differences_qvars_3, label='qvar_fae')
        plt.legend(loc="upper left")
        plt.savefig("plot/errorN=" + str(N) + ".png")

        plt.cla() 
        plt.clf() 
        plt.close('all')

        f_mse.write(str(N) + "," + str(np.mean(differences_qvars_1)) + "-" + str(np.var(differences_qvars_1)) +
                         "," + str(np.mean(differences_qvars_2)) + "-" + str(np.var(differences_qvars_2)) +
                         "," + str(np.mean(differences_qvars_3)) + "-" + str(np.var(differences_qvars_3)) + '\n')

        f_result.close()


    f_mse.close()

def test_m_ae():
    mlist = [1,2,3,4,5,6]
    N = 32
    n_variances = 10

    f_mse = open('result/mse.csv', 'w')

    f_mse.write("m,mean_qvar_ae,std_qvar_ae,mean_qvar_ae_ml,std_qvar_ae_ml\n")
    
    classic_variances = []
    ran_values = []
    for i in range(n_variances):
        np.random.seed(i*123)
        ran_values.append([np.random.uniform(-1,1) for _ in range(N)])
        classic_variances.append(np.var(ran_values[i]))

    for m in mlist:
        f_result = open('result/m=' + str(m) + '.csv', 'w')
        f_result.write("classic,qvar_ae,qvar_ae_ml\n")
        
        qvars_ae = []
        qvars_ae_ml = []
        qvar_ae = qvar_AE(m)
        for i in range(n_variances):
            print("m = " + str(m) + " - v" + str(i))
            ae_result, ae_result_mle = qvar_ae.compute_variance(np.arcsin(ran_values[i]))
            qvars_ae.append(ae_result)
            qvars_ae_ml.append(ae_result_mle)
            f_result.write(str(classic_variances[i]) + ',' + str(qvars_ae[i]) + ',' + str(qvars_ae_ml[i]) + '\n')

        plt.plot(qvars_ae, label='QVAR')
        plt.plot(qvars_ae_ml, label='ML-QVAR')
        plt.plot(classic_variances, label='VAR')
        plt.legend(loc="upper left")
        plt.savefig("plot/m=" + str(m) + ".png")

        plt.cla() 
        plt.clf() 
        plt.close('all')

        differences_qvars_ae = [(q-c)**2 for q,c in zip(qvars_ae, classic_variances)]
        differences_qvars_ae_ml = [(q-c)**2 for q,c in zip(qvars_ae_ml, classic_variances)]

        plt.plot(differences_qvars_ae, label='QVAR')
        plt.plot(differences_qvars_ae_ml, label='ML-QVAR')
        plt.legend(loc="upper left")
        plt.savefig("plot/errorm=" + str(m) + ".png")

        plt.cla() 
        plt.clf() 
        plt.close('all')

        f_mse.write(str(m) + "," + str(np.mean(differences_qvars_ae)) + "," + str(np.var(differences_qvars_ae)) +
                     "," + str(np.mean(differences_qvars_ae_ml)) + "," + str(np.var(differences_qvars_ae_ml)) + '\n') 

        f_result.close()


    f_mse.close()

def test_m(m, idx, chunk):
    qvar_ae = qvar_AE(m)
    filename = 'result/chunk_' + str(idx) + '.csv'
    f = open(filename, 'w')

    for ran in chunk:
        ae_result, ae_result_mle = qvar_ae.compute_variance(np.arcsin(ran))
        f.write(str(np.var(ran)) + ',' + str(ae_result) + ',' + str(ae_result_mle) + '\n')
    f.close()

def test_m_ae_parallel(n_processes):
    mlist = [1,2,3,4,5,6]
    N = 32
    n_variances = 5

    f_mse = open('result/mse.csv', 'w')

    f_mse.write("m,mean_qvar_ae,std_qvar_ae,mean_qvar_ae_ml,std_qvar_ae_ml\n")
    
    classic_variances = []
    ran_values = []
    for i in range(n_variances):
        np.random.seed(i*123)
        ran_values.append([np.random.uniform(-1,1) for _ in range(N)])

    for m in mlist:
        print("m="+str(m)+"\n")
        chunks = np.array_split(ran_values, n_processes)

        processes = [mp.Process(target=test_m, args=(m, idx, chunk))  for idx, chunk in enumerate(chunks)]
        for p in processes:
                p.start()

        for p in processes:
            p.join()
            print("process ", p, " terminated")

        print("Processes joined")

        filename = 'result/random' + "_m" + str(m) + ".csv"
        f = open(filename, 'w')
        f.write("idx,cs_var,qvar_ae,qvar_ae_ml\n")
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

        df_m = pd.read_csv(filename,sep=',')
        qvars_ae = df_m['qvar_ae']
        classic_variances = df_m['cs_var']
        qvars_ae_ml = df_m['qvar_ae_ml']

        differences_qvars_ae = [(q-c)**2 for q,c in zip(qvars_ae, classic_variances)]
        differences_qvars_ae_ml = [(q-c)**2 for q,c in zip(qvars_ae_ml, classic_variances)]

        #plt.plot(differences_qvars_ae, label='QVAR')
        #plt.plot(differences_qvars_ae_ml, label='ML-QVAR')
        #plt.legend(loc="upper left")
        #plt.savefig("plot/errorm=" + str(m) + ".png")

        #plt.cla() 
        #plt.clf() 
        #plt.close('all')

        f_mse.write(str(m) + "," + str(np.mean(differences_qvars_ae)) + "," + str(np.var(differences_qvars_ae)) +
                     "," + str(np.mean(differences_qvars_ae_ml)) + "," + str(np.var(differences_qvars_ae_ml)) + '\n') 
    f_mse.close()


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("ERROR: type '" + str(sys.argv[0]) + " n_processes' to execute the test")
        exit()
   
    try:
        n_processes = int(sys.argv[1])
    except ValueError:
        print("ERROR: specify a positive integer for the number of processes")
        exit()
    
    if n_processes < 0:
        print("ERROR: specify a positive integer for the number of processes")
        exit()

    test_m_ae_parallel(n_processes)
    exit()

    points = []
    m = 3
    for k in range(2**m):
        points.append(np.sin((math.pi*k)/(2**m))**2)
    dist = []
    for i in range(2**m-1):
        dist.append((points[i]-points[i+1])**2)
    print("max dist " + str(max(dist)))
    print("max err: " + str(np.mean(dist)/2))
    #print("max squared err: " + str((max(dist)/2)**2))

    print("\nqvar_ae\n")
    N = 4
    classic_variances = []
    qvars_1 = []
    qvar_1 = qvar_AE(m, post_processing=False)
    for i in range(20):
        print(i)
        ran_values = [np.random.uniform(-1,1) for _ in range(N)]
        classic_variances.append(np.var(ran_values))
        ran_values = np.arcsin(ran_values)
        qvars_1.append(qvar_1.compute_variance(ran_values))
    print("MSE: " + str(np.mean([(q-c)**2 for q,c in zip(qvars_1, classic_variances)])))
