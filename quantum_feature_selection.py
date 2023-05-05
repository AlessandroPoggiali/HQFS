from quantum_variance import qvar_AE, qvar_FAE, qvar_measure_all
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from util import rbo, generate_dataset
import sys

import multiprocessing as mp

def test_sample_variance():
    N = 8
    df = datasets.load_wine(as_frame=True).frame
    df = df.drop('target', axis=1)

    
    for idx, col in enumerate(df.columns):
        variances = []
        variances_wc = []
        print("Column: " + col)
        for i in range(1):
            df_sample = df.sample(n=N, random_state=15*i)
            correction_factor = N/(N-1)
            variances.append(np.var(df_sample[col])*correction_factor)
            variances_wc.append(np.var(df_sample[col]))
        print("mean of sample variances: " + str(np.mean(variances)))
        print("mean of sample variances without correction factor: " + str(np.mean(variances_wc)))
        print("actual variance: " + str(np.var(df[col])))
        print("")



def plot_ranking():
    dataset_name = 'wine'

    filename = 'result/' + dataset_name + ".csv"
    result = pd.read_csv(filename, sep=',')

    #plt.plot(result['qvar_s'], label='qvar_s')
    plt.plot(result['qvar_ae'], label='qvar_ae')
    plt.plot(result['qvar_ae_ml'], label='qvar_ae_ml')
    #plt.plot(result['qvar_fae'], label='qvar_fae')
    plt.plot(result['cs_var'], label='cs_var')
    #plt.plot(result['cp_var'], label='cp_var')
    plt.legend(loc="upper left")

    filename = 'plot/QFS.png'

    if filename is not None:
            plt.savefig(filename)
    else:
        plt.show()

def parallel_test(dataset, dataset_name, sample_size, shots, eval_qubits, n_processes, max_iter=1):

    sample = dataset.sample(n=sample_size, random_state=123)
    correction_factor = sample_size/(sample_size-1)   
    correction_factor = 1

    chunks = np.array_split(dataset.columns, n_processes)

    processes = [mp.Process(target=quantum_feature_selection, args=(dataset, sample, correction_factor, idx, columns, shots, eval_qubits, max_iter))  for idx, columns in enumerate(chunks)]
    for p in processes:
            p.start()

    for p in processes:
        p.join()
        print("process ", p, " terminated")

    print("Processes joined")

    filename = 'result/' + dataset_name + "_m" + str(eval_qubits) + ".csv"
    f = open(filename, 'w')
    f.write("idx,feature,cs_var,cp_var,qvar_s,qvar_ae,qvar_ae_ml\n")
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

    #plot_ranking()

def quantum_feature_selection(dataset, sample, correction_factor, chunk_idx, columns, shots, eval_qubits, max_iter=1):

    #qvar_measures = qvar_measure_all(shots)
    qvar_ae = qvar_AE(eval_qubits)
    #qvar_fae = qvar_FAE(0.01, max_iter)

    filename = 'result/chunk_' + str(chunk_idx) + '.csv'
    f = open(filename, 'w')

    for column_name in columns:
        print("Computing variance for column: " + column_name)
        cs_var = np.var(sample[column_name])*correction_factor
        cp_var = np.var(dataset[column_name])
        
        ran_values = np.arcsin(sample[column_name])

        #quantum_variance_measures = qvar_measures.compute_variance(ran_values)*correction_factor
        quantum_variance_ae, quantum_variance_ae_ml  = qvar_ae.compute_variance(ran_values)*correction_factor
        #quantum_variance_fae = qvar_fae.compute_variance(ran_values)
        
        quantum_variance_fae = quantum_variance_measures = 0
        
        #f.write('cs_var,cp_var,qvar_s,qvar_ae,qvar_fae\n')
        f.write(column_name+','+str(cs_var)+","+str(cp_var)+","+str(quantum_variance_measures)+","+str(quantum_variance_ae)+","+str(quantum_variance_ae_ml)+'\n')
        #f.write(column_name+','+str(cs_var)+","+str(cp_var)+","+str(quantum_variance_measures)+","+str(quantum_variance_ae)+","+str(quantum_variance_ae_ml)+'\n')
    f.close()


def synthetic_test_AE():
    dataset = generate_dataset(32, 10, 7, mu=0.5, std=0.2)
    print("Dataset synth2 std=0.20\n")
    for eval_qubits in [2,3,4,5]:
        print("\n\nTest QFS with " + str(eval_qubits) + " additional qubits\n")
        parallel_test(dataset, "synth2", 32, 1000, eval_qubits, n_processes)
    print("Dataset synth2 std=0.35\n")
    dataset = generate_dataset(32, 10, 7, mu=0.5, std=0.35)
    for eval_qubits in [2,3,4,5]:
        print("\n\nTest QFS with " + str(eval_qubits) + " additional qubits\n")
        parallel_test(dataset, "synth4", 32, 1000, eval_qubits, n_processes)

def synthetic_test_measureall():
    dataset = generate_dataset(32, 10, 7, mu=0.5, std=0.5)
    #scaler = MinMaxScaler(feature_range=(-0.9999, 0.9999))
    #dataset.loc[:,:] = scaler.fit_transform(dataset.loc[:,:])

    for eval_qubits in [2,3,4,5,6,7]:
        shots = (2**eval_qubits-1)**2
        print("\n\nTest QFS with " + str(shots) + " shots\n")
        parallel_test(dataset, "synth2", 32, shots, eval_qubits, n_processes)

def synthetic_test_FAE():
    dataset = generate_dataset(64, 10, 7, mu=0.5, std=0.05)

    for max_iter in [1,2,3,4,5]:
        print("\n\nTest QFS (FAE) with " + str(max_iter) + " iterations\n")
        parallel_test(dataset, "synth1", 64, 1000, max_iter, n_processes, max_iter)

def wine_test():
    dataset = datasets.load_wine(as_frame=True).frame
    dataset = dataset.drop('target', axis=1)
    scaler = MinMaxScaler(feature_range=(-0.9999, 0.9999))
    dataset.loc[:,:] = scaler.fit_transform(dataset.loc[:,:])
    sample_size = 128
    for l in [1,2,3,4,5,6,7]:
        #m = 7
        #shots = (2**m-1)**2
        print("\n\nTest QFS (FAE) with " + str(l) + " iterations\n")
        parallel_test(dataset, 'wine', sample_size, 0, l, n_processes, l)

def lungcancer_test(n_processes):
    col = ['f'+str(x) for x in list(range(57))]
    dataset = pd.read_csv('./data/lung-cancer.data',header=None,names=col)
    dataset = dataset.replace('?', np.nan)
    
    for c in col:
        mean_val = pd.to_numeric(dataset[c], errors='coerce').mean()
        dataset[c].fillna(value=mean_val,inplace=True)
    print(dataset.head())
    scaler = MinMaxScaler(feature_range=(-0.9999, 0.9999))
    dataset.loc[:,:] = scaler.fit_transform(dataset.loc[:,:])

    for eval_qubits in [2,3,4,5]:
        print("\n\nTest QFS with " + str(eval_qubits) + " evaluation qubits\n")
        parallel_test(dataset, "lung_cancer", 32, 1000, eval_qubits, n_processes)

if __name__ == "__main__":

    #test_sample_variance()
    #exit()

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

    lungcancer_test(n_processes)
    exit()
    #synthetic_test_measureall()   
    #wine_test()
    #synthetic_test_AE()
    #exit()    

    dataset = datasets.load_wine(as_frame=True).frame
    dataset = dataset.drop('target', axis=1)
    scaler = MinMaxScaler(feature_range=(-0.9999, 0.9999))
    dataset.loc[:,:] = scaler.fit_transform(dataset.loc[:,:])
    sample_size = 64
    shots = 10000
    eval_qubits = 3

    for eval_qubits in [2,3,4,5,6,7,8]:
        print("\n\nTest QFS with " + str(eval_qubits) + " additional qubits\n")
        parallel_test(dataset, 'wine', sample_size, shots, eval_qubits, n_processes)
    exit()

    # loading dataset 
    N = 16
    df = datasets.load_wine(as_frame=True).frame
    df = df.drop('target', axis=1)
    scaler = MinMaxScaler(feature_range=(-0.9999, 0.9999))
    df.loc[:,:] = scaler.fit_transform(df.loc[:,:])
    df_sample = df.sample(n=N, random_state=123)


    shots = 1024*N
    eval_qubits = 3

    qvar_measures = qvar_measure_all(shots)
    qvar_ae = qvar_AE(eval_qubits)
    qvar_fae = qvar_FAE()

    classic_sample_variances = []
    classic_population_variances = []
    quantum_variances_measures = []
    quantum_variances_ae = []
    quantum_variances_fae = []

    correction_factor = N/(N-1)

    # compute variances
    for idx, col in enumerate(df.columns):
        print("Computing variance for column " + str(idx) + ": " + col)
        cs_var = np.var(df_sample[col])*correction_factor
        print("classic sample variance: " + str(cs_var))
        classic_sample_variances.append(cs_var)
        cp_var = np.var(df[col])
        print("classic population variance: " + str(cp_var))
        classic_population_variances.append(cp_var)

        ran_values = np.arcsin(df_sample[col])

        quantum_variance_measures = qvar_measures.compute_variance(ran_values)*correction_factor
        print("qvar_s: " + str(quantum_variance_measures))
        quantum_variances_measures.append(quantum_variance_measures)

        quantum_variance_ae = qvar_ae.compute_variance(ran_values)*correction_factor
        print("qvar_ae: " + str(quantum_variance_ae))
        quantum_variances_ae.append(quantum_variance_ae)

        quantum_variance_fae = qvar_fae.compute_variance(ran_values)*correction_factor
        print("qvar_fae: " + str(quantum_variance_fae))
        quantum_variances_fae.append(quantum_variance_fae)

    plt.plot(quantum_variances_measures, label='qvar_s')
    plt.plot(quantum_variances_ae, label='qvar_ae')
    plt.plot(quantum_variances_fae, label='qvar_fae')
    plt.plot(classic_sample_variances, label='classic variance')
    #plt.plot(classic_population_variances, label='classic variance populaiton')
    plt.legend(loc="upper left")

    filename = 'plot/QFS.png'

    if filename is not None:
            plt.savefig(filename)
    else:
        plt.show()

    # compute ranking
    quantum_ranking_measures = [(v,c) for v, c in sorted(zip(quantum_variances_measures, df.columns))]
    quantum_ranking_ae = [(v,c) for v, c in sorted(zip(quantum_variances_ae, df.columns))]
    classical_ranking = [(v,c) for v, c in sorted(zip(classic_sample_variances, df.columns))]
    classical_population_ranking = [(v,c) for v, c in sorted(zip(classic_population_variances, df.columns))]

    f = open('log.txt', 'w')

    f.write("\n\nVariance Ranking (classical - qvar_s - qvar_ae)\n")
    for idx, t in enumerate(zip(quantum_ranking_measures, quantum_ranking_ae, classical_ranking)):
        f.write(str(idx) + ") " + t[2][1] + "(" + str(round(t[2][0],2)) + ")" + " - " + t[0][1]  + "(" + str(round(t[0][0],2)) + ")" + " - " + t[1][1]  + "(" + str(round(t[1][0],2)) + ")\n")
    
    quantum_ranking_measures = [x[1] for x in quantum_ranking_measures]
    quantum_ranking_ae = [x[1] for x in quantum_ranking_ae]
    classical_ranking = [x[1] for x in classical_ranking]

    t1, p1 = stats.kendalltau(quantum_ranking_measures, classical_ranking)
    t2, p2 = stats.kendalltau(quantum_ranking_ae, classical_ranking)
    f.write("\nKendalltau correlation between rankings (classical vs qvar_s): " + str(t1) + "\n")
    f.write("Kendalltau correlation between rankings (classical vs qvar_ae): " + str(t2) + "\n")
    f.write("Rank Biased Overlap between rankings (classical vs qvar_s): " + str(rbo(quantum_ranking_measures, classical_ranking, p=0.75)) + "\n")
    f.write("Rank Biased Overlap between rankings (classical vs qvar_ae): " + str(rbo(quantum_ranking_ae, classical_ranking, p=0.75)) + "\n")

    f.close()
