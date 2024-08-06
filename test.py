
from hqfs import HQFS
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from util import generate_dataset
import sys

def synthetic_test():

    dataset = generate_dataset(32, 10, 7, mu=0.5, std=0.05)
    print("Dataset synth1 std=0.05\n")
    for m in [2]:
        print("\n\nTest HQFS with " + str(m) + " additional qubits\n")
        features = HQFS(dataset, dataset_name="synth1_"+str(m)+"_AE", threshold=0.08, n_processes=n_processes, version="AE", eval_qubits=m, post_processing=False)
        print("Selected features (HQFS): ", features)

        features = HQFS(dataset, dataset_name="synth1_"+str(m)+"_ML", threshold=0.08, n_processes=n_processes, version="AE", eval_qubits=m, post_processing=True)
        print("Selected features (ML-HQFS): ", features)

        
    print("Dataset synth2 std=0.35\n")
    dataset = generate_dataset(32, 10, 7, mu=0.5, std=0.5)
    for m in [2]:
        print("\n\nTest HQFS with " + str(m) + " additional qubits\n")

        features = HQFS(dataset, dataset_name="synth2_"+str(m)+"_AE", threshold=0.08, n_processes=n_processes, version="AE", eval_qubits=m, post_processing=False)
        print("Selected features (HQFS): ", features)

        features = HQFS(dataset, dataset_name="synth2_"+str(m)+"_ML", threshold=0.08, n_processes=n_processes, version="AE", eval_qubits=m, post_processing=True)
        print("Selected features (ML-HQFS): ", features)
    

def wine_test(n_processes):

    dataset = datasets.load_wine(as_frame=True).frame
    dataset = dataset.drop('target', axis=1)
    scaler = MinMaxScaler(feature_range=(-0.9999, 0.9999))
    dataset.loc[:,:] = scaler.fit_transform(dataset.loc[:,:])
    dataset = dataset.sample(n=128, random_state=123)

    for m in [2,3,4,5]:
        print("\n\nTest HQFS with " + str(m) + " evaluation qubits\n")

        features = HQFS(dataset, threshold=0.08, n_processes=n_processes, version="AE", eval_qubits=m, post_processing=False)
        print("Selected features (HQFS): ", features)

        print("\n\nTest ML-HQFS with " + str(m) + " evaluation qubits\n")

        features = HQFS(dataset, threshold=0.08, n_processes=n_processes, version="AE", eval_qubits=m, post_processing=True)
        print("Selected features (ML-HQFS): ", features)

def breast_test(n_processes):

    dataset = datasets.load_breast_cancer(as_frame=True).frame
    dataset = dataset.drop('target', axis=1)
    scaler = MinMaxScaler(feature_range=(-0.9999, 0.9999))
    dataset.loc[:,:] = scaler.fit_transform(dataset.loc[:,:])
    dataset = dataset.sample(n=16, random_state=123)

    for m in [3]:
        #print("\n\nTest HQFS with " + str(m) + " evaluation qubits\n")
        #qvar = QVAR(m)
        #features = HQFS(dataset, qvar, threshold=0.08, n_processes=n_processes)
        #print("Selected features (HQFS): ", features)

        print("\n\nTest ML-HQFS with " + str(m) + " evaluation qubits\n")

        features = HQFS(dataset, threshold=0.08, n_processes=n_processes, version="AE", eval_qubits=m, post_processing=True)
        print("Selected features (ML-HQFS): ", features)


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

    #wine_test(n_processes)
    #breast_test(n_processes)
    synthetic_test()
   
