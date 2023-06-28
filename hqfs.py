import numpy as np
import multiprocessing as mp

def HQFS_worker(to_drop, qvar, dataset, threshold, correction_factor, features):
    for f in features:
        print("featrue " + str(f))
        values = np.arcsin(dataset[f])
        variance = qvar.compute_variance(values)*correction_factor
        if variance < threshold:
            to_drop.put(f)

def HQFS(dataset, qvar, threshold=1e-02, sample_size=None, correction_factor=1, n_processes=1):
    features = list(dataset.columns)
    to_drop = mp.Queue() 
    if n_processes > 1:
        if sample_size is not None:
            dataset = dataset.sample(n=sample_size, random_state=123)
    
        chunks = np.array_split(dataset.columns, n_processes)
        processes = [mp.Process(target=HQFS_worker, args=(to_drop, qvar, dataset, threshold, correction_factor, features))  for _, features in enumerate(chunks)]
        for p in processes:
                p.start()
        for p in processes:
            p.join()

    else:
        if sample_size is not None:
            dataset = dataset.sample(n=sample_size, random_state=123)
        HQFS_worker(to_drop, qvar, dataset, threshold, correction_factor, features)

    while not to_drop.empty():
        features.remove(to_drop.get())

    return features

if __name__ == "__main__":

    from util import generate_dataset

    dataset = generate_dataset(8, 10, 7, mu=0.05, std=0.2)

    from qvar import ML_QVAR

    qvar = ML_QVAR(3)
    features = HQFS(dataset, qvar, threshold=0.08, n_processes=1)
    print("Selected features = ", features)
