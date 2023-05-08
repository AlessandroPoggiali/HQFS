# Hybrid Quantum Feature Selection (HQFS) Algorithm

The Hybrid Quantum Feature Selection (HQFS) algorithm is an innovative method for feature selection that utilizes both quantum and classical computing techniques. This approach is designed to provide a more efficient and accurate means of selecting relevant features in complex data sets.

## Variance Estimation Techniques

HQFS uses two different variance estimation techniques, **QVAR** and **ML-QVAR**, to create a hybrid quantum-classical version of the variance filter feature selection process. 

The **QVAR** quantum subroutine computes the variance of a given feature using the Amplitude Estimation algorithm, a quantum algorithm used for estimating the amplitude of a specific state in a quantum system. This allows HQFS to quickly and accurately compute the variance of a feature using quantum computing techniques.

The **ML-QVAR** subroutine combines quantum and classical computing techniques by using the Amplitude Estimation algorithm with the *Maximum Likelihood Estimator* postprocessing to compute the variance of a given feature. This provides even more accurate estimates of feature variances.

