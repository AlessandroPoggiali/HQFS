# Hybrid Quantum Feature Selection (HQFS) Algorithm

The Hybrid Quantum Feature Selection (HQFS) algorithm is an innovative method for feature selection that utilizes both quantum and classical computing techniques. This approach is designed to provide a more efficient and accurate means of selecting relevant features in complex data sets.

HQFS uses [QVAR](https://github.com/AlessandroPoggiali/QVAR) to compute the variance of every feature and discard those whose variance is below a given threshold

## Quickstart

To run a simple demostration of the HQFS algorithm, follow these steps:
* Make sure you have Qiskit installed on your computer
* Install the QVAR package `pip install -i https://test.pypi.org/simple/ QVAR` (the PyPI version is not available yet)
* Clone this repo with `git clone https://github.com/AlessandroPoggiali/HQFS.git`
* Navigate to the HQFS directory and run the command `python3 hqfs.py`

The `test.py` file contains code that will run the HQFS algorithm on both synthetic and real datasets. By running this file, you will be able to see how the algorithm performs on different types of data. The output of the algorithm will be displayed in the terminal window.

Note that this is a simple demonstration, and the algorithm's performance may vary depending on the data being used. If you are interested in using the HQFS algorithm for your own data analysis, you may need to modify the code to suit your specific needs.

