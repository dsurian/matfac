# Matrix factorisation with a shared latent space
This is an implementation of matrix factorisation with a shared latent space in Cython used in [1]. Given two matrices, **R** and **T**, where **R** is trials vs features matrix (e.g. tf-idf, LDA topic distributions, etc.) and **T** trials vs systematic reviews matrix with missing information (i.e. link information), the aim is to predict missing values in **T**, i.e. the missing link information between trial and systematic reviews. See [1] for more details.


### Environment
---
The code was built and tested on:
* Ubuntu 14.04 LTS
* Python 2.7.12 | Anaconda 4.3.23
* NumPy 1.11.3
* Cython 0.23.4


### Usage
---
To compile:
```sh
$ python setup.py build_ext --inplace
```
(Optional) To generate the ''annotate'' html file:
```sh
$ cython matfac.pyx -a
```


Once you have compiled the Cython code, you can import directly from the Python code:
```python
import matfac
```
and call:
```python
T_est = np.asarray(matfac.run(R, T, estP, estP, estQ, estW, PS_K, numNonZeroT, K, numRow, numCol1, numCol2, numIter, alpha_par, lambda_par, lambda_t_par, T_est, VERBOSE))
```
Note that you have to import NumPy first as *np* at the beginning of your code. The script has been optimised to minimise the Python interaction so it could run faster. In order to do so, some variables are initiated in advance in the Python code. The initialisation values and regularisation parameters may need to be adapted to your own application. Below is the description of the input and output:

#### Input:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**R** : *ndarray*, [*n_samples*, *n_features_1*]. *n_features_1* is the column size of **R**.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**T** : *ndarray*, [*n_samples*, *n_features_2*]. *n_features_2* is the column size of **T**.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**estP** : *ndarray*, [*n_samples*, *k_factors*], initialized using small random numbers (e.g. np.random.random_sample / 10).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**estQ** : *ndarray*, [*n_features_1*, *k_factors*], initialized using small random numbers (e.g. np.random.random_sample / 10).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**estW** : *ndarray*, [*n_features_2*, *k_factors*], initialized using small random numbers (e.g. np.random.random_sample / 10).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**PS_K** : *ndarray*, initiated by np.zeros(*k_factors*)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**numNonZeroT** : *float* value, represents the number of non-zero entries in matrix **T**.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**K** : *int* value, number of latent factors (i.e. *k_factors*).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**numRow** : *int* value, number of samples (i.e. *n_samples*, which is the number of rows in **R** (or **T**)).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**numCol1** : *int* value, number of features in matrix **R** (i.e. *n_features_1*).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**numCol2** : *int* value, number of features in matrix **T** (i.e. *n_features_2*).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**numIter** : *int* value, maximum number of iterations.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**alpha_par**: *float* value, learning rate. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**lambda_par**, **lambda_t_par** : *float* value, regularisation parameters.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**T_est** : *ndarray*, [*n_samples*, *n_features_2*]. This matrix is initialised to zero and will hold the predicted **T**.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**VERBOSE** : *int* value, set to 1 to show on screen the learning progress and 0 to hide.

#### Return:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**T_est** : Cython memoryview, [*n_samples*, *n_features_2*].

# Reference
---
1. *A shared latent space matrix factorisation method for recommending new trial evidence for systematic review updates*. Didi Surian, Adam G. Dunn, Liat Orenstein, Rabia Bashir, Enrico Coiera, Florence T. Bourgeois. Journal of Biomedical Informatics Vol. 79, March 2018, p. 32-40
