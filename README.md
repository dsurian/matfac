# Matrix factorisation with shared latent space
This is an implementation of *matrix factorisation with shared latent space*. Given two matrices





### Usage
```sh
$ python setup.py build_ext --inplace
```

Once you have compiled the cython code, you can import directly from your python code:
```python
import matfac
```
and call
```python
T_est = matfac.run(R, T, estP, estP, estQ, estW, PS_K, numNonZeroT, K, numRow, numCol1, numCol2, numIter, alpha_par, lambda_par, lambda_t_par, T_est, VERBOSE)
```



# Reference



