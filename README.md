# fhog-python
A python wrapper for the fhog function of PDollar Toolbox.
This function is widely used in visual tracking (e.g fDSST, ECO) to extract HOG feature.

#### Install
`python setup.py build_ext --inplace`

#### Usage
```python
import fhog
def fhog(im_patch):
    M = np.zeros(im_patch.shape[:2], dtype='float32')
    O = np.zeros(im_patch.shape[:2], dtype='float32')
    H = np.zeros([im_patch.shape[0]//4,im_patch.shape[1]//4, 32], dtype='float32') # python3
    fhog.gradientMag(im_patch.astype(np.float32),M,O)
    fhog.gradientHist(M,O,H)
    return H
```

#### TODO
- [ ] Warning in `fhog.cpp`:
```
- DeprecationWarning: NPY_ARRAY_UPDATEIFCOPY, NPY_ARRAY_INOUT_ARRAY, and NPY_ARRAY_INOUT_FARRAY are deprecated, use NPY_WRITEBACKIFCOPY, NPY_ARRAY_INOUT_ARRAY2, or NPY_ARRAY_INOUT_FARRAY2 respectively instead, and call PyArray_ResolveWritebackIfCopy before the array is deallocated, i.e. before the last call to Py_DECREF.
- DeprecationWarning: UPDATEIFCOPY detected in array_dealloc.  Required call to PyArray_ResolveWritebackIfCopy or PyArray_DiscardWritebackIfCopy is missing
```
#### References
- [AlvinZhu/matlabtb](https://github.com/AlvinZhu/matlabtb)
- [pdollar/toolbox](https://github.com/pdollar/toolbox)
