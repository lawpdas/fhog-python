# fhog-python
A python wrapper for the fhog function of PDollar Toolbox. This function is widely used in visual tracking to extract HOG feature (e.g fDSST, ECO).

Warning in `fhog.cpp`:
- DeprecationWarning: NPY_ARRAY_UPDATEIFCOPY, NPY_ARRAY_INOUT_ARRAY, and NPY_ARRAY_INOUT_FARRAY are deprecated, use NPY_WRITEBACKIFCOPY, NPY_ARRAY_INOUT_ARRAY2, or NPY_ARRAY_INOUT_FARRAY2 respectively instead, and call PyArray_ResolveWritebackIfCopy before the array is deallocated, i.e. before the last call to Py_DECREF.
- DeprecationWarning: UPDATEIFCOPY detected in array_dealloc.  Required call to PyArray_ResolveWritebackIfCopy or PyArray_DiscardWritebackIfCopy is missing

## References
- [AlvinZhu/matlabtb](https://github.com/AlvinZhu/matlabtb)
- [pdollar/toolbox](https://github.com/pdollar/toolbox)
