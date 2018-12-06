#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "gradient.hpp"

static PyObject *
gradientMag(PyObject *dummy, PyObject *args) {
    PyObject *arg1 = NULL, *arg2 = NULL, *arg3 = NULL;
    PyArrayObject *I_arr = NULL, *M_arr = NULL, *O_arr = NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!",   
        &PyArray_Type, &arg1, 
        &PyArray_Type, &arg2, 
        &PyArray_Type, &arg3))
        return NULL;


    I_arr = (PyArrayObject *) PyArray_FROM_OF(arg1, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    M_arr = (PyArrayObject *) PyArray_FROM_OF(arg2, NPY_ARRAY_INOUT_FARRAY2);
    O_arr = (PyArrayObject *) PyArray_FROM_OF(arg3, NPY_ARRAY_INOUT_FARRAY2);

    void *I = PyArray_DATA(I_arr);
    void *M = PyArray_DATA(M_arr);
    void *O = PyArray_DATA(O_arr);

    npy_intp *I_shape = PyArray_DIMS(I_arr);

    int type_I = PyArray_TYPE(I_arr);


    int h = (int) I_shape[0];
    int w = (int) I_shape[1];
    int d = (int) I_shape[2];

    int full = 1;

    gradMag((float *) I, (float *) M, (float *) O, h, w, d, full>0);




    Py_DECREF(I_arr);
    PyArray_ResolveWritebackIfCopy(M_arr);	
    PyArray_ResolveWritebackIfCopy(O_arr);	
    Py_DECREF(M_arr);
    Py_DECREF(O_arr);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
gradientHist(PyObject *self, PyObject *args) {
    PyObject *arg1 = NULL, *arg2 = NULL, *arg3 = NULL;
    PyArrayObject *H_arr = NULL, *M_arr = NULL, *O_arr = NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!",   
        &PyArray_Type, &arg1, 
        &PyArray_Type, &arg2, 
        &PyArray_Type, &arg3))
        return NULL;


    M_arr = (PyArrayObject *) PyArray_FROM_OF(arg1, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    O_arr = (PyArrayObject *) PyArray_FROM_OF(arg2, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    H_arr = (PyArrayObject *) PyArray_FROM_OF(arg3, NPY_ARRAY_INOUT_FARRAY2);

    void *M = PyArray_DATA(M_arr);
    void *O = PyArray_DATA(O_arr);
    void *H = PyArray_DATA(H_arr);

    npy_intp *M_shape = PyArray_DIMS(M_arr);

    int type_M = PyArray_TYPE(M_arr);


    int h = (int) M_shape[0];
    int w = (int) M_shape[1];

    
    int binSize  = 4;
    int nOrients = 9;
    int softBin  = -1;
    int useHog   = 2;
    float clipHog  = 0.2f;


    fhog((float *) M, (float *) O, (float *) H, h, w, binSize, nOrients, softBin, clipHog);


    Py_DECREF(M_arr);
    Py_DECREF(O_arr);
    PyArray_ResolveWritebackIfCopy(H_arr);	
    Py_DECREF(H_arr);
    Py_INCREF(Py_None);
    return Py_None;
}


static struct PyMethodDef methods[] = {
        {"gradientMag", gradientMag, METH_VARARGS, "gradientMag"},
        {"gradientHist", gradientHist, METH_VARARGS, "gradientHist"},
        {NULL, NULL, 0, NULL}
};


/* Module structure */
static struct PyModuleDef fhogmodule = {
    PyModuleDef_HEAD_INIT,

    "python MATLAB fhog",           /* name of module */
    "A fhog module",  /* Doc string (may be NULL) */
    -1,                 /* Size of per-interpreter state or -1 */
    methods       /* Method table */
};

PyMODINIT_FUNC
PyInit_fhog(void) {
     /* IMPORTANT: this must be called */
    import_array();
    return PyModule_Create(&fhogmodule);
}
