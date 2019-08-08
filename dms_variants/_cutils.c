// Fast C versions of some functions in utils
// Written by Jesse Bloom.
//
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


static PyObject *
reverse_complement(PyObject *self, PyObject *args)
{
    // define variables
    PyObject *py_rc;
    const char *s;
    size_t slen, i;

    // parse arguments
    if (! PyArg_ParseTuple(args, "s", &s)) {
        return NULL;
    }
    slen = strlen(s);

    // build up new string
    char *rc = PyMem_New(char, slen + 1);
    if (rc == NULL) {
        PyErr_SetString(PyExc_MemoryError, "cannot allocate rc");
        return NULL;
    }
    rc[slen] = '\0'; // string termination character
    for (i = 0; i < slen; i++) {
        switch (s[slen - 1 - i]) {
            case 'A' : rc[i] = 'T';
                       break;
            case 'C' : rc[i] = 'G';
                       break;
            case 'G' : rc[i] = 'C';
                       break;
            case 'T' : rc[i] = 'A';
                       break;
            case 'N' : rc[i] = 'N';
                       break;
            default : PyErr_SetString(PyExc_ValueError, "invalid nt");
                      return NULL;
        }
    }
    py_rc = PyUnicode_FromString(rc);
    PyMem_Del(rc);
    return py_rc;
}


static PyMethodDef cutilsMethods[] = {
    {"reverse_complement", reverse_complement, METH_VARARGS,
     "Fast version of `utils.reverse_complement`."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cutilsmodule = {
    PyModuleDef_HEAD_INIT,
    "_cutils",
    "Fast implementations of some functions in `utils`.",
    -1,
    cutilsMethods
};

PyMODINIT_FUNC
PyInit__cutils(void)
{
    return PyModule_Create(&cutilsmodule);
}
