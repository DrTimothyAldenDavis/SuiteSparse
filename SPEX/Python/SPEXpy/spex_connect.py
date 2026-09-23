#-------------------------------------------------------------------------------
# SPEX/Python/SPEX/spex_connect.py: link SPEX to use in Python
#-------------------------------------------------------------------------------

# SPEX: (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
# Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

#------------------------------------------------------------------------------

import ctypes
import numpy as np
import sys
from numpy.ctypeslib import ndpointer
from .SPEX_error import *

def spex_connect( A, b, order, charOut, algorithm ):
    ## A is a scipy.sparse.csc_matrix (data must be float64) #technically it only needs to be numerical
    ## b is a numpy.array (data must be float64)

    ##--------------------------------------------------------------------------
    ## Load the library with the "C bridge code"
    ##--------------------------------------------------------------------------
    # Check if we are on a Mac ('darwin') or Linux/Windows
    ext = '.dylib' if sys.platform == 'darwin' else '.so'
    lib_path = f'../build/libspexpython{ext}'

    lib = ctypes.CDLL(lib_path)
    c_backslash = lib.spex_python

    ##--------------------------------------------------------------------------
    ## Specify the parameter types and return type of the C function
    ##--------------------------------------------------------------------------
    c_backslash.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                            ndpointer(dtype=np.int64, ndim=1, flags=None),
                            ndpointer(dtype=np.int64, ndim=1, flags=None),
                            ndpointer(dtype=np.float64, ndim=1, flags=None),
                            ndpointer(dtype=np.float64, ndim=1, flags=None),
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_bool]
    c_backslash.restype = ctypes.c_int

    m,n=A.shape #number of columns/rows of A

    x_v = (ctypes.c_void_p*n)()

    ##--------------------------------------------------------------------------
    ## Solve Ax=b using REF Sparse  Factorization
    ##--------------------------------------------------------------------------
    ok=c_backslash(x_v,
                A.indptr.astype(np.int64), #without the cast it would be int32 and it would not be compatible with the C method
                A.indices.astype(np.int64),
                A.data.astype(np.float64),
                b,
                m,
                n,
                A.nnz,
                order,
                algorithm,
                charOut)

    if ok!=0:
        raise SPEX_error(determine_error(ok))

    ##--------------------------------------------------------------------------
    ## Cast solution into correct type (string or double)
    ##--------------------------------------------------------------------------
    if charOut:
        val = ctypes.cast(x_v, ctypes.POINTER(ctypes.c_char_p))
        x=[]
        for i in range(n):
            x.append(val[i])
    else:
        #x = ctypes.cast(x_v, ctypes.POINTER(ctypes.c_double))
        x=[]
        for i in range(n):
            val=ctypes.cast(x_v[i], ctypes.POINTER(ctypes.c_double))
            x.append(val[0]) ##this can also be changed to be a numpy array instead of a list

    return np.array(x)

def spex_connect_rank(A):
    ## Load the library
    # Check if we are on a Mac ('darwin') or Linux/Windows
    ext = '.dylib' if sys.platform == 'darwin' else '.so'
    lib_path = f'../build/libspexpython{ext}'

    lib = ctypes.CDLL(lib_path)
    c_rank = lib.spex_python_rank

    ## Specify parameter types
    c_rank.argtypes = [
        ctypes.POINTER(ctypes.c_int64),                  # rank_out
        ndpointer(dtype=np.int64, ndim=1, flags=None),   # Ap
        ndpointer(dtype=np.int64, ndim=1, flags=None),   # Ai
        ndpointer(dtype=np.float64, ndim=1, flags=None), # Ax
        ctypes.c_int,                                    # m
        ctypes.c_int,                                    # n
        ctypes.c_int                                     # nz
        ]
    c_rank.restype = ctypes.c_int

    m,n = A.shape
    rank_out = ctypes.c_int64(0) # Prepare a C integer to hold the answer

    ## Call the C function
    ok = c_rank(
        ctypes.byref(rank_out),
        A.indptr.astype(np.int64),
        A.indices.astype(np.int64),
        A.data.astype(np.float64),
        m, n, A.nnz
    )

    if ok != 0:
        raise SPEX_error(determine_error(ok))

    return rank_out.value
