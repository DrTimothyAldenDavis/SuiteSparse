#-------------------------------------------------------------------------------
# SPEX/Python/SPEXpy/lu_backslash.py: solve Ax=b using LU factorization
#-------------------------------------------------------------------------------

# SPEX: (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
# Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

#------------------------------------------------------------------------------

from .Options import Options
from .SPEX_error import *
from .spex_connect import spex_connect

import scipy
from scipy.sparse import isspmatrix, isspmatrix_csc, linalg

def lu_backslash( A, b, options=Options('double', 'colamd')):
    ## A is a scipy.sparse(data must be float64) #technically it only needs to be numerical
    ## b is a numpy.array (data must be float64)
    ## options is a dictionary that specifies what tipe the solution should be, this by default is double

    ##--------------------------------------------------------------------------
    ## Verify inputs
    ##--------------------------------------------------------------------------
    if not isspmatrix(A):
        raise SPEX_error(determine_error(3))
    ## If the sparse input matrix is not in csc form, convert it into csc form
    if not isspmatrix_csc(A):
        A.tocsc()
    # Check input shape
    if A.shape[1]!=b.shape[0]:
        raise SPEX_error(determine_error(3))

    if options.ordering==None:
        options.default_lu()

    ##--------------------------------------------------------------------------
    ## Call SPEX
    ##--------------------------------------------------------------------------
    x=spex_connect(A,b,options.order(),options.charOut(),2) #2 calls lu

    return x

