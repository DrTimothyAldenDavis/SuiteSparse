#-------------------------------------------------------------------------------
# SPEX/Python/SPEXpy/backslash.py: solve Ax=b
#-------------------------------------------------------------------------------

# SPEX: (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
# Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

#------------------------------------------------------------------------------

from .Options import Options
from .SPEX_error import *
from .spex_connect import spex_connect_rank
import scipy
from scipy.sparse import isspmatrix, isspmatrix_csc

def rank(A):
    ## Verify inputs
    if not isspmatrix(A):
        raise SPEX_error(determine_error(3))

    ## Convert to csc if needed (must assign back to A!)
    if not isspmatrix_csc(A):
        A = A.tocsc()

    ## Call the ctypes connection
    r = spex_connect_rank(A)

    return r

