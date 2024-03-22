#-------------------------------------------------------------------------------
# SPEX/Python/SPEXpy/spex_from_matrix_file.py: read matrix from file
#-------------------------------------------------------------------------------

# SPEX: (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
# Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

#------------------------------------------------------------------------------


import numpy as np
from scipy.sparse import coo_matrix

def spex_matrix_from_file(fname):
    #fname is the name of the file that contains matrix A

    ##--------------------------------------------------------------------------
    ## Load file data and store it in mat
    ##--------------------------------------------------------------------------
    mat = np.loadtxt(fname, delimiter=' ')

    ##--------------------------------------------------------------------------
    ## Splice mat to get the componets of matrix A and populate a scipy sparse matrix
    ##--------------------------------------------------------------------------
    data = mat[1:,2]
    row = mat[1:,0].astype(int)
    col = mat[1:,1].astype(int)
    n = mat[0][0].astype(int)
    m = mat[0][1].astype(int)
    ## The matrix in the file is in triplet form
    triplet=coo_matrix((data, (row, col)),shape=(n, m))
    ## Change the triplet matrix into a csc matrix
    A = triplet.tocsc()

    return A
