//------------------------------------------------------------------------------
// CHOLMOD/MATLAB/mread: read a matrix in Matrix Market format
//------------------------------------------------------------------------------

// CHOLMOD/MATLAB Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// [A Z] = mread (filename, prefer_binary)
//
// Read a sparse or dense matrix from a file in Matrix Market format.
//
// All MatrixMarket formats are supported.
// The Matrix Market "integer" format is converted into real, but the values
// are preserved.  The "pattern" format is converted into real.  If a pattern
// matrix is unsymmetric, all of its values are equal to one.  If a pattern is
// symmetric, the kth diagonal entry is set to one plus the number of
// off-diagonal nonzeros in row/column k, and off-diagonal entries are set to
// -1.
//
// Explicit zero entries are returned as the binary pattern of the matrix Z.

#include "sputil2.h"

// maximum file length
#define MAXLEN 1030

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    void *G ;
    cholmod_dense *X = NULL ;
    cholmod_sparse *A = NULL, *Z = NULL ;
    cholmod_common Common, *cm ;
    int64_t *Ap = NULL, *Ai ;
    double *Ax, *Az = NULL ;
    char filename [MAXLEN] ;
    int64_t nz, k, is_complex = FALSE, nrow = 0, ncol = 0 ;
    int mtype ;

    //--------------------------------------------------------------------------
    // start CHOLMOD and set parameters
    //--------------------------------------------------------------------------

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil2_config (SPUMONI, cm) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    if (nargin < 1 || nargin > 2 || nargout > 2)
    {
        mexErrMsgTxt ("usage: [A Z] = mread (filename, prefer_binary)") ;
    }
    if (!mxIsChar (pargin [0]))
    {
        mexErrMsgTxt ("mread requires a filename") ;
    }
    mxGetString (pargin [0], filename, MAXLEN) ;
    sputil2_file = fopen (filename, "r") ;
    if (sputil2_file == NULL)
    {
        mexErrMsgTxt ("cannot open file") ;
    }
    if (nargin > 1)
    {
        cm->prefer_binary = (mxGetScalar (pargin [1]) != 0) ;
    }

    //--------------------------------------------------------------------------
    // read the matrix, as either a dense or sparse matrix
    //--------------------------------------------------------------------------

    G = cholmod_l_read_matrix (sputil2_file, 1, &mtype, cm) ;
    fclose (sputil2_file) ;
    sputil2_file = NULL ;
    if (G == NULL)
    {
        mexErrMsgTxt ("could not read file") ;
    }

    // get the specific matrix (A or X), and change to complex if needed
    if (mtype == CHOLMOD_SPARSE)
    {
        A = (cholmod_sparse *) G ;
        nrow = A->nrow ;
        ncol = A->ncol ;
        Ap = A->p ;
        Ai = A->i ;
        if (A->xtype == CHOLMOD_ZOMPLEX)
        {
            // if complex, ensure A is complex, not zomplex
            cholmod_l_sparse_xtype (CHOLMOD_COMPLEX, A, cm) ;
        }
        is_complex = (A->xtype == CHOLMOD_COMPLEX) ;
        Ax = A->x ;
    }
    else if (mtype == CHOLMOD_DENSE)
    {
        X = (cholmod_dense *) G ;
        nrow = X->nrow ;
        ncol = X->ncol ;
        if (X->xtype == CHOLMOD_ZOMPLEX)
        {
            // if complex, ensure X is complex, not zomplex
            cholmod_l_dense_xtype (CHOLMOD_COMPLEX, X, cm) ;
        }
        is_complex = (X->xtype == CHOLMOD_COMPLEX) ;
        Ax = X->x ;
    }
    else
    {
        mexErrMsgTxt ("invalid file") ;
    }

    //--------------------------------------------------------------------------
    // if requested, extract the zero entries and place them in Z
    //--------------------------------------------------------------------------

    if (nargout > 1)
    {
        if (mtype == CHOLMOD_SPARSE)
        {
            // A is a sparse real/zomplex double matrix
            Z = sputil2_extract_zeros (A, cm) ;
        }
        else
        {
            // input is full; just return an empty Z matrix
            Z = cholmod_l_spzeros (nrow, ncol, 0, CHOLMOD_REAL, cm) ;
        }
    }

    //--------------------------------------------------------------------------
    // change a complex matrix to real if its imaginary part is all zero
    //--------------------------------------------------------------------------

    if (is_complex)
    {
        if (mtype == CHOLMOD_SPARSE)
        {
            nz = Ap [ncol] ;
        }
        else
        {
            nz = nrow * ncol ;
        }
        bool allzero = true ;
        for (k = 0 ; k < nz ; k++)
        {
            if (Ax [2*k+1] != 0)
            {
                allzero = false ;
                break ;
            }
        }
        if (allzero)
        {
            // discard the all-zero imaginary part
            if (mtype == CHOLMOD_SPARSE)
            {
                cholmod_l_sparse_xtype (CHOLMOD_REAL, A, cm) ;
            }
            else
            {
                cholmod_l_dense_xtype (CHOLMOD_REAL, X, cm) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // return results to MATLAB
    //--------------------------------------------------------------------------

    if (mtype == CHOLMOD_SPARSE)
    {
        // drop explicit zeros from A; their pattern is kept in Z
        pargout [0] = sputil2_put_sparse (&A, mxDOUBLE_CLASS,
            /* drop explicit zeros */ true, cm) ;
    }
    else
    {
        pargout [0] = sputil2_put_dense (&X, mxDOUBLE_CLASS, cm) ;
    }
    if (nargout > 1)
    {
        pargout [1] = sputil2_put_sparse (&Z, mxDOUBLE_CLASS,
            /* Z is binary so it has no zeros to drop */ false, cm) ;
    }

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    cholmod_l_finish (cm) ;
    if (SPUMONI > 0) cholmod_l_print_common (" ", cm) ;
}

