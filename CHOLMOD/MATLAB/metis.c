//------------------------------------------------------------------------------
// CHOLMOD/MATLAB/metis: MATLAB interface to CHOLMOD nested dissection via METIS
//------------------------------------------------------------------------------

// CHOLMOD/MATLAB Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Nested dissection using METIS_NodeND
//
// Usage:
//
//      p = metis (A)           orders A, using just tril(A)
//      p = metis (A,'sym')     orders A, using just tril(A)
//      p = metis (A,'row')     orders A*A'
//      p = metis (A,'col')     orders A'*A
//
// METIS_NodeND's ordering is followed by CHOLMOD's etree or column etree
// postordering.  Requires METIS and the CHOLMOD Partition Module.

#include "sputil2.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
#ifndef NPARTITION
    double dummy = 0 ;
    int64_t *Perm ;
    cholmod_sparse *A, Amatrix ;
    cholmod_common Common, *cm ;
    int64_t n, transpose, c, postorder ;
    char buf [LEN] ;

    //--------------------------------------------------------------------------
    // start CHOLMOD and set defaults
    //--------------------------------------------------------------------------

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil2_config (SPUMONI, cm) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    if (nargout > 1 || nargin < 1 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: p = metis (A, mode)") ;
    }

    //--------------------------------------------------------------------------
    // get input matrix A
    //--------------------------------------------------------------------------

    A = sputil2_get_sparse_pattern (pargin [0], CHOLMOD_DOUBLE, &Amatrix, cm) ;

    //--------------------------------------------------------------------------
    // get A->stype, default is to use tril(A)
    //--------------------------------------------------------------------------

    A->stype = -1 ;
    transpose = FALSE ;

    if (nargin > 1)
    {
        buf [0] = '\0' ;
        if (mxIsChar (pargin [1]))
        {
            mxGetString (pargin [1], buf, LEN) ;
        }
        c = buf [0] ;
        if (tolower (c) == 'r')
        {
            // unsymmetric case (A*A') if string starts with 'r'
            transpose = FALSE ;
            A->stype = 0 ;
        }
        else if (tolower (c) == 'c')
        {
            // unsymmetric case (A'*A) if string starts with 'c'
            transpose = TRUE ;
            A->stype = 0 ;
        }
        else if (tolower (c) == 's')
        {
            // symmetric case (A) if string starts with 's'
            transpose = FALSE ;
            A->stype = -1 ;
        }
        else
        {
            mexErrMsgTxt ("metis: p=metis(A,mode) ; unrecognized mode") ;
        }
    }

    if (A->stype && A->nrow != A->ncol)
    {
        mexErrMsgTxt ("metis: A must be square") ;
    }

    //--------------------------------------------------------------------------
    // ordering A, AA', or A'A with METIS
    //--------------------------------------------------------------------------

    postorder = (nargin < 3) ;
    int ok ;

    if (transpose)
    {
        // C = A', and then order C*C' with METIS
        cholmod_sparse *C = cholmod_l_transpose (A, 0, cm) ;
        n = C->nrow ;
        Perm = cholmod_l_malloc (n, sizeof (int64_t), cm) ;
        ok = cholmod_l_metis (C, NULL, 0, postorder, Perm, cm) ;
        cholmod_l_free_sparse (&C, cm) ;
    }
    else
    {
        // order A or A*A' with METIS
        n = A->nrow ;
        Perm = cholmod_l_malloc (n, sizeof (int64_t), cm) ;
        ok = cholmod_l_metis (A, NULL, 0, postorder, Perm, cm) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return results
    //--------------------------------------------------------------------------

    sputil2_free_sparse (&A, &Amatrix, 0, cm) ;
    if (!ok) mexErrMsgTxt ("metis failed") ;
    pargout [0] = sputil2_put_int (Perm, n, 1) ;
    cholmod_l_free (n, sizeof (int64_t), Perm, cm) ;
    cholmod_l_finish (cm) ;
    if (SPUMONI > 0) cholmod_l_print_common (" ", cm) ;
#else
    mexErrMsgTxt ("METIS and the CHOLMOD Partition Module not installed\n") ;
#endif
}

