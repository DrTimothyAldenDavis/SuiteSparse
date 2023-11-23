//------------------------------------------------------------------------------
// CHOLMOD/MATLAB/bisect: MATLAB interface to CHOLMOD graph bisection
//------------------------------------------------------------------------------

// CHOLMOD/MATLAB Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// MATLAB(tm) is a Trademark of The MathWorks, Inc.
// METIS is Copyrighted by G. Karypis

// Find an node separator of an undirected graph, using
// METIS_ComputeVertexSeparator.
//
// Usage:
//
//      s = bisect (A)          bisects A, uses tril(A)
//      s = bisect (A, 'sym')   bisects A, uses tril(A)
//      s = bisect (A, 'row')   bisects A*A'
//      s = bisect (A, 'col')   bisects A'*A
//
// Node i of the graph is in the left graph if s(i)=0, the right graph if
// s(i)=1, and in the separator if s(i)=2.
//
// Requirse METIS and the CHOLMOD Partition Module.

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
    int64_t *Partition ;
    cholmod_sparse *A, Amatrix, *C ;
    cholmod_common Common, *cm ;
    int64_t n, transpose, c ;
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

    if (nargout > 1 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: p = bisect (A, mode)") ;
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
        c = buf [0];
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
            mexErrMsgTxt ("bisect: p=bisect(A,mode) ; unrecognized mode") ;
        }
    }

    if (A->stype && A->nrow != A->ncol)
    {
        mexErrMsgTxt ("bisect: A must be square") ;
    }

    //--------------------------------------------------------------------------
    // bisect with CHOLMOD's interface to METIS_ComputeVertexSeparator
    //--------------------------------------------------------------------------

    bool ok ;
    if (transpose)
    {
        // C = A', and then bisect C*C'
        cholmod_sparse *C = cholmod_l_transpose (A, 0, cm) ;
        n = C->nrow ;
        Partition = cholmod_l_malloc (n, sizeof (int64_t), cm) ;
        ok = (cholmod_l_bisect (C, NULL, 0, TRUE, Partition, cm) >= 0) ;
        cholmod_l_free_sparse (&C, cm) ;
    }
    else
    {
        // biset A or A*A'
        n = A->nrow ;
        Partition = cholmod_l_malloc (n, sizeof (int64_t), cm) ;
        ok = (cholmod_l_bisect (A, NULL, 0, TRUE, Partition, cm) >= 0) ;
    }

    //--------------------------------------------------------------------------
    // return results and free workspace
    //--------------------------------------------------------------------------

    sputil2_free_sparse (&A, &Amatrix, 0, cm) ;
    if (!ok) mexErrMsgTxt ("bisect failed") ;
    pargout [0] = sputil2_put_int (Partition, n, 0) ;
    cholmod_l_free (n, sizeof (int64_t), Partition, cm) ;
    cholmod_l_finish (cm) ;
    if (SPUMONI > 0) cholmod_l_print_common (" ", cm) ;
#else
    mexErrMsgTxt ("METIS and the CHOLMOD Partition Module not installed\n") ;
#endif
}

