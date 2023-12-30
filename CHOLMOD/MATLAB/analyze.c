//------------------------------------------------------------------------------
// CHOLMOD/MATLAB/analyze: MATLAB interface to CHOLMOD symbolic analysis
//------------------------------------------------------------------------------

// CHOLMOD/MATLAB Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// MATLAB(tm) is a Trademark of The MathWorks, Inc.

// Order a matrix and then analyze it, using CHOLMOD's best-effort ordering.
// Returns the count of the number of nonzeros in each column of L for the
// permuted matrix A.
//
// Usage:
//
//      [p count] = analyze (A)         orders A, using just tril(A)
//      [p count] = analyze (A,'sym')   orders A, using just tril(A)
//      [p count] = analyze (A,'row')   orders A*A'
//      [p count] = analyze (A,'col')   orders A'*A
//
// with an optional 3rd parameter:
//
//      [p count] = analyze (A,'sym',k) orders A, using just tril(A)
//      [p count] = analyze (A,'row',k) orders A*A'
//      [p count] = analyze (A,'col',k) orders A'*A
//
//      k=0 is the default.  k != 0 selects the ordering strategy.
//
// See analyze.m for more details.

#include "sputil2.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double dummy = 0 ;
    cholmod_factor *L ;
    cholmod_sparse *A, Amatrix ;
    cholmod_common Common, *cm ;
    int64_t transpose, c ;
    char buf [LEN] ;

    //--------------------------------------------------------------------------
    // start CHOLMOD and set defaults
    //--------------------------------------------------------------------------

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil2_config (SPUMONI, cm) ;

    // only do the simplicial analysis (L->Perm and L->ColCount)
    cm->supernodal = CHOLMOD_SIMPLICIAL ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    if (nargout > 2 || nargin < 1 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: [p count] = analyze (A, mode)") ;
    }
    if (nargin == 3)
    {
        cm->nmethods = mxGetScalar (pargin [2]) ;
        if (cm->nmethods == -1 || cm->nmethods == 1 || cm->nmethods == 2)
        {
            // use AMD only
            cm->nmethods = 1 ;
            cm->method [0].ordering = CHOLMOD_AMD ;
            cm->postorder = TRUE ;
        }
        else if (cm->nmethods == -2)
        {
            // use METIS only
            cm->nmethods = 1 ;
            cm->method [0].ordering = CHOLMOD_METIS ;
            cm->postorder = TRUE ;
        }
        else if (cm->nmethods == -3)
        {
            // use NESDIS only
            cm->nmethods = 1 ;
            cm->method [0].ordering = CHOLMOD_NESDIS ;
            cm->postorder = TRUE ;
        }
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
            mexErrMsgTxt ("analyze: unrecognized mode") ;
        }
    }

    if (A->stype && A->nrow != A->ncol)
    {
        mexErrMsgTxt ("analyze: A must be square") ;
    }

    //--------------------------------------------------------------------------
    // analyze and order the matrix
    //--------------------------------------------------------------------------

    if (transpose)
    {
        // C = A', and then order C*C'
        cholmod_sparse *C = cholmod_l_transpose (A, 0, cm) ;
        L = cholmod_l_analyze (C, cm) ;
        cholmod_l_free_sparse (&C, cm) ;
    }
    else
    {
        // order A or A*A'
        L = cholmod_l_analyze (A, cm) ;
    }

    //--------------------------------------------------------------------------
    // return results and free workspace
    //--------------------------------------------------------------------------

    sputil2_free_sparse (&A, &Amatrix, 0, cm) ;
    pargout [0] = sputil2_put_int (L->Perm, L->n, 1) ;
    if (nargout > 1)
    {
        pargout [1] = sputil2_put_int (L->ColCount, L->n, 0) ;
    }
    cholmod_l_free_factor (&L, cm) ;
    cholmod_l_finish (cm) ;
    if (SPUMONI > 0) cholmod_l_print_common (" ", cm) ;
}

