// CXSparse/MATLAB/Test/cs_reach_mex: reach of b in the graph of L
// CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs_mex.h"
/* find nonzero pattern of x=L\sparse(b).  L must be sparse and lower
 * triangular.  b must be a sparse vector. */

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs_dl Lmatrix, Bmatrix, *L, *B ;
    double *x ;
    int64_t k, i, j, top, *xi, *perm ;

    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: x = cs_reach(L,b)") ;
    }

    /* get inputs */
    L = cs_dl_mex_get_sparse (&Lmatrix, 1, 1, pargin [0]) ;
    B = cs_dl_mex_get_sparse (&Bmatrix, 0, 1, pargin [1]) ;
    cs_mex_check (0, L->n, 1, 0, 1, 1, pargin [1]) ;
    perm = cs_dl_malloc (L->n, sizeof (int64_t)) ;
    for (k = 0 ; k < L->n ; k++) perm [k] = k ;

    xi = cs_dl_calloc (3*L->n, sizeof (int64_t)) ;

    top = cs_dl_reach (L, B, 0, xi, perm) ;

    pargout [0] = mxCreateDoubleMatrix (L->n - top, 1, mxREAL) ;
    x = mxGetPr (pargout [0]) ;
    for (j = 0, i = top ; i < L->n ; i++, j++) x [j] = xi [i] ;

    cs_dl_free (xi) ;
    cs_dl_free (perm) ;
}
