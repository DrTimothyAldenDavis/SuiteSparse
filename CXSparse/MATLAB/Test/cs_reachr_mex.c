// CXSparse/MATLAB/Test/cs_reachr_mex: reach of b in graph of L (recursive)
// CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs_mex.h"
/* find nonzero pattern of x=L\sparse(b).  L must be sparse and lower
 * triangular.  b must be a sparse vector. */

static
void dfsr (int64_t j, const cs_dl *L, int64_t *top, int64_t *xi, int64_t *w)
{
    int64_t p ;
    w [j] = 1 ;                                 /* mark node j */
    for (p = L->p [j] ; p < L->p [j+1] ; p++)   /* for each i in L(:,j) */
    {
        if (w [L->i [p]] != 1)                  /* if i is unmarked */
        {
            dfsr (L->i [p], L, top, xi, w) ;    /* start a dfs at i */
        }
    }
    xi [--(*top)] = j ;                         /* push j onto the stack */
}

/* w [0..n-1] == 0 on input, <= 1 on output.  size n */
static
int64_t reachr (const cs_dl *L, const cs_dl *B, int64_t *xi, int64_t *w)
{
    int64_t p, n = L->n ;
    int64_t top = n ;                            /* stack is empty */
    for (p = B->p [0] ; p < B->p [1] ; p++)     /* for each i in pattern of b */
    {
        if (w [B->i [p]] != 1)                  /* if i is unmarked */
        {
            dfsr (B->i [p], L, &top, xi, w) ;   /* start a dfs at i */
        }
    }
    return (top) ;                              /* return top of stack */
}

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
    int64_t i, j, top, *xi ;

    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: x = cs_reachr(L,b)") ;
    }

    /* get inputs */
    L = cs_dl_mex_get_sparse (&Lmatrix, 1, 1, pargin [0]) ;
    B = cs_dl_mex_get_sparse (&Bmatrix, 0, 1, pargin [1]) ;
    cs_mex_check (0, L->n, 1, 0, 1, 1, pargin [1]) ;

    xi = cs_dl_calloc (2*L->n, sizeof (int64_t)) ;

    top = reachr (L, B, xi, xi + L->n) ;

    pargout [0] = mxCreateDoubleMatrix (L->n - top, 1, mxREAL) ;
    x = mxGetPr (pargout [0]) ;
    for (j = 0, i = top ; i < L->n ; i++, j++) x [j] = xi [i] ;

    cs_dl_free (xi) ;
}
