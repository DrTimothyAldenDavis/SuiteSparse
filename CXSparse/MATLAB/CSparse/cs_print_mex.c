// CXSparse/MATLAB/CSparse/cs_print_mex: print a sparse matrix
// CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs_mex.h"
/* cs_print: print the contents of a sparse matrix. */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    int64_t brief ;
    if (nargout > 0 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: cs_print(A,brief)") ;
    }
    brief = (nargin < 2) ? 0 : mxGetScalar (pargin [1]) ;   /* get brief */
    if (mxIsComplex (pargin [0]))
    {
#ifndef NCOMPLEX
        cs_cl Amatrix, *A ;
        A = cs_cl_mex_get_sparse (&Amatrix, 0, pargin [0]) ;    /* get A */
        cs_cl_print (A, brief) ;                            /* print A */
        cs_cl_free (A->x) ;
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl Amatrix, *A ;
        A = cs_dl_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;/* get A */
        cs_dl_print (A, brief) ;                               /* print A */
    }
}
