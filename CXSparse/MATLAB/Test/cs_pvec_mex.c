// CXSparse/MATLAB/Test/cs_pvec_mex: permute a dense vector
// CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs_mex.h"
/* x = b(p) */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    int64_t n, k, *p ;
    double *xx ;
    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: x = cs_pvec(b,p)") ;
    }
    n = mxGetNumberOfElements (pargin [0]) ;
    if (n != mxGetNumberOfElements (pargin [1]))
    {
        mexErrMsgTxt ("b or p wrong size") ;
    }

    xx = mxGetPr (pargin [1]) ;
    p = cs_dl_malloc (n, sizeof (int64_t)) ;
    for (k = 0 ; k < n ; k++) p [k] = xx [k] - 1 ;

    if (mxIsComplex (pargin [0]))
    {
#ifdef NCOMPLEX
        mexErrMsgTxt ("complex case not supported") ;
#else
        cs_complex_t *x, *b ;
        b = cs_cl_mex_get_double (n, pargin [0]) ;
        x = cs_dl_malloc (n, sizeof (cs_complex_t)) ;
        cs_cl_pvec (p, b, x, n) ;
        pargout [0] = cs_cl_mex_put_double (n, x) ;
        cs_cl_free (b) ;       /* free copy of complex values */
#endif
    }
    else
    {
        double *x, *b ;
        b = cs_dl_mex_get_double (n, pargin [0]) ;
        pargout [0] = mxCreateDoubleMatrix (n, 1, mxREAL) ;
        x = mxGetPr (pargout [0]) ;
        cs_dl_pvec (p, b, x, n) ;
    }
    cs_dl_free (p) ;
}
