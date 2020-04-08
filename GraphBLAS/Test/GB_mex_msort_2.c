//------------------------------------------------------------------------------
// GB_mex_msort_2: sort using GB_msort_2
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "[I,J] = GB_mex_msort_2 (I,J,nthreads)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    bool malloc_debug = GB_mx_get_global (true) ;

    // check inputs
    if (nargin != 3 || nargout != 2)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }
    if (!mxIsClass (pargin [0], "int64"))
    {
        mexErrMsgTxt ("I must be a int64 array") ;
    }
    if (!mxIsClass (pargin [1], "int64"))
    {
        mexErrMsgTxt ("J must be a int64 array") ;
    }

    int64_t *I = mxGetData (pargin [0]) ;
    int64_t n = (uint64_t) mxGetNumberOfElements (pargin [0]) ;

    int64_t *J = mxGetData (pargin [1]) ;
    if (n != (uint64_t) mxGetNumberOfElements (pargin [1])) 
    {
        mexErrMsgTxt ("I and J must be the same length") ;
    }

    int GET_SCALAR (2, int, nthreads, 1) ;
    nthreads = GB_MSORT_NTHREADS (nthreads) ;

    // make a copy of the input arrays
    pargout [0] = mxCreateNumericMatrix (n, 1, mxINT64_CLASS, mxREAL) ;
    int64_t *Iout = mxGetData (pargout [0]) ;
    memcpy (Iout, I, n * sizeof (int64_t)) ;

    pargout [1] = mxCreateNumericMatrix (n, 1, mxINT64_CLASS, mxREAL) ;
    int64_t *Jout = mxGetData (pargout [1]) ;
    memcpy (Jout, J, n * sizeof (int64_t)) ;

    // get workspace
    int64_t *Work_0 = NULL ;
    int64_t *Work_1 = NULL ;
    if (nthreads > 1)
    {
        Work_0 = mxMalloc ((n+1) * sizeof (int64_t)) ;
        Work_1 = mxMalloc ((n+1) * sizeof (int64_t)) ;
    }

    GB_MEX_TIC ;

    GB_msort_2 (Iout, Jout, Work_0, Work_1, n, nthreads) ;

    GB_MEX_TOC ;

    // free workspace
    if (nthreads > 1)
    {
        mxFree (Work_0) ;
        mxFree (Work_1) ;
    }

    GB_mx_put_global (true, 0) ;
}

