//------------------------------------------------------------------------------
// GB_mex_complex: convert a real matrix into a complex one
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// If A is real, C has an all-zero imaginary part.
// If A is complex, then C = A.

// This is a sparse version of the MATLAB 'complex' function, which does not
// work for sparse matrices.  This is self-contained and does not use GraphBLAS
// at all.

#include "mex.h"
#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define USAGE "C = GB_mex_complex (A)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    // check inputs
    if (nargout > 1 || nargin != 1)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get the input matrix
    const mxArray *A = pargin [0] ;
    if (!mxIsSparse (A))
    {
        mexErrMsgTxt ("A must be sparse") ;
    }
    if (mxIsLogical (A))
    {
        mexErrMsgTxt ("A must be double or double complex") ;
    }

    int64_t *Ap = (int64_t *) mxGetJc (A) ;
    int64_t *Ai = (int64_t *) mxGetIr (A) ;
    double  *Ax = NULL ;
    if (mxIsComplex (A))
    {
        Ax = (double *) mxGetComplexDoubles (pargin [0]) ;
    }
    else
    {
        Ax = (double *) mxGetDoubles (pargin [0]) ;
    }

    int64_t m = mxGetM (A) ;
    int64_t n = mxGetN (A) ;
    int64_t anz = Ap [n] ;

    // create the output matrix
    pargout [0] = mxCreateSparse (m, n, anz+1, mxCOMPLEX) ;
    mxArray *C = pargout [0] ;
    int64_t *Cp = (int64_t *) mxGetJc (C) ;
    int64_t *Ci = (int64_t *) mxGetIr (C) ;
    double  *Cx = (double  *) mxGetComplexDoubles (C) ;

    // copy the pattern of A into C
    memcpy (Cp, Ap, (n+1) * sizeof (int64_t)) ;
    memcpy (Ci, Ai, anz   * sizeof (int64_t)) ;

    // copy the values of A into C
    if (mxIsComplex (A))
    {
        memcpy (Cx, Ax, anz * 2 * sizeof (double)) ;
    }
    else
    {
        for (int64_t k = 0 ; k < anz ; k++)
        {
            Cx [2*k  ] = Ax [k] ;
            Cx [2*k+1] = 0 ;
        }
    }
}

