//------------------------------------------------------------------------------
// ParU/MATLAB/paru: MATLAB interface to ParU x=A\b
//------------------------------------------------------------------------------

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//------------------------------------------------------------------------------

// Usage:
//      x = paru (A, b)
//      [x,stats] = paru (A, b, options)

#define BLAS_Intel10_64ilp
#include "sputil2.h"
#include "ParU.h"

#define OK(ok,error_message)                                        \
{                                                                   \
    if (!(ok))                                                      \
    {                                                               \
        mexErrMsgIdAndTxt ("ParU:error", "ParU: " error_message) ;  \
    }                                                               \
}

#define PARU_OK(method,error_message)                               \
{                                                                   \
    OK ((method) == PARU_SUCCESS, error_message) ;                  \
}

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    cholmod_sparse Amatrix, *A ;
    cholmod_dense Bmatrix, *X, *B ;
    cholmod_common Common, *cm ;
    int64_t n ;

    //--------------------------------------------------------------------------
    // start CHOLMOD
    //--------------------------------------------------------------------------

    SuiteSparse_start ( ) ;
    cm = &Common ;
    OK (cholmod_l_start (cm), "error initializing CHOLMOD") ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    if (nargout > 2 || nargin < 2 || nargin > 3)
    {
	mexErrMsgTxt ("usage: [x,stats] = part (A,b,options)") ;
    }
    n = mxGetM (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || (n != mxGetN (pargin [0])))
    {
    	mexErrMsgTxt ("A must be square and sparse") ;
    }
    if (n != mxGetM (pargin [1]))
    {
    	mexErrMsgTxt ("# of rows of A and B must match") ;
    }
    if (mxIsSparse (pargin [1]))
    {
    	mexErrMsgTxt ("B must be dense")  ;
    }

    // get sparse matrix A
    size_t A_xsize = 0 ;
    A = sputil2_get_sparse (pargin [0], 0, CHOLMOD_DOUBLE, &Amatrix,
        &A_xsize, cm) ;
    OK (A != NULL, "error getting A matrix") ;

    // get dense matrix B */
    size_t B_xsize = 0 ;
    B = sputil2_get_dense (pargin [1], CHOLMOD_DOUBLE, &Bmatrix,
        &B_xsize, cm) ;
    OK (B != NULL, "error getting B matrix") ;
    int64_t nrhs = B->ncol ;

    // create the solution X
    X = cholmod_l_allocate_dense (n, nrhs, n, CHOLMOD_DOUBLE + CHOLMOD_REAL,
        cm) ;
    OK (X != NULL, "error creating X matrix") ;

    // get the options (TODO)   FIXME NOW

    //--------------------------------------------------------------------------
    // x = A\b using ParU
    //--------------------------------------------------------------------------

    ParU_C_Control Control ;
    ParU_C_Symbolic *Sym = NULL ;
    ParU_C_Numeric *Num = NULL ;

    PARU_OK (ParU_C_Init_Control (&Control), "initialization failed") ;
    PARU_OK (ParU_C_Analyze (A, &Sym, &Control), "symbolic analysis failed") ;
    PARU_OK (ParU_C_Factorize (A, Sym, &Num, &Control), "factorization failed");
    PARU_OK (ParU_C_Solve_AXB (Sym, Num, nrhs, (double *) B->x, (double *) X->x,
        &Control), "solve failed") ;
    PARU_OK (ParU_C_FreeNumeric (&Num, &Control), "free numeric failed") ;
    PARU_OK (ParU_C_FreeSymbolic (&Sym, &Control), "free symbolic failed") ;

    //--------------------------------------------------------------------------
    // free workspace and return solution to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = sputil2_put_dense (&X, mxDOUBLE_CLASS, cm) ;
    // return statistics, if requested */
    if (nargout > 1)
    {
	// pargout [1] = ... TODO   FIXME NOW
    }
    sputil2_free_sparse (&A, &Amatrix, A_xsize, cm) ;
    sputil2_free_dense  (&B, &Bmatrix, B_xsize, cm) ;
    cholmod_l_finish (cm) ;
}

