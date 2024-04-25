//------------------------------------------------------------------------------
// ParU/MATLAB/paru: MATLAB interface to ParU x=A\b
//------------------------------------------------------------------------------

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//------------------------------------------------------------------------------

// Usage:
//      x = paru (A, b)
//      [x,stats] = paru (A, b, opts)

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
    if ((method) != PARU_SUCCESS)                                   \
    {                                                               \
        ParU_C_FreeNumeric (&Num, &Control) ;                       \
        ParU_C_FreeSymbolic (&Sym, &Control) ;                      \
        mexErrMsgIdAndTxt ("ParU:error", "ParU: " error_message) ;  \
    }                                                               \
}

static const char *stat_names [ ] =
{
    "analysis_time",        // analysis time in seconds
    "factorize_time",       // factorization time in seconds
    "solve_time",           // solve time in seconds
} ;

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    cholmod_sparse Amatrix, *A = NULL ;
    cholmod_dense Bmatrix, *X = NULL, *B = NULL ;
    cholmod_common Common, *cm = NULL ;
    int64_t n ;
    ParU_C_Symbolic *Sym = NULL ;
    ParU_C_Numeric *Num = NULL ;

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
	mexErrMsgTxt ("usage: [x,stats] = part (A,b,opts)") ;
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
    if (nargin == 3 && !mxIsStruct (pargin [2]))
    {
    	mexErrMsgTxt ("3rd input must be a MATLAB struct")  ;
    }

    // initialize the ParU Control struct
    ParU_C_Control Control ;
    PARU_OK (ParU_C_Init_Control (&Control), "initialization failed") ;

    // change the default ordering to AMD/COLAMD
    Control.umfpack_ordering = UMFPACK_ORDERING_AMD ;

    // get the opts
    if (nargin == 3)
    {
        mxArray *field ;
        #define STRLEN 256
        char option [STRLEN+1] ;

        // tol: pivot tolerance
        if ((field = mxGetField (pargin [2], 0, "tol")) != NULL)
        {
            Control.piv_toler = mxGetScalar (field) ;
        }

        // diagtol: pivot tolerance for diagonal entries
        if ((field = mxGetField (pargin [2], 0, "diagtol")) != NULL)
        {
            Control.diag_toler = mxGetScalar (field) ;
        }

        // ordering: fill-reducing ordering method to use
        if ((field = mxGetField (pargin [2], 0, "ordering")) != NULL)
        {
            if (mxGetString (field, option, STRLEN) == 0)
            {
                option [STRLEN] = '\0' ;
                if (strncmp (option, "amd", STRLEN) == 0)
                {
                    Control.umfpack_ordering = UMFPACK_ORDERING_AMD ;
                }
                else if (strncmp (option, "cholmod", STRLEN) == 0)
                {
                    Control.umfpack_ordering = UMFPACK_ORDERING_CHOLMOD ;
                }
                else if (strncmp (option, "metis", STRLEN) == 0)
                {
                    Control.umfpack_ordering = UMFPACK_ORDERING_METIS ;
                }
                else if (strncmp (option, "metis_guard", STRLEN) == 0)
                {
                    Control.umfpack_ordering = UMFPACK_ORDERING_METIS_GUARD ;
                }
                else if (strncmp (option, "best", STRLEN) == 0)
                {
                    Control.umfpack_ordering = UMFPACK_ORDERING_BEST ;
                }
                else if (strncmp (option, "none", STRLEN) == 0)
                {
                    Control.umfpack_ordering = UMFPACK_ORDERING_NONE ;
                }
                else
                {
                    mexErrMsgIdAndTxt ("ParU:error",
                        "unrecognized opts.ordering: %s", option) ;
                }
            }
            else
            {
                mexErrMsgIdAndTxt ("ParU:error", "unrecognized opts.ordering") ;
            }
        }

        // prescale: whether or not to prescale the input matrix
        if ((field = mxGetField (pargin [2], 0, "prescale")) != NULL)
        {
            // 0: no scaling, 1: prescale each row by max abs value
            Control.prescale = (int) (mxGetScalar (field) != 0) ;
        }

        // strategy: both ParU and UMFPACK factorization strategy
        if ((field = mxGetField (pargin [2], 0, "strategy")) != NULL)
        {
            if (mxGetString (field, option, STRLEN) == 0)
            {
                option [STRLEN] = '\0' ;
                if (strncmp (option, "auto", STRLEN) == 0)
                {
                    Control.paru_strategy = PARU_STRATEGY_AUTO ;
                }
                else if (strncmp (option, "unsymmetric", STRLEN) == 0)
                {
                    // FIXME: also set UMFPACK strategy
                    Control.paru_strategy = PARU_STRATEGY_UNSYMMETRIC ;
                }
                else if (strncmp (option, "symmetric", STRLEN) == 0)
                {
                    Control.paru_strategy = PARU_STRATEGY_SYMMETRIC ;
                }
                else
                {
                    mexErrMsgIdAndTxt ("ParU:error",
                        "unrecognized opts.strategy: %s", option) ;
                }
            }
            else
            {
                mexErrMsgIdAndTxt ("ParU:error", "unrecognized opts.strategy") ;
            }
        }

        // FUTURE: opts.paru_strategy and opts.umfpack_strategy
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

    //--------------------------------------------------------------------------
    // change the memory manager to the ANSI C malloc/calloc/realloc/free
    //--------------------------------------------------------------------------

    // ParU needs a thread-safe memory manager; mxMalloc/mxFree is not
    // thread-safe.

    #undef malloc
    #undef calloc
    #undef realloc
    #undef free
    SuiteSparse_config_malloc_func_set (malloc) ;
    SuiteSparse_config_calloc_func_set (calloc) ;
    SuiteSparse_config_realloc_func_set (realloc) ;
    SuiteSparse_config_free_func_set (free) ;

    //--------------------------------------------------------------------------
    // x = A\b using ParU
    //--------------------------------------------------------------------------

    double t [3], t0, t1 ;
    if (nargout > 1)
    {
        t0 = SuiteSparse_time ( ) ;
    }

    PARU_OK (ParU_C_Analyze (A, &Sym, &Control), "symbolic analysis failed") ;

    if (nargout > 1)
    {
        t1 = SuiteSparse_time ( ) ;
        t [0] = t1 - t0 ;
        t0 = t1 ;
    }

    PARU_OK (ParU_C_Factorize (A, Sym, &Num, &Control),
        "factorization failed") ;

    if (nargout > 1)
    {
        t1 = SuiteSparse_time ( ) ;
        t [1] = t1 - t0 ;
        t0 = t1 ;
    }

    PARU_OK (ParU_C_Solve_AXB (Sym, Num, nrhs, (double *) B->x, (double *) X->x,
        &Control), "solve failed") ;

    if (nargout > 1)
    {
        t1 = SuiteSparse_time ( ) ;
        t [2] = t1 - t0 ;
        t0 = t1 ;
    }

    ParU_C_FreeNumeric (&Num, &Control) ;
    ParU_C_FreeSymbolic (&Sym, &Control) ;

    //--------------------------------------------------------------------------
    // set the memory manager back to mxMalloc/mxCalloc/mxRealloc/mxFree
    //--------------------------------------------------------------------------

    SuiteSparse_config_malloc_func_set (mxMalloc) ;
    SuiteSparse_config_calloc_func_set (mxCalloc) ;
    SuiteSparse_config_realloc_func_set (mxRealloc) ;
    SuiteSparse_config_free_func_set (mxFree) ;

    //--------------------------------------------------------------------------
    // free workspace and return solution to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = sputil2_put_dense (&X, mxDOUBLE_CLASS, cm) ;
    sputil2_free_sparse (&A, &Amatrix, A_xsize, cm) ;
    sputil2_free_dense  (&B, &Bmatrix, B_xsize, cm) ;
    cholmod_l_finish (cm) ;

    //--------------------------------------------------------------------------
    // return statistics to MATLAB, if requested
    //--------------------------------------------------------------------------

    if (nargout > 1)
    {
        pargout [1] = mxCreateStructMatrix (1, 1, 3, stat_names) ;
        mxSetFieldByNumber (pargout [1], 0, 0, mxCreateDoubleScalar (t [0])) ;
        mxSetFieldByNumber (pargout [1], 0, 1, mxCreateDoubleScalar (t [1])) ;
        mxSetFieldByNumber (pargout [1], 0, 2, mxCreateDoubleScalar (t [2])) ;
        // FIXME: add nnz(L), nnz(U), rcond, flop count if available, ...
    }

}

