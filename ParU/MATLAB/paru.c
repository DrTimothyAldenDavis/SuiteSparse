//------------------------------------------------------------------------------
// ParU/MATLAB/paru: MATLAB interface to ParU x=A\b
//------------------------------------------------------------------------------

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//------------------------------------------------------------------------------

// FIXME test the paru mexFunction on the Mac (intel and m1)

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
    "strategy",             // strategy used, symmetric or unsymmetric
    "ordering",             // ordering used
    "rcode",                // rough estimate of reciprocal condition number
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
                    // Auto: UMFPACK will select symmetric or unsymmetric
                    // strategy, as determined by the properties of the matrix.
                    // ParU will then select the strategy that UMFPACK
                    // selected.
                    Control.umfpack_strategy = UMFPACK_STRATEGY_AUTO ;
                    Control.paru_strategy = PARU_STRATEGY_AUTO ;
                }
                else if (strncmp (option, "unsymmetric", STRLEN) == 0)
                {
                    // both UMFPACK and ParU will use the unsymmetric strategy
                    Control.umfpack_strategy = UMFPACK_STRATEGY_UNSYMMETRIC ;
                    Control.paru_strategy = PARU_STRATEGY_UNSYMMETRIC ;
                }
                else if (strncmp (option, "symmetric", STRLEN) == 0)
                {
                    // both UMFPACK and ParU will use the symmetric strategy
                    Control.umfpack_strategy = UMFPACK_STRATEGY_SYMMETRIC ;
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
        "numeric factorization failed") ;

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

    // FIXME use ParU_get for these 2 results:
    int32_t paru_strategy = Sym->paru_strategy ;
    int32_t umfpack_ordering = Sym->umfpack_ordering ;
    double rcond = Num->rcond ;

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
        pargout [1] = mxCreateStructMatrix (1, 1, 5, stat_names) ;

        // analysis, factorization, and solve times:
        mxSetFieldByNumber (pargout [1], 0, 0, mxCreateDoubleScalar (t [0])) ;
        mxSetFieldByNumber (pargout [1], 0, 1, mxCreateDoubleScalar (t [1])) ;
        mxSetFieldByNumber (pargout [1], 0, 2, mxCreateDoubleScalar (t [2])) ;

        // UMFPACK and ParU strategy:
        bool symmetric = (paru_strategy == PARU_STRATEGY_SYMMETRIC) ;
        mxSetFieldByNumber (pargout [1], 0, 3,
            mxCreateString (symmetric ? "symmetric" : "unsymmetric")) ;

        // UMFPACK ordering used
        char *ordering ;
        switch (umfpack_ordering)
        {
            case UMFPACK_ORDERING_AMD:
                ordering = symmetric ? "amd(A+A')" : "colamd(A)" ;
                break ;

            case UMFPACK_ORDERING_METIS:
                ordering = symmetric ? "metis(A+A')" : "metis(A'*A)" ;
                break ;

            case UMFPACK_ORDERING_NONE:
                ordering = "none" ;
                break ;

            // These cases cannot occur.  Some of them can be specified on
            // input, with opts.ordering, but they are ordering strategies
            // that select amd, colamd, or metis.
            case UMFPACK_ORDERING_GIVEN:
            case UMFPACK_ORDERING_BEST:
            case UMFPACK_ORDERING_USER:
            case UMFPACK_ORDERING_METIS_GUARD:
            case UMFPACK_ORDERING_CHOLMOD:
            default:
                ordering = "undefined" ;
                break ;
        }
        mxSetFieldByNumber (pargout [1], 0, 4, mxCreateString (ordering)) ;

        // numeric factorization statistics:
        mxSetFieldByNumber (pargout [1], 0, 5, mxCreateDoubleScalar (rcond)) ;

        // FIXME: add nnz(L), nnz(U), flop count if available, ...
    }
}

