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

// With additional outputs for P (row permutations), Q (column permutations),
// and R (row scaling vector), for experimental testing only, compiled with
// -DDEVELOPER.  P and Q are returned as 0-based int64 vectors.
//      [x,stats,P,Q,R] = paru (A, b, opts)

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
        ParU_C_FreeNumeric (&Num, Control) ;                        \
        ParU_C_FreeSymbolic (&Sym, Control) ;                       \
        ParU_C_FreeControl (&Control) ;                             \
        mexErrMsgIdAndTxt ("ParU:error", "ParU: " error_message) ;  \
    }                                                               \
}

static const char *stat_names [ ] =
{
    "analysis_time",        //  0: analysis time in seconds
    "factorization_time",   //  1: factorization time in seconds
    "solve_time",           //  2: solve time in seconds
    "strategy_used",        //  3: strategy used, symmetric or unsymmetric
    "ordering_used",        //  4: ordering used
    "factorization_flop_count", //  5: flop count for LU factorization
    "lnz",                  //  6: nnz (L)
    "unz",                  //  7: nnz (U)
    "rcond",                //  8: rough estimate of reciprocal condition number
    "blas",                 //  9: BLAS library used
    "front_tree_tasking",   // 10: frontal tree task: sequential or parallel
    "openmp",               // 11: ParU using OpenMP or not
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
    ParU_C_Symbolic Sym = NULL ;
    ParU_C_Numeric Num = NULL ;
    ParU_C_Control Control = NULL ;

    //--------------------------------------------------------------------------
    // start CHOLMOD
    //--------------------------------------------------------------------------

    SuiteSparse_start ( ) ;
    cm = &Common ;
    OK (cholmod_l_start (cm), "error initializing CHOLMOD") ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    #ifdef DEVELOPER
    #define MAX_NARGOUT 5
    #else
    #define MAX_NARGOUT 2
    #endif
    if (nargout > MAX_NARGOUT || nargin < 2 || nargin > 3)
    {
        mexErrMsgTxt ("usage: [x,stats] = paru (A,b,opts)") ;
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
    // allocate result for P, Q, and R (developer only)
    //--------------------------------------------------------------------------

    #ifdef DEVELOPER
    mwSize dims [2] ;
    dims [0] = n ;
    dims [1] = 1 ;
    int64_t *P = NULL, *Q = NULL ;
    double *R = NULL ;
    if (nargout > 2)
    {
        pargout [2] = mxCreateNumericArray (2, dims, mxINT64_CLASS, mxREAL) ;
        P = mxGetInt64s (pargout [2]) ;
    }
    if (nargout > 3)
    {
        pargout [3] = mxCreateNumericArray (2, dims, mxINT64_CLASS, mxREAL) ;
        Q = mxGetInt64s (pargout [3]) ;
    }
    if (nargout > 4)
    {
        pargout [4] = mxCreateNumericArray (2, dims, mxDOUBLE_CLASS, mxREAL) ;
        R = mxGetDoubles (pargout [4]) ;
    }
    #endif

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
    // initialize the ParU Control struct
    //--------------------------------------------------------------------------

    PARU_OK (ParU_C_InitControl (&Control), "initialization failed") ;

    // change the default ordering to AMD/COLAMD
    PARU_OK (ParU_C_Set_Control_INT64 (PARU_CONTROL_ORDERING,
        PARU_ORDERING_AMD, Control), "opts failed") ;

    // get the opts
    if (nargin == 3)
    {
        mxArray *field ;
        #define STRLEN 256
        char option [STRLEN+1] ;

        // tol: pivot tolerance
        if ((field = mxGetField (pargin [2], 0, "tol")) != NULL)
        {
            PARU_OK (ParU_C_Set_Control_FP64 (PARU_CONTROL_PIVOT_TOLERANCE,
                (double) mxGetScalar (field), Control), "opts failed") ;
        }

        // diagtol: pivot tolerance for diagonal entries
        if ((field = mxGetField (pargin [2], 0, "diagtol")) != NULL)
        {
            PARU_OK (ParU_C_Set_Control_FP64 (PARU_CONTROL_DIAG_PIVOT_TOLERANCE,
                (double) mxGetScalar (field), Control), "opts failed") ;
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
                    PARU_OK (ParU_C_Set_Control_INT64 (PARU_CONTROL_STRATEGY,
                        PARU_STRATEGY_AUTO, Control), "opts failed") ;
                    PARU_OK (ParU_C_Set_Control_INT64 (
                        PARU_CONTROL_UMFPACK_STRATEGY,
                        UMFPACK_STRATEGY_AUTO, Control), "opts failed") ;
                }
                else if (strncmp (option, "unsymmetric", STRLEN) == 0)
                {
                    // both UMFPACK and ParU will use the unsymmetric strategy
                    PARU_OK (ParU_C_Set_Control_INT64 (PARU_CONTROL_STRATEGY,
                        PARU_STRATEGY_UNSYMMETRIC, Control), "opts failed") ;
                    PARU_OK (ParU_C_Set_Control_INT64 (
                        PARU_CONTROL_UMFPACK_STRATEGY,
                        UMFPACK_STRATEGY_UNSYMMETRIC, Control), "opts failed") ;
                }
                else if (strncmp (option, "symmetric", STRLEN) == 0)
                {
                    // both UMFPACK and ParU will use the symmetric strategy
                    PARU_OK (ParU_C_Set_Control_INT64 (PARU_CONTROL_STRATEGY,
                        PARU_STRATEGY_SYMMETRIC, Control), "opts failed") ;
                    PARU_OK (ParU_C_Set_Control_INT64 (
                        PARU_CONTROL_UMFPACK_STRATEGY,
                        UMFPACK_STRATEGY_SYMMETRIC, Control), "opts failed") ;
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

        // ordering: fill-reducing ordering method to use
        if ((field = mxGetField (pargin [2], 0, "ordering")) != NULL)
        {
            if (mxGetString (field, option, STRLEN) == 0)
            {
                option [STRLEN] = '\0' ;
                if (strncmp (option, "amd", STRLEN) == 0)
                {
                    PARU_OK (ParU_C_Set_Control_INT64 (PARU_CONTROL_ORDERING,
                        PARU_ORDERING_AMD, Control), "opts failed") ;
                }
                else if (strncmp (option, "cholmod", STRLEN) == 0)
                {
                    PARU_OK (ParU_C_Set_Control_INT64 (PARU_CONTROL_ORDERING,
                        PARU_ORDERING_CHOLMOD, Control), "opts failed") ;
                }
                else if (strncmp (option, "metis", STRLEN) == 0)
                {
                    PARU_OK (ParU_C_Set_Control_INT64 (PARU_CONTROL_ORDERING,
                        PARU_ORDERING_METIS, Control), "opts failed") ;
                }
                else if (strncmp (option, "metis_guard", STRLEN) == 0)
                {
                    PARU_OK (ParU_C_Set_Control_INT64 (PARU_CONTROL_ORDERING,
                        PARU_ORDERING_METIS_GUARD, Control), "opts failed") ;
                }
                else if (strncmp (option, "none", STRLEN) == 0)
                {
                    PARU_OK (ParU_C_Set_Control_INT64 (PARU_CONTROL_ORDERING,
                        PARU_ORDERING_NONE, Control), "opts failed") ;
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

        // prescale: 'sum', 'max', or 'none'
        if ((field = mxGetField (pargin [2], 0, "prescale")) != NULL)
        {
            if (mxGetString (field, option, STRLEN) == 0)
            {
                option [STRLEN] = '\0' ;
                if (strncmp (option, "none", STRLEN) == 0)
                {
                    PARU_OK (ParU_C_Set_Control_INT64 (PARU_CONTROL_PRESCALE,
                        PARU_PRESCALE_NONE, Control), "opts failed") ;
                }
                else if (strncmp (option, "sum", STRLEN) == 0)
                {
                    PARU_OK (ParU_C_Set_Control_INT64 (PARU_CONTROL_PRESCALE,
                        PARU_PRESCALE_SUM, Control), "opts failed") ;
                }
                else if (strncmp (option, "max", STRLEN) == 0)
                {
                    PARU_OK (ParU_C_Set_Control_INT64 (PARU_CONTROL_PRESCALE,
                        PARU_PRESCALE_MAX, Control), "opts failed") ;
                }
                else
                {
                    mexErrMsgIdAndTxt ("ParU:error",
                        "unrecognized opts.ordering: %s", option) ;
                }
            }
            else
            {
                mexErrMsgIdAndTxt ("ParU:error", "unrecognized opts.prescale") ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // x = A\b using ParU
    //--------------------------------------------------------------------------

    double t [3], t0, t1 ;
    if (nargout > 1)
    {
        t0 = SUITESPARSE_TIME ;
    }

    PARU_OK (ParU_C_Analyze (A, &Sym, Control), "symbolic analysis failed") ;

    if (nargout > 1)
    {
        t1 = SUITESPARSE_TIME ;
        t [0] = t1 - t0 ;
        t0 = t1 ;
    }

    PARU_OK (ParU_C_Factorize (A, Sym, &Num, Control),
        "numeric factorization failed") ;

    if (nargout > 1)
    {
        t1 = SUITESPARSE_TIME ;
        t [1] = t1 - t0 ;
        t0 = t1 ;
    }

    PARU_OK (ParU_C_Solve_AXB (Sym, Num, nrhs, (double *) B->x, (double *) X->x,
        Control), "solve failed") ;

    if (nargout > 1)
    {
        t1 = SUITESPARSE_TIME ;
        t [2] = t1 - t0 ;
        t0 = t1 ;
    }

    // get statistics from ParU
    int64_t strategy_used, ordering_used, lnz, unz, using_openmp ;
    double rcond, flops ;
    const char *blas_name, *front_tree_tasking ;
    PARU_OK (ParU_C_Get_Control_INT64 (PARU_CONTROL_OPENMP,
        &using_openmp, Control), "stats failed") ;
    PARU_OK (ParU_C_Get_INT64 (Sym, Num, PARU_GET_STRATEGY,
        &strategy_used, Control), "stats failed") ;
    PARU_OK (ParU_C_Get_INT64 (Sym, Num, PARU_GET_ORDERING,
        &ordering_used, Control), "stats failed") ;
    PARU_OK (ParU_C_Get_FP64 (Sym, Num, PARU_GET_FLOPS_BOUND,
        &flops, Control), "stats failed") ;
    PARU_OK (ParU_C_Get_INT64 (Sym, Num, PARU_GET_LNZ_BOUND,
        &lnz, Control), "stats failed") ;
    PARU_OK (ParU_C_Get_INT64 (Sym, Num, PARU_GET_UNZ_BOUND,
        &unz, Control), "stats failed") ;
    PARU_OK (ParU_C_Get_FP64 (Sym, Num, PARU_GET_RCOND_ESTIMATE,
        &rcond, Control), "stats failed") ;
    PARU_OK (ParU_C_Get_Control_CONSTCHAR (PARU_CONTROL_BLAS_LIBRARY_NAME,
        &blas_name, Control), "stats failed") ;
    PARU_OK (ParU_C_Get_Control_CONSTCHAR (PARU_CONTROL_FRONT_TREE_TASKING,
        &front_tree_tasking, Control), "stats failed") ;

    //--------------------------------------------------------------------------
    // get P, Q, and R (developer only)
    //--------------------------------------------------------------------------

    #ifdef DEVELOPER
    if (nargout > 2)
    {
        PARU_OK (ParU_C_Get_INT64 (Sym, Num, PARU_GET_P, P, Control), "fail") ;
    }
    if (nargout > 3)
    {
        PARU_OK (ParU_C_Get_INT64 (Sym, Num, PARU_GET_Q, Q, Control), "fail") ;
    }
    if (nargout > 4)
    {
        PARU_OK (ParU_C_Get_FP64 (Sym, Num, PARU_GET_ROW_SCALE_FACTORS, R,
            Control), "fail") ;
    }
    #endif

    //--------------------------------------------------------------------------
    // free the ParU factorization and Control object
    //--------------------------------------------------------------------------

    ParU_C_FreeNumeric (&Num, Control) ;
    ParU_C_FreeSymbolic (&Sym, Control) ;
    ParU_C_FreeControl (&Control) ;

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
        pargout [1] = mxCreateStructMatrix (1, 1, 12, stat_names) ;

        // analysis, factorization, and solve times:
        mxSetFieldByNumber (pargout [1], 0, 0, mxCreateDoubleScalar (t [0])) ;
        mxSetFieldByNumber (pargout [1], 0, 1, mxCreateDoubleScalar (t [1])) ;
        mxSetFieldByNumber (pargout [1], 0, 2, mxCreateDoubleScalar (t [2])) ;

        // UMFPACK and ParU strategy:
        bool symmetric = (strategy_used == PARU_STRATEGY_SYMMETRIC) ;
        mxSetFieldByNumber (pargout [1], 0, 3,
            mxCreateString (symmetric ? "symmetric" : "unsymmetric")) ;

        // ordering used
        char *ordering ;
        switch (ordering_used)
        {
            case PARU_ORDERING_AMD:
                ordering = symmetric ? "amd(A+A')" : "colamd(A)" ;
                break ;

            case PARU_ORDERING_METIS:
                ordering = symmetric ? "metis(A+A')" : "metis(A'*A)" ;
                break ;

            case PARU_ORDERING_NONE:
                ordering = "none" ;
                break ;

            // These cases cannot occur.  Some of them can be specified on
            // input, with opts.ordering, but they are ordering strategies
            // that select amd, colamd, or metis.
            case PARU_ORDERING_CHOLMOD:
            case PARU_ORDERING_BEST:
            case PARU_ORDERING_METIS_GUARD:
            case UMFPACK_ORDERING_GIVEN:
            case UMFPACK_ORDERING_USER:
            default:
                ordering = "undefined" ;
                break ;
        }
        mxSetFieldByNumber (pargout [1], 0, 4, mxCreateString (ordering)) ;

        // numeric factorization statistics:
        mxSetFieldByNumber (pargout [1], 0, 5, mxCreateDoubleScalar (flops)) ;
        mxSetFieldByNumber (pargout [1], 0, 6, mxCreateDoubleScalar (lnz)) ;
        mxSetFieldByNumber (pargout [1], 0, 7, mxCreateDoubleScalar (unz)) ;
        mxSetFieldByNumber (pargout [1], 0, 8, mxCreateDoubleScalar (rcond)) ;

        // BLAS library used
        mxSetFieldByNumber (pargout [1], 0, 9, mxCreateString (blas_name)) ;

        // frontal tree tasking
        mxSetFieldByNumber (pargout [1], 0, 10,
            mxCreateString (front_tree_tasking)) ;

        // openmp:
        mxSetFieldByNumber (pargout [1], 0, 11,
            mxCreateString (using_openmp ? "yes" :
            "not in ParU itself but could be used in the BLAS,"
            " depending on MATLAB (see 'version -blas')")) ;
    }
}

#if defined ( __clang__ ) && defined ( CLANG_NEEDS_MAIN )
// when using clang inside MATLAB, it seems to require a "_main" symbol.
int main (void) { return (0) ; }
#endif

