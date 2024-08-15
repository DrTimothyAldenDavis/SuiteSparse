// ============================================================================/
// ======================= ParU.h =============================================/
// ============================================================================/

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//------------------------------------------------------------------------------

// This is the ParU.h file. All user callable routines are in this file and
// all of them start with ParU_*. This file must be included in all user code
// that use ParU.
//
// ParU is a parallel sparse direct solver. This package uses OpenMP tasking
// for parallelism. ParU calls UMFPACK for symbolic analysis phase, after that
// some symbolic analysis is done by ParU itself and then numeric phase
// starts. The numeric computation is a task parallel phase using OpenMP and
// each task calls parallel BLAS; i.e. nested parallism.
//
// The performance of BLAS has a heavy impact on the performance of ParU.
// However, depending on the input problem performance of parallelism in BLAS
// sometimes does not have an effect on the ParU performance.
//
// General Usage for solving Ax = b, where A is a sparse matrix in a CHOLMOD
// sparse matrix data structure with double entries and b is a dense vector of
// double (or a dense matrix B for multiple rhs):
//
//      info = ParU_InitControl (&Control) ;
//      info = ParU_Analyze (A, &Sym, Control) ;
//      info = ParU_Factorize (A, Sym, &Num, Control) ;
//      info = ParU_Solve (Sym, Num, b, x, Control) ;
//
// See paru_demo for more examples

#ifndef PARU_H
#define PARU_H

// ============================================================================/
// include files and ParU version
// ============================================================================/

#include "SuiteSparse_config.h"
#include "cholmod.h"
#include "umfpack.h"

typedef enum ParU_Info
{
    PARU_SUCCESS = 0,           // everying is fine
    PARU_OUT_OF_MEMORY = -1,    // ParU ran out of memory
    PARU_INVALID = -2,          // inputs are invalid (NULL, for example)
    PARU_SINGULAR = -3,         // matrix is numerically singular
    PARU_TOO_LARGE = -4         // problem too large for the BLAS
} ParU_Info ;

#define PARU_DATE "Aug 20, 2024"
#define PARU_VERSION_MAJOR  0
#define PARU_VERSION_MINOR  3
#define PARU_VERSION_UPDATE 0

#define PARU__VERSION SUITESPARSE__VERCODE(0,3,0)
#if !defined (SUITESPARSE__VERSION) || \
    (SUITESPARSE__VERSION < SUITESPARSE__VERCODE(7,8,0))
#error "ParU 0.3.0 requires SuiteSparse_config 7.8.0 or later"
#endif

#if !defined (UMFPACK__VERSION) || \
    (UMFPACK__VERSION < SUITESPARSE__VERCODE(6,3,4))
#error "ParU 0.3.0 requires UMFPACK 6.3.4 or later"
#endif

#if !defined (CHOLMOD__VERSION) || \
    (CHOLMOD__VERSION < SUITESPARSE__VERCODE(5,3,0))
#error "ParU 0.3.0 requires CHOLMOD 5.3.0 or later"
#endif

//  the same values as UMFPACK_STRATEGY defined in UMFPACK/Include/umfpack.h
#define PARU_STRATEGY_AUTO 0         // decided to use sym. or unsym. strategy
#define PARU_STRATEGY_UNSYMMETRIC 1  // COLAMD(A), metis, ...
#define PARU_STRATEGY_SYMMETRIC 3    // prefer diagonal

#if 0
#define UMFPACK_STRATEGY_AUTO 0         /* use sym. or unsym. strategy */
#define UMFPACK_STRATEGY_UNSYMMETRIC 1  /* COLAMD(A), coletree postorder,
                                           not prefer diag*/
#define UMFPACK_STRATEGY_OBSOLETE 2     /* 2-by-2 is no longer available */
#define UMFPACK_STRATEGY_SYMMETRIC 3    /* AMD(A+A'), no coletree postorder,
                                           prefer diagonal */
#endif

// enum for ParU_Get for Symbolic/Numeric objects
typedef enum
{
    // int64_t scalars:
    PARU_GET_N = 0,                 // # of rows/columns of A and its factors
    PARU_GET_ANZ = 1,               // # of entries in input matrix
    PARU_GET_LNZ_BOUND = 2,         // # of entries held in L
    PARU_GET_UNZ_BOUND = 3,         // # of entries held in U
    PARU_GET_NROW_SINGLETONS = 4,   // # of row singletons
    PARU_GET_NCOL_SINGLETONS = 5,   // # of column singletons
    PARU_GET_STRATEGY = 6,          // strategy used by ParU
    PARU_GET_UMFPACK_STRATEGY = 7,  // strategy used by UMFPACK
    PARU_GET_ORDERING = 8,          // ordering used by UMFPACK

    // int64_t arrays of size n:
    PARU_GET_P = 101,               // partial pivoting row ordering
    PARU_GET_Q = 102,               // fill-reducing column ordering

    // double scalars:
    PARU_GET_FLOPS_BOUND = 201,     // flop count for factorization (bound)
    PARU_GET_RCOND_ESTIMATE = 202,  // rcond estimate
    PARU_GET_MIN_UDIAG = 203,       // min (abs (diag (U)))
    PARU_GET_MAX_UDIAG = 204,       // max (abs (diag (U)))

    // double array of size n:
    PARU_GET_ROW_SCALE_FACTORS = 301,   // row scaling factors

}
ParU_Get_enum ;

// enum for ParU_Set/ParU_Get for Control object
typedef enum
{

    // int64_t parameters for ParU_Set and ParU_Get:
    PARU_CONTROL_MAX_THREADS = 1001,          // max number of threads
    PARU_CONTROL_STRATEGY = 1002,             // ParU strategy
    PARU_CONTROL_UMFPACK_STRATEGY = 1003,     // UMFPACK strategy
    PARU_CONTROL_ORDERING = 1004,             // UMFPACK ordering
    PARU_CONTROL_RELAXED_AMALGAMATION = 1005, // goal for # pivots in fronts
    PARU_CONTROL_PANEL_WIDTH = 1006,          // # of pivots in a panel
    PARU_CONTROL_DGEMM_TINY = 1007,           // dimension of tiny dgemm's
    PARU_CONTROL_DGEMM_TASKED = 1008,         // dimension of tasked dgemm's
    PARU_CONTROL_DTRSM_TASKED = 1009,         // dimension of tasked dtrsm's
    PARU_CONTROL_PRESCALE = 1010,             // prescale input matrix
    PARU_CONTROL_SINGLETONS = 1011,           // filter singletons, or not
    PARU_CONTROL_MEM_CHUNK = 1012,            // chunk size of memset and memcpy

    // int64_t parameter, for ParU_Get only:
    PARU_CONTROL_OPENMP = 1013,               // if ParU compiled with OpenMP;
                                              // (for ParU_Get only, not set)
    PARU_CONTROL_NUM_THREADS = 1014,          // actual number of threads used

    // double parameters for ParU_Set and ParU_Get:
    PARU_CONTROL_PIVOT_TOLERANCE = 2001,      // pivot tolerance
    PARU_CONTROL_DIAG_PIVOT_TOLERANCE = 2002, // diagonal pivot tolerance

    // pointer to const string (const char **), for ParU_Get only:
    PARU_CONTROL_BLAS_LIBRARY_NAME = 3001,    // BLAS library used
    PARU_CONTROL_FRONT_TREE_TASKING = 3002,   // parallel or sequential

}
ParU_Control_enum ;

// ordering options available to ParU:
#define PARU_ORDERING_CHOLMOD UMFPACK_ORDERING_CHOLMOD
#define PARU_ORDERING_AMD     UMFPACK_ORDERING_AMD
#define PARU_ORDERING_METIS   UMFPACK_ORDERING_METIS
#define PARU_ORDERING_BEST    UMFPACK_ORDERING_BEST
#define PARU_ORDERING_NONE    UMFPACK_ORDERING_NONE
#define PARU_ORDERING_METIS_GUARD UMFPACK_ORDERING_METIS_GUARD

// scaling options for ParU:
#define PARU_PRESCALE_NONE 0
#define PARU_PRESCALE_SUM 1
#define PARU_PRESCALE_MAX 2

    // ordering options described:
    // PARU_ORDERING_CHOLMOD: use CHOLMOD (AMD/COLAMD then METIS, see below)
    // PARU_ORDERING_AMD: use AMD on A+A' (symmetric strategy) or COLAMD
    //      (unsymmetric strategy)
    // PARU_ORDERING_METIS: use METIS on A+A' (symmetric strategy) or A'A
    //      (unsymmetric strategy)
    // PARU_ORDERING_BEST: try many orderings, pick best
    // PARU_ORDERING_NONE: natural ordering
    // PARU_ORDERING_METIS_GUARD: use METIS, AMD, or COLAMD.  Symmetric
    // strategy: always use METIS on A+A'.  Unsymmetric strategy: use METIS on
    // A'A, unless A has one or more very dense rows.  In that case, A'A is
    // very costly to form, and COLAMD is used instead of METIS.

// default values of Control parameters
#define PARU_DEFAULT_MAX_THREADS            (0) /* get default from OpenMP */
#define PARU_DEFAULT_STRATEGY               PARU_STRATEGY_AUTO
#define PARU_DEFAULT_UMFPACK_STRATEGY       UMFPACK_STRATEGY_AUTO
#define PARU_DEFAULT_ORDERING               PARU_ORDERING_METIS_GUARD
#define PARU_DEFAULT_RELAXED_AMALGAMATION   (32)
#define PARU_DEFAULT_PANEL_WIDTH            (32)
#define PARU_DEFAULT_DGEMM_TINY             (4)
#define PARU_DEFAULT_DGEMM_TASKED           (512)
#define PARU_DEFAULT_DTRSM_TASKED           (4096)
#define PARU_DEFAULT_PRESCALE               PARU_PRESCALE_MAX
#define PARU_DEFAULT_SINGLETONS             (1)
#define PARU_DEFAULT_MEM_CHUNK              (1024*1024)
#define PARU_DEFAULT_PIVOT_TOLERANCE        (0.1)
#define PARU_DEFAULT_DIAG_PIVOT_TOLERANCE   (0.001)

// Note that the default UMFPACK scaling is SUM, not MAX.

// =============================================================================
// ParU C++ definitions ========================================================
// =============================================================================

#ifdef __cplusplus

// The following definitions are only available from C++:

// silence these diagnostics:
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-extensions"
#endif

#include <cmath>

//------------------------------------------------------------------------------
// opaque objects (ParU_Symbolic, ParU_Numeric, and ParU_Control):
//------------------------------------------------------------------------------

typedef struct ParU_Symbolic_struct *ParU_Symbolic ;
typedef struct ParU_Numeric_struct  *ParU_Numeric ;
typedef struct ParU_Control_struct  *ParU_Control ;

//------------------------------------------------------------------------------
// ParU_Version:
//------------------------------------------------------------------------------

// return the version and date of the ParU library.

ParU_Info ParU_Version (int ver [3], char date [128]) ;

//------------------------------------------------------------------------------
// ParU_Analyze: Symbolic analysis is done in this routine. UMFPACK is called
// here and after that some more specialized symbolic computation is done for
// ParU. ParU_Analyze can be called once and can be used for different
// ParU_Factorize calls.
//------------------------------------------------------------------------------

ParU_Info ParU_Analyze
(
    // input:
    cholmod_sparse *A,          // input matrix to analyze of size n-by-n
    // output:
    ParU_Symbolic *Sym_handle,  // output, symbolic analysis
    // control:
    ParU_Control Control
) ;

//------------------------------------------------------------------------------
// ParU_Factorize: Numeric factorization is done in this routine. Scaling and
// making Sx matrix, computing factors and permutations is here. ParU_Symbolic
// structure is computed ParU_Analyze and is an input in this routine.
//------------------------------------------------------------------------------

ParU_Info ParU_Factorize
(
    // input:
    cholmod_sparse *A,          // input matrix to factorize
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    // output:
    ParU_Numeric *Num_handle,
    // control:
    ParU_Control Control
) ;

//------------------------------------------------------------------------------
//--------------------- Solve routines -----------------------------------------
//------------------------------------------------------------------------------

// In all the solve routines Num structure must come with the same Sym struct
// that comes from ParU_Factorize

// The vectors x and b have length n, where the matrix factorized is n-by-n.
// The matrices X and B have size n-by-?  nrhs, and are held in column-major
// storage.

//-------- x = A\x -------------------------------------------------------------
ParU_Info ParU_Solve            // solve Ax=b, overwriting b with solution x
(
    // input:
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    const ParU_Numeric Num,     // numeric factorization from ParU_Factorize
    // input/output:
    double *x,                  // vector of size n-by-1; right-hand on input,
                                // solution on output
    // control:
    ParU_Control Control
) ;

//-------- x = A\b -------------------------------------------------------------
ParU_Info ParU_Solve            // solve Ax=b
(
    // input:
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    const ParU_Numeric Num,     // numeric factorization from ParU_Factorize
    double *b,                  // vector of size n-by-1
    // output
    double *x,                  // vector of size n-by-1
    // control:
    ParU_Control Control
) ;

//-------- X = A\X -------------------------------------------------------------
ParU_Info ParU_Solve            // solve AX=B, overwriting B with solution X
(
    // input
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    const ParU_Numeric Num,     // numeric factorization from ParU_Factorize
    int64_t nrhs,               // # of right-hand sides
    // input/output:
    double *X,                  // X is n-by-nrhs, where A is n-by-n;
                                // holds B on input, solution X on input
    // control:
    ParU_Control Control
) ;

//-------- X = A\B -------------------------------------------------------------
ParU_Info ParU_Solve            // solve AX=B
(
    // input
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    const ParU_Numeric Num,     // numeric factorization from ParU_Factorize
    int64_t nrhs,               // # of right-hand sides
    double *B,                  // n-by-nrhs, in column-major storage
    // output:
    double *X,                  // n-by-nrhs, in column-major storage
    // control:
    ParU_Control Control
) ;

// Solve L*x=b where x and b are vectors (no scaling or permutations)
ParU_Info ParU_LSolve           // solve Lx=b
(
    // input
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    const ParU_Numeric Num,     // numeric factorization from ParU_Factorize
    // input/output:
    double *x,                  // n-by-1, in column-major storage;
                                // holds b on input, solution x on input
    // control:
    ParU_Control Control
) ;

// Solve L*X=B where X and B are matrices (no scaling or permutations)
ParU_Info ParU_LSolve           // solve LX=B
(
    // input
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    const ParU_Numeric Num,     // numeric factorization from ParU_Factorize
    int64_t nrhs,               // # of right-hand-sides (# columns of X)
    // input/output:
    double *X,                  // X is n-by-nrhs, where A is n-by-n;
                                // holds B on input, solution X on input
    // control:
    ParU_Control Control
) ;

// Solve U*x=b where x and b are vectors (no scaling or permutations)
ParU_Info ParU_USolve           // solve Ux=b
(
    // input
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    const ParU_Numeric Num,     // numeric factorization from ParU_Factorize
    // input/output
    double *x,                  // n-by-1, in column-major storage;
                                // holds b on input, solution x on input
    // control:
    ParU_Control Control
) ;

// Solve U*X=B where X and B are matrices (no scaling or permutations)
ParU_Info ParU_USolve           // solve UX=B
(
    // input
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    const ParU_Numeric Num,     // numeric factorization from ParU_Factorize
    int64_t nrhs,               // # of right-hand-sides (# columns of X)
    // input/output:
    double *X,                  // X is n-by-nrhs, where A is n-by-n;
                                // holds B on input, solution X on input
    // control:
    ParU_Control Control
) ;

//------------------------------------------------------------------------------
// permutation and inverse permutation, with optional scaling
//------------------------------------------------------------------------------

// apply inverse perm x(p) = b, or with scaling: x(p)=b ; x=x./s
ParU_Info ParU_InvPerm
(
    // inputs
    const int64_t *P,   // permutation vector of size n
    const double *s,    // vector of size n (optional)
    const double *b,    // vector of size n
    int64_t n,          // length of P, s, B, and X
    // output
    double *x,          // vector of size n
    // control:
    ParU_Control Control
) ;

// apply inverse perm X(p,:) = B or with scaling: X(p,:)=B ; X = X./s
ParU_Info ParU_InvPerm
(
    // inputs
    const int64_t *P,   // permutation vector of size nrows
    const double *s,    // vector of size nrows (optional)
    const double *B,    // array of size nrows-by-ncols
    int64_t nrows,      // # of rows of X and B
    int64_t ncols,      // # of columns of X and B
    // output
    double *X,          // array of size nrows-by-ncols
    // control:
    ParU_Control Control
) ;

// apply perm and scale x = b(P) / s
ParU_Info ParU_Perm
(
    // inputs
    const int64_t *P,   // permutation vector of size n
    const double *s,    // vector of size n (optional)
    const double *b,    // vector of size n
    int64_t n,          // length of P, s, B, and X
    // output
    double *x,          // vector of size n
    // control:
    ParU_Control Control
) ;

// apply perm and scale X = B(P,:) / s
ParU_Info ParU_Perm
(
    // inputs
    const int64_t *P,   // permutation vector of size nrows
    const double *s,    // vector of size nrows (optional)
    const double *B,    // array of size nrows-by-ncols
    int64_t nrows,      // # of rows of X and B
    int64_t ncols,      // # of columns of X and B
    // output
    double *X,          // array of size nrows-by-ncols
    // control:
    ParU_Control Control
) ;

//------------------------------------------------------------------------------
//-------------- computing residual --------------------------------------------
//------------------------------------------------------------------------------

// resid = norm1(b-A*x) / (norm1(A) * norm1 (x))
ParU_Info ParU_Residual
(
    // inputs:
    cholmod_sparse *A,  // an n-by-n sparse matrix
    double *x,          // vector of size n, solution to Ax=b
    double *b,          // vector of size n
    // output:
    double &resid,      // residual: norm1(b-A*x) / (norm1(A) * norm1 (x))
    double &anorm,      // 1-norm of A
    double &xnorm,      // 1-norm of x
    // control:
    ParU_Control Control
) ;

// resid = norm1(B-A*X) / (norm1(A) * norm1 (X))
// (multiple rhs)
ParU_Info ParU_Residual
(
    // inputs:
    cholmod_sparse *A,  // an n-by-n sparse matrix
    double *X,          // array of size n-by-nrhs, solution to AX=B
    double *B,          // array of size n-by-nrhs
    int64_t nrhs,
    // output:
    double &resid,      // residual: norm1(B-A*X) / (norm1(A) * norm1 (X))
    double &anorm,      // 1-norm of A
    double &xnorm,      // 1-norm of X
    // control:
    ParU_Control Control
) ;

//------------------------------------------------------------------------------
//------------ Get statistics and contents of factorization --------------------
//------------------------------------------------------------------------------

ParU_Info ParU_Get              // get int64_t from the symbolic/numeric objects
(
    // input:
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    const ParU_Numeric Num,     // numeric factorization from ParU_Factorize
    ParU_Get_enum field,        // field to get
    // output:
    int64_t *result,            // int64_t result: a scalar or an array
    // control:
    ParU_Control Control
) ;

ParU_Info ParU_Get              // get double from the symbolic/numeric objects
(
    // input:
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    const ParU_Numeric Num,     // numeric factorization from ParU_Factorize
    ParU_Get_enum field,        // field to get
    // output:
    double *result,             // double result: a scalar or an array
    // control:
    ParU_Control Control
) ;

//------------------------------------------------------------------------------
//------------ Get/Set control parameters --------------------------------------
//------------------------------------------------------------------------------

ParU_Info ParU_Set              // set int32_t parameter in Control
(
    // input
    ParU_Control_enum field,    // field to set
    int32_t c,                  // value to set it to
    // control:
    ParU_Control Control
) ;

ParU_Info ParU_Set              // set int64_t parameter in Control
(
    // input
    ParU_Control_enum field,    // field to set
    int64_t c,                  // value to set it to
    // control:
    ParU_Control Control
) ;

ParU_Info ParU_Set              // set double parameter in Control
(
    // input
    ParU_Control_enum field,    // field to set
    double c,                   // value to set it to
    // control:
    ParU_Control Control
) ;

ParU_Info ParU_Set              // set float parameter in Control
(
    // input
    ParU_Control_enum field,    // field to set
    float c,                    // value to set it to
    // control:
    ParU_Control Control
) ;

ParU_Info ParU_Get              // get int64_t parameter from Control
(
    // input
    ParU_Control_enum field,    // field to get
    // output:
    int64_t *c,                 // value of field
    // control:
    ParU_Control Control
) ;

ParU_Info ParU_Get              // get double parameter from Control
(
    // input
    ParU_Control_enum field,    // field to get
    // output:
    double *c,                  // value of field
    // control:
    ParU_Control Control
) ;

ParU_Info ParU_Get              // get string from Control
(
    // input:
    ParU_Control_enum field,    // field to get
    // output:
    const char **result,        // string result
    // control:
    ParU_Control Control
) ;

//------------------------------------------------------------------------------
//------------ Free routines----------------------------------------------------
//------------------------------------------------------------------------------

ParU_Info ParU_FreeNumeric
(
    // input/output:
    ParU_Numeric *Num_handle,       // numeric object to free
    // control:
    ParU_Control Control
) ;

ParU_Info ParU_FreeSymbolic
(
    // input/output:
    ParU_Symbolic *Sym_handle,      // symbolic object to free
    // control:
    ParU_Control Control
) ;

ParU_Info ParU_InitControl
(
    // output:
    ParU_Control *Control_handle    // Control object to create
) ;

ParU_Info ParU_FreeControl
(
    // input/output:
    ParU_Control *Control_handle    // Control object to free
) ;

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif

// =============================================================================
// ParU C definitions ==========================================================
// =============================================================================

// The following definitions are available in both C and C++:

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ParU_C_Symbolic_struct *ParU_C_Symbolic ;
typedef struct ParU_C_Numeric_struct  *ParU_C_Numeric ;
typedef struct ParU_C_Control_struct  *ParU_C_Control ;

//------------------------------------------------------------------------------
// ParU_Version: return the version and date of ParU
//------------------------------------------------------------------------------

ParU_Info ParU_C_Version (int ver [3], char date [128]) ;

//------------------------------------------------------------------------------
// ParU_C_InitControl: initialize C data structure
//------------------------------------------------------------------------------

ParU_Info ParU_C_InitControl (ParU_C_Control *Control_C_handle) ;

//------------------------------------------------------------------------------
// ParU_C_Analyze: Symbolic analysis is done in this routine. UMFPACK is called
// here and after that some more speciaized symbolic computation is done for
// ParU. ParU_Analyze can be called once and can be used for different
// ParU_Factorize calls.
//------------------------------------------------------------------------------

ParU_Info ParU_C_Analyze
(
    // input:
    cholmod_sparse *A,  // input matrix to analyze of size n-by-n
    // output:
    ParU_C_Symbolic *Sym_handle_C,  // output, symbolic analysis
    // control:
    ParU_C_Control Control_C
) ;

//------------------------------------------------------------------------------
// ParU_C_Factorize: Numeric factorization is done in this routine. Scaling and
// making Sx matrix, computing factors and permutations is here.
// ParU_C_Symbolic structure is computed ParU_Analyze and is an input in this
// routine.
//------------------------------------------------------------------------------

ParU_Info ParU_C_Factorize
(
    // input:
    cholmod_sparse *A,              // input matrix to factorize of size n-by-n
    const ParU_C_Symbolic Sym_C,    // symbolic analysis from ParU_Analyze
    // output:
    ParU_C_Numeric *Num_handle_C,   // output numerical factorization
    // control:
    ParU_C_Control Control_C
) ;

//------------------------------------------------------------------------------
//--------------------- Solve routines -----------------------------------------
//------------------------------------------------------------------------------

// x = A\x, where right-hand side is overwritten with the solution x.
ParU_Info ParU_C_Solve_Axx       // solve Ax=b, overwriting b with solution x
(
    // input:
    const ParU_C_Symbolic Sym_C, // symbolic analysis from ParU_C_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    // input/output:
    double *x,              // vector of size n-by-1; right-hand on input,
                            // solution on output
    // control:
    ParU_C_Control Control_C
) ;

// x = A\b, for vectors x and b
ParU_Info ParU_C_Solve_Axb        // solve Ax=b
(
    // input:
    const ParU_C_Symbolic Sym_C, // symbolic analysis from ParU_C_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    double *b,              // vector of size n-by-1
    // output
    double *x,              // vector of size n-by-1
    // control:
    ParU_C_Control Control_C
) ;

// X = A\X, where right-hand side is overwritten with the solution X.
ParU_Info ParU_C_Solve_AXX       // solve AX=B, overwriting B with solution X
(
    // input
    const ParU_C_Symbolic Sym_C, // symbolic analysis from ParU_C_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    int64_t nrhs,
    // input/output:
    double *X,              // array of size n-by-nrhs in column-major storage,
                            // right-hand-side on input, solution on output.
    // control:
    ParU_C_Control Control_C
) ;

// X = A\B, for matrices X and B
ParU_Info ParU_C_Solve_AXB       // solve AX=B, overwriting B with solution X
(
    // input
    const ParU_C_Symbolic Sym_C, // symbolic analysis from ParU_C_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    int64_t nrhs,
    double *B,              // array of size n-by-nrhs in column-major storage
    // output:
    double *X,              // array of size n-by-nrhs in column-major storage
    // control:
    ParU_C_Control Control_C
) ;

// x = L\x, where right-hand side is overwritten with the solution x.
ParU_Info ParU_C_Solve_Lxx       // solve Lx=b, overwriting b with solution x
(
    // input:
    const ParU_C_Symbolic Sym_C, // symbolic analysis from ParU_C_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    // input/output:
    double *x,              // vector of size n-by-1; right-hand on input,
                            // solution on output
    // control:
    ParU_C_Control Control_C
) ;

// X = L\X, where right-hand side is overwritten with the solution X.
ParU_Info ParU_C_Solve_LXX       // solve LX=B, overwriting B with solution X
(
    // input
    const ParU_C_Symbolic Sym_C, // symbolic analysis from ParU_C_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    int64_t nrhs,
    // input/output:
    double *X,              // array of size n-by-nrhs in column-major storage,
                            // right-hand-side on input, solution on output.
    // control:
    ParU_C_Control Control_C
) ;

// x = U\x, where right-hand side is overwritten with the solution x.
ParU_Info ParU_C_Solve_Uxx       // solve Ux=b, overwriting b with solution x
(
    // input:
    const ParU_C_Symbolic Sym_C, // symbolic analysis from ParU_C_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    // input/output:
    double *x,              // vector of size n-by-1; right-hand on input,
                            // solution on output
    // control:
    ParU_C_Control Control_C
) ;

// X = U\X, where right-hand side is overwritten with the solution X.
ParU_Info ParU_C_Solve_UXX       // solve UX=B, overwriting B with solution X
(
    // input
    const ParU_C_Symbolic Sym_C, // symbolic analysis from ParU_C_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    int64_t nrhs,
    // input/output:
    double *X,              // array of size n-by-nrhs in column-major storage,
                            // right-hand-side on input, solution on output.
    // control:
    ParU_C_Control Control_C
) ;

//------------------------------------------------------------------------------
// Perm and InvPerm
//------------------------------------------------------------------------------

// apply permutation to a vector, x=b(p)./s
ParU_Info ParU_C_Perm
(
    // inputs
    const int64_t *P,   // permutation vector of size n
    const double *s,    // vector of size n (optional)
    const double *b,    // vector of size n
    int64_t n,          // length of P, s, B, and X
    // output
    double *x,          // vector of size n
    // control:
    ParU_C_Control Control_C
) ;

// apply permutation to a matrix, X=B(p,:)./s
ParU_Info ParU_C_Perm_X
(
    // inputs
    const int64_t *P,   // permutation vector of size nrows
    const double *s,    // vector of size nrows (optional)
    const double *B,    // array of size nrows-by-ncols
    int64_t nrows,      // # of rows of X and B
    int64_t ncols,      // # of columns of X and B
    // output
    double *X,          // array of size nrows-by-ncols
    // control:
    ParU_C_Control Control_C
) ;

// apply inverse permutation to a vector, x(p)=b, then scale x=x./s
ParU_Info ParU_C_InvPerm
(
    // inputs
    const int64_t *P,   // permutation vector of size n
    const double *s,    // vector of size n (optional)
    const double *b,    // vector of size n
    int64_t n,          // length of P, s, B, and X
    // output
    double *x,          // vector of size n
    // control
    ParU_C_Control Control_C
) ;

// apply inverse permutation to a matrix, X(p,:)=b, then scale X=X./s
ParU_Info ParU_C_InvPerm_X
(
    // inputs
    const int64_t *P,   // permutation vector of size nrows
    const double *s,    // vector of size nrows (optional)
    const double *B,    // array of size nrows-by-ncols
    int64_t nrows,      // # of rows of X and B
    int64_t ncols,      // # of columns of X and B
    // output
    double *X,          // array of size nrows-by-ncols
    // control
    ParU_C_Control Control_C
) ;

//------------------------------------------------------------------------------
//-------------- computing residual --------------------------------------------
//------------------------------------------------------------------------------

// resid = norm1(b-A*x) / (norm1(A) * norm1 (x))
ParU_Info ParU_C_Residual_bAx
(
    // inputs:
    cholmod_sparse *A,  // an n-by-n sparse matrix
    double *x,          // vector of size n
    double *b,          // vector of size n
    // output:
    double *residc,     // residual: norm1(b-A*x) / (norm1(A) * norm1 (x))
    double *anormc,     // 1-norm of A
    double *xnormc,     // 1-norm of x
    // control:
    ParU_C_Control Control_C
) ;

// resid = norm1(B-A*X) / (norm1(A) * norm1 (X))
ParU_Info ParU_C_Residual_BAX
(
    // inputs:
    cholmod_sparse *A,  // an n-by-n sparse matrix
    double *X,          // array of size n-by-nrhs
    double *B,          // array of size n-by-nrhs
    int64_t nrhs,
    // output:
    double *residc,     // residual: norm1(B-A*X) / (norm1(A) * norm1 (X))
    double *anormc,     // 1-norm of A
    double *xnormc,     // 1-norm of X
    // control:
    ParU_C_Control Control_C
) ;

//------------------------------------------------------------------------------
//------------ ParU_C_Get_*-----------------------------------------------------
//------------------------------------------------------------------------------

ParU_Info ParU_C_Get_INT64       // get int64_t contents of Sym_C and Num_C
(
    // input:
    const ParU_C_Symbolic Sym_C, // symbolic analysis from ParU_C_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    ParU_Get_enum field,         // field to get
    // output:
    int64_t *result,             // int64_t result: a scalar or an array
    // control:
    ParU_C_Control Control_C
) ;

ParU_Info ParU_C_Get_FP64        // get double contents of Sym_C and Num_C
(
    // input:
    const ParU_C_Symbolic Sym_C, // symbolic analysis from ParU_C_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    ParU_Get_enum field,         // field to get
    // output:
    double *result,              // double result: a scalar or an array
    // control:
    ParU_C_Control Control_C
) ;

ParU_Info ParU_C_Get_Control_INT64  // get int64_t contents of Control
(
    // input:
    ParU_Control_enum field,      // field to get
    // output:
    int64_t *result,              // int64_t result: a scalar or an array
    // control:
    ParU_C_Control Control_C
) ;

ParU_Info ParU_C_Get_Control_FP64   // get double contents of Control
(
    // input:
    ParU_Control_enum field,      // field to get
    // output:
    double *result,               // int64_t result: a scalar or an array
    // control:
    ParU_C_Control Control_C
) ;

ParU_Info ParU_C_Get_Control_CONSTCHAR   // get string from Control
(
    // input:
    ParU_Control_enum field,      // field to get
    // output:
    const char **result,          // string result
    // control:
    ParU_C_Control Control_C
) ;

//------------------------------------------------------------------------------
//------------ ParU_C_Set_*-----------------------------------------------------
//------------------------------------------------------------------------------

ParU_Info ParU_C_Set_Control_INT64      // set int64_t parameter in Control
(
    // input
    ParU_Control_enum field,    // field to set
    int64_t c,                  // value to set it to
    // control:
    ParU_C_Control Control_C
) ;

ParU_Info ParU_C_Set_Control_FP64       // set double parameter in Control
(
    // input
    ParU_Control_enum field,    // field to set
    double c,                   // value to set it to
    // control:
    ParU_C_Control Control_C
) ;

//------------------------------------------------------------------------------
//------------ Free routines----------------------------------------------------
//------------------------------------------------------------------------------

ParU_Info ParU_C_FreeNumeric
(
    ParU_C_Numeric *Num_handle_C,    // numeric object to free
    // control:
    ParU_C_Control Control_C
) ;

ParU_Info ParU_C_FreeSymbolic
(
    ParU_C_Symbolic *Sym_handle_C,   // symbolic object to free
    // control:
    ParU_C_Control Control_C
) ;

ParU_Info ParU_C_FreeControl
(
    ParU_C_Control *Control_handle_C    // Control object to free
) ;

#ifdef __cplusplus
}
#endif

#endif

