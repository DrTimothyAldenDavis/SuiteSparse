//------------------------------------------------------------------------------
// SLIP_LU/slip_internal: include file for internal use in SLIP_LU
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

// This file is not intended to be #include'd in user applications.  Use
// SLIP_LU.h instead.

#ifndef SLIP_LU_INTERNAL_H
#define SLIP_LU_INTERNAL_H

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-value"

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------------C Libraries------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// Standard C libraries
#include <setjmp.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <inttypes.h>

// SuiteSparse headers
#include "SuiteSparse_config.h"
#include "colamd.h"
#include "amd.h"

//------------------------------------------------------------------------------
// debugging
//------------------------------------------------------------------------------

#ifdef SLIP_DEBUG

#ifdef MATLAB_MEX_FILE

#define ASSERT(x)                                                             \
{                                                                             \
    if (!(x))                                                                 \
    {                                                                         \
        mexErrMsgTxt ("assertion failed: %s line %d\n", __FILE__, __LINE__) ; \
    }                                                                         \
}

#else

#include <assert.h>
#define ASSERT(x) assert (x)

#endif

#else

// debugging disabled
#define ASSERT(x)

#endif

//------------------------------------------------------------------------------
//-------------------------Common Macros----------------------------------------
//------------------------------------------------------------------------------

#ifdef MATLAB_MEX_FILE
#include "mex.h"
#include "matrix.h"
#endif

#define SLIP_MAX(a,b) (((a) > (b)) ? (a) : (b))
#define SLIP_MIN(a,b) (((a) < (b)) ? (a) : (b))
#define SLIP_FLIP(i) (-(i)-2)
#define SLIP_UNFLIP(i) (((i) < 0) ? SLIP_FLIP(i) : (i))
#define SLIP_MARKED(Ap,j) (Ap [j] < 0)
#define SLIP_MARK(Ap,j) { Ap [j] = SLIP_FLIP (Ap [j]) ; }

// SLIP_CHECK(method) is a macro that calls a SLIP LU method and checks the
// status; if a failure occurs, it frees all allocated workspace and returns
// the error status to the caller.  To use SLIP_CHECK, the #include'ing file
// must declare a SLIP_info info, and must define SLIP_FREE_ALL as a macro that
// frees all workspace if an error occurs. The method can be a scalar as well,
// so that SLIP_CHECK(info) works.

// the default is to free nothing
#ifndef SLIP_FREE_ALL
#define SLIP_FREE_ALL
#endif

#define SLIP_CHECK(method)      \
{                               \
    info = (method) ;           \
    if (info != SLIP_OK)        \
    {                           \
        SLIP_FREE_ALL ;         \
        return (info) ;         \
    }                           \
}

#include "SLIP_LU.h"

//------------------------------------------------------------------------------
// printing control
//------------------------------------------------------------------------------

// SLIP_LU uses SuiteSparse_config.printf_func instead of a mere call to printf
// (the default function is printf, or mexPrintf when in MATLAB).  If this
// function pointer is NULL, no printing is done.

#define SLIP_PRINTF(...)                                    \
{                                                           \
    if (SuiteSparse_config.printf_func != NULL)             \
    {                                                       \
        SuiteSparse_config.printf_func (__VA_ARGS__) ;      \
    }                                                       \
}

#define SLIP_PR1(...) { if (pr >= 1) SLIP_PRINTF (__VA_ARGS__) }
#define SLIP_PR2(...) { if (pr >= 2) SLIP_PRINTF (__VA_ARGS__) }
#define SLIP_PR3(...) { if (pr >= 3) SLIP_PRINTF (__VA_ARGS__) }

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------------functions for GMP wrapper----------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// uncomment this to print memory debugging info
// #define SLIP_GMP_MEMORY_DEBUG

#ifdef SLIP_GMP_MEMORY_DEBUG
void slip_gmp_dump ( void ) ;
#endif

extern int64_t slip_gmp_ntrials ;

#ifndef SLIP_GMP_LIST_INIT
// A size of 32 ensures that the list never needs to be increased in size.
// The test coverage suite in SLIP_LU/Tcov reduces this initial size to
// exercise the code, in SLIP_LU/Tcov/Makefile.
#define SLIP_GMP_LIST_INIT 32
#endif


bool slip_gmp_init (void) ;

void slip_gmp_finalize (void) ;

void *slip_gmp_allocate (size_t size) ;

void slip_gmp_free (void *p, size_t size) ;

void *slip_gmp_reallocate (void *p_old, size_t old_size, size_t new_size );

void slip_gmp_failure (int status) ;


// Tolerance used in the pivoting schemes. This number can be anything in
// between 0 and 1. A value of 0 selects the diagonal element exclusively and a
// value of 1 selects the smallest or largest pivot exclusively only in a
// tolerance pivoting based method
#define SLIP_DEFAULT_TOL 1

// Check parameter. If this = 1 then the solution to the system is checked
// for accuracy
#define SLIP_DEFAULT_CHECK false

// Pivoting scheme used for SLIP LU.
//  SLIP_SMALLEST = 0,              Smallest pivot
//  SLIP_DIAGONAL = 1,              Diagonal pivoting
//  SLIP_FIRST_NONZERO = 2,         First nonzero per column chosen as pivot
//  SLIP_TOL_SMALLEST = 3,          Diagonal pivoting with tolerance for small
//  SLIP_TOL_LARGEST = 4,           Diagonal pivoting with tolerance for large
//  SLIP_LARGEST = 5                Largest pivot
#define SLIP_DEFAULT_PIVOT SLIP_TOL_SMALLEST

// Column ordering used.
//  SLIP_NO_ORDERING = 0,           None: Not recommended for sparse matrices
//  SLIP_COLAMD = 1,                COLAMD: Default
//  SLIP_AMD = 2                    AMD
#define SLIP_DEFAULT_ORDER SLIP_COLAMD

// Defines printing to be done
#define SLIP_DEFAULT_PRINT_LEVEL 0

// MPFR precision used (quad is default)
#define SLIP_DEFAULT_PRECISION 128

//------------------------------------------------------------------------------
// Type of MPFR rounding used.
//------------------------------------------------------------------------------

// The MPFR library utilizes an internal rounding scheme. The options are
//  MPFR_RNDN: round to nearest (roundTiesToEven in IEEE 754-2008),
//  MPFR_RNDZ: round toward zero (roundTowardZero in IEEE 754-2008),
//  MPFR_RNDU: round toward plus infinity (roundTowardPositive in
//             IEEE 754-2008),
//  MPFR_RNDD: round toward minus infinity (roundTowardNegative in
//             IEEE 754-2008),
//  MPFR_RNDA: round away from zero.
//  MPFR_RNDF: faithful rounding. This is not stable
//
// SLIP LU utilizes MPFR_RNDN by default.

#define SLIP_DEFAULT_MPFR_ROUND MPFR_RNDN

//------------------------------------------------------------------------------
// Macros to utilize the default if option is NULL
//------------------------------------------------------------------------------

#define SLIP_OPTION(option,parameter,default_value) \
    ((option == NULL) ? (default_value) : (option->parameter))

#define SLIP_OPTION_TOL(option) \
    SLIP_OPTION (option, tol, SLIP_DEFAULT_TOL)

#define SLIP_OPTION_CHECK(option) \
    SLIP_OPTION (option, check, false)

#define SLIP_OPTION_PIVOT(option) \
    SLIP_OPTION (option, pivot, SLIP_DEFAULT_PIVOT)

#define SLIP_OPTION_ORDER(option) \
    SLIP_OPTION (option, order, SLIP_DEFAULT_ORDER)

#define SLIP_OPTION_PREC(option) \
    SLIP_OPTION (option, prec, SLIP_DEFAULT_PRECISION)

#define SLIP_OPTION_PRINT_LEVEL(option) \
    SLIP_OPTION (option, print_level, SLIP_DEFAULT_PRINT_LEVEL)

#define SLIP_OPTION_ROUND(option) \
    SLIP_OPTION (option, round, SLIP_DEFAULT_MPFR_ROUND)

//------------------------------------------------------------------------------
// Field access macros for MPZ/MPQ/MPFR struct
//------------------------------------------------------------------------------

// (similar definition in gmp-impl.h and mpfr-impl.h)

#define SLIP_MPZ_SIZ(x)   ((x)->_mp_size)
#define SLIP_MPZ_PTR(x)   ((x)->_mp_d)
#define SLIP_MPZ_ALLOC(x) ((x)->_mp_alloc)
#define SLIP_MPQ_NUM(x)   mpq_numref(x)
#define SLIP_MPQ_DEN(x)   mpq_denref(x)
#define SLIP_MPFR_MANT(x) ((x)->_mpfr_d)
#define SLIP_MPFR_EXP(x)  ((x)->_mpfr_exp)
#define SLIP_MPFR_PREC(x) ((x)->_mpfr_prec)
#define SLIP_MPFR_SIGN(x) ((x)->_mpfr_sign)

/*re-define but same result: */
#define SLIP_MPFR_REAL_PTR(x) (&((x)->_mpfr_d[-1]))

/* Invalid exponent value (to track bugs...) */
#define SLIP_MPFR_EXP_INVALID \
 ((mpfr_exp_t) 1 << (GMP_NUMB_BITS*sizeof(mpfr_exp_t)/sizeof(mp_limb_t)-2))

/* Macros to set the pointer in mpz_t/mpq_t/mpfr_t variable to NULL. It is best
 * practice to call these macros immediately after mpz_t/mpq_t/mpfr_t variable
 * is declared, and before the mp*_init function is called. It would help to
 * prevent error when SLIP_MP*_CLEAR is called before the variable is
 * successfully initialized.
 */

#define SLIP_MPZ_SET_NULL(x)                \
    SLIP_MPZ_PTR(x) = NULL;                 \
    SLIP_MPZ_SIZ(x) = 0;                    \
    SLIP_MPZ_ALLOC(x) = 0;

#define SLIP_MPQ_SET_NULL(x)                     \
    SLIP_MPZ_PTR(SLIP_MPQ_NUM(x)) = NULL;        \
    SLIP_MPZ_SIZ(SLIP_MPQ_NUM(x)) = 0;           \
    SLIP_MPZ_ALLOC(SLIP_MPQ_NUM(x)) = 0;         \
    SLIP_MPZ_PTR(SLIP_MPQ_DEN(x)) = NULL;        \
    SLIP_MPZ_SIZ(SLIP_MPQ_DEN(x)) = 0;           \
    SLIP_MPZ_ALLOC(SLIP_MPQ_DEN(x)) = 0;

#define SLIP_MPFR_SET_NULL(x)               \
    SLIP_MPFR_MANT(x) = NULL;               \
    SLIP_MPFR_PREC(x) = 0;                  \
    SLIP_MPFR_SIGN(x) = 1;                  \
    SLIP_MPFR_EXP(x) = SLIP_MPFR_EXP_INVALID;

/* GMP does not give a mechanism to tell a user when an mpz, mpq, or mpfr
 * item has been cleared; thus, if mp*_clear is called on an object that
 * has already been cleared, gmp will crash. It is also not possible to
 * set a mp*_t = NULL. Thus, this mechanism modifies the internal GMP
 * size of entries to avoid crashing in the case that a mp*_t is cleared
 * multiple times.
 */

#define SLIP_MPZ_CLEAR(x)                        \
{                                                \
    if ((x) != NULL && SLIP_MPZ_PTR(x) != NULL)  \
    {                                            \
        mpz_clear(x);                            \
        SLIP_MPZ_SET_NULL(x);                    \
    }                                            \
}

#define SLIP_MPQ_CLEAR(x)                   \
{                                           \
    SLIP_MPZ_CLEAR(SLIP_MPQ_NUM(x));        \
    SLIP_MPZ_CLEAR(SLIP_MPQ_DEN(x));        \
}

#define SLIP_MPFR_CLEAR(x)                        \
{                                                 \
    if ((x) != NULL && SLIP_MPFR_MANT(x) != NULL) \
    {                                             \
        mpfr_clear(x);                            \
        SLIP_MPFR_SET_NULL(x);                    \
    }                                             \
}


// ============================================================================
//                           Internal Functions
// ============================================================================

// check if SLIP_initialize* has been called
bool slip_initialized ( void ) ;        // true if called, false if not
void slip_set_initialized (bool s) ;    // set global initialzed flag to s

/* Purpose: This function takes as input a mpz_t SLIP_matrix and divides
 * it by an mpz_t constant storing the solution in a mpq_t dense SLIP_matrix
 * array. This is used internally to divide the solution vector by the
 * determinant of the matrix.
 */
SLIP_info slip_matrix_div // divides the x matrix by a scalar
(
    SLIP_matrix **x2_handle,    // x2 = x/scalar
    SLIP_matrix* x,             // input vector x
    const mpz_t scalar,         // the scalar
    const SLIP_options *option
) ;

/* Purpose: This function multiplies matrix x a scalar
 */
SLIP_info slip_matrix_mul   // multiplies x by a scalar
(
    SLIP_matrix *x,         // matrix to be multiplied
    const mpz_t scalar      // scalar to multiply by
) ;

/* Purpose: This function performs sparse REF forward substitution. This is
 * essentially the same as the sparse REF triangular solve applied to each
 * column of the right hand side vectors. Like the normal one, this function
 * expects that the matrix x is dense. As a result,the nonzero pattern is not
 * computed and each nonzero in x is iterated across.  The system to solve is
 * LDx = x.  On output, the mpz_t** x structure is modified.
 */
SLIP_info slip_forward_sub
(
    const SLIP_matrix *L,   // lower triangular matrix
    SLIP_matrix *x,         // right hand side matrix
    const SLIP_matrix *rhos // sequence of pivots used in factorization
);

/* Purpose: This function performs sparse REF backward substitution. In essense
 * it solves the sysem Ux = x. Note that prior to this, we expect x to be
 * multiplied by the determinant of A.  The input argument bx is modified on
 * output.
 */
SLIP_info slip_back_sub  // performs sparse REF backward substitution
(
    const SLIP_matrix *U,   // input upper triangular matrix
    SLIP_matrix *bx        // right hand side matrix of size n*numRHS
)  ;


//------------------------------------------------------------------------------
// mpfr_vector: a 1D mpfr_t array
//------------------------------------------------------------------------------

// Creates a simple 1D array, where A[i] is an entry of type mpfr_t.

/* Purpose: This function creates a MPFR array of desired precision*/
mpfr_t* slip_create_mpfr_array
(
    int64_t n,     // size of the array
    const SLIP_options* option
);

// Creates a simple 1D array, where A[i] is an entry of type mpq_t.

/* Purpose: This function creates an mpq array of size n.
 * This function must be called for all mpq arrays created.
 */
mpq_t* slip_create_mpq_array
(
    int64_t n              // size of the array
);


//------------------------------------------------------------------------------
// mpz_vector: a 1D mpz_t array
//------------------------------------------------------------------------------

// Creates a simple 1D array, where A[i] is an entry of type mpz_t.

/* Purpose: This function creates an mpz array of size n and allocates
 * default size.
 */
mpz_t* slip_create_mpz_array
(
    int64_t n              // Size of x
);

/* SLIP_check_solution: checks the solution of the linear system.  Performs a
 * quick rational arithmetic check of A*x=b.
 */
SLIP_info slip_check_solution
(
    const SLIP_matrix *A,          // input matrix
    const SLIP_matrix *x,          // solution vector
    const SLIP_matrix *b,          // right hand side
    const SLIP_options* option           // Command options
);

/* Purpose: p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1]
 * into c.  This function is lightly modified from CSparse.
 */
SLIP_info slip_cumsum
(
    int64_t *p,          // vector to store the sum of c
    int64_t *c,          // vector which is summed
    int64_t n           // size of c
);

/* Purpose: This function performs a depth first search of the graph of the
 * matrix starting at node j. The output of this function is the set of nonzero
 * indices in the xi vector.
 */
void slip_dfs // performs a dfs of the graph of the matrix starting at node j
(
    int64_t *top,          // beginning of stack
    int64_t j,             // What node to start DFS at
    SLIP_matrix* L,        // matrix which represents the Graph of L
    int64_t* xi,           // the nonzero pattern
    int64_t* pstack,       // workspace vector
    const int64_t* pinv   // row permutation
);

/* Purpose: This function converts a double array of size n to an appropriate
 * mpz array of size n. To do this, the number is multiplied by 10^17 then, the
 * GCD is found. This function allows the use of matrices in double precision
 * to work with SLIP LU.
 */
SLIP_info slip_expand_double_array
(
    mpz_t *x_out,   // integral final array
    double* x,      // double array that needs to be made integral
    mpq_t scale,    // the scaling factor used (x_out = scale * x)
    int64_t n,      // size of x
    const SLIP_options* option
);

/* Purpose: This function converts a mpfr array of size n and precision prec to
 * an appropriate mpz array of size n. To do this, the number is multiplied by
 * the appropriate power of 10 then the gcd is found. This function allows mpfr
 * arrays to be used within SLIP LU.
 */
SLIP_info slip_expand_mpfr_array
(
    mpz_t *x_out,   // integral final array
    mpfr_t* x,      // mpfr array to be expanded
    mpq_t scale,    // scaling factor used (x_out = scale*x)
    int64_t n,      // size of x
    const SLIP_options *option // command options containing the prec for mpfr
);

/* Purpose: This function converts a mpq array of size n into an appropriate mpz
 * array of size n. To do this, the lcm of the denominators is found as a
 * scaling factor. This function allows mpq arrays to be used in SLIP LU
 */
SLIP_info slip_expand_mpq_array
(
    mpz_t *x_out, // integral final array
    mpq_t* x,     // mpq array that needs to be converted
    mpq_t scale,  // scaling factor. x_out = scale*x
    int64_t n,     // size of x
    const SLIP_options* option // Command options
);

/* Purpose: This function converts a mpq matrix of size m*n into an appropriate
 * mpz matrix of size m*n. To do this, the lcm of the denominators is found as a
 * scaling factor. This function allows mpq matrix to be used in SLIP LU.
 * on output, x2 is modified.
 */
SLIP_info slip_expand_mpq_mat
(
    mpz_t **x_out,// integral final mat
    mpq_t **x,    // mpq mat that needs to be converted
    mpq_t scale,  // scaling factor. x_out = scale*x
    int64_t m,    // number of rows of x
    int64_t n     // number of columns of x
);

/* This function performs the pivoting for the SLIP LU factorization.
 * The optional Order is:
 *     0: Smallest pivot
 *     1: Natural/Diagonal pivoting
 *     2: Choose first nonzero (not recommended, for comparison only)
 *     3: Diagonal with tolerance and smallest pivot (default)
 *     4: Diagonal with tolerance and largest pivoting
 *     5: Largest pivot (not recommended, for comparison only)
 *
 * On output, the pivs, pinv, and row_perm arrays and rhos matrix are all modified.
 */
SLIP_info slip_get_pivot
(
    int64_t *pivot,      // found index of pivot entry
    SLIP_matrix* x,      // kth column of L and U
    int64_t* pivs,       // vector indicating which rows have been pivotal
    int64_t n,           // dimension of the problem
    int64_t top,         // nonzero pattern is located in xi[top..n-1]
    int64_t* xi,         // nonzero pattern of x
    int64_t col,         // current column of A (real kth column i.e., q[k])
    int64_t k,           // iteration of the algorithm
    SLIP_matrix* rhos,   // vector of pivots
    int64_t* pinv,       // row permutation
    int64_t* row_perm,   // opposite of pinv. if pinv[i] = j then row_perm[j] = i
    const SLIP_options *option // command option
);

/* Purpose: This function selects the pivot element as the largest in the
 * column. This is activated if the user sets option->pivot = SLIP_LARGEST.
 * NOTE: This pivoting scheme is NOT recommended for SLIP LU.  On output, the
 * index of the largest pivot is returned.
 */
SLIP_info slip_get_largest_pivot
(
    int64_t *pivot,         // the index of largest pivot
    SLIP_matrix* x,         // kth column of L and U
    int64_t* pivs,          // vector which indicates whether each row has been pivotal
    int64_t n,              // dimension of problem
    int64_t top,            // nonzero pattern is located in xi[top..n-1]
    int64_t* xi             // nonzero pattern of x
);

/* This function obtains the first eligible nonzero pivot.  This is enabled if
 * the user sets option->pivot = SLIP_FIRST_NONZERO.  NOTE: This pivoting
 * scheme is not recommended.  On output, the kth pivot is returned.
 */
SLIP_info slip_get_nonzero_pivot // find the first eligible nonzero pivot
(
    int64_t *pivot,      // the index of first eligible nonzero pivot
    SLIP_matrix* x,      // kth column of L and U
    int64_t* pivs,       // vector indicating which rows are pivotal
    int64_t n,           // size of x
    int64_t top,         // nonzero pattern is located in xi[top..n-1]
    int64_t* xi          // nonzero pattern of x
);

/* Purpose: This function selects the pivot element as the smallest in the
 * column. This is activated by default or if the user sets option->pivot =
 * SLIP_SMALLEST.  NOTE: This is the recommended pivoting scheme for SLIP LU.
 * On output, the index of kth pivot is returned.
 */
SLIP_info slip_get_smallest_pivot
(
    int64_t *pivot,         // index of smallest pivot
    SLIP_matrix *x,         // kth column of L and U
    int64_t* pivs,          // vector indicating whether each row has been pivotal
    int64_t n,              // dimension of problem
    int64_t top,            // nonzeros are stored in xi[top..n-1]
    int64_t* xi             // nonzero pattern of x
);

/* Purpose: This function prints the basic info about SLIP_LU library */
void slip_lu_info(void);

/* Purpose: This function permutes b for forward substitution.
 * That is, b = P'*b.
 */
SLIP_info slip_permute_b
(
    SLIP_matrix **b_handle,     // permuted RHS vector
    const SLIP_matrix *b2,      // unpermuted RHS vector (not modified)
    const int64_t *pinv,        // inverse row permutation
    const SLIP_options* option
);

/* Purpose: SLIP_permute_x permutes x to get it back in its original form.
 * That is x = Q*x.
 */
SLIP_info slip_permute_x
(
    SLIP_matrix **x_handle,    // permuted Solution vector
    SLIP_matrix *x2,           // unpermuted Solution vector (not modified)
    SLIP_LU_analysis *S,  // symbolic analysis with the column ordering Q
    const SLIP_options* option  // Command options
                          // has been checked in the only caller SLIP_LU_solve
) ;

/* Purpose: This function collapses a SLIP matrix. Essentially it shrinks the
 * size of x and i. so that they only take up the number of elements in the
 * matrix. For example if A->nzmax = 1000 but nnz(A) = 500, r and x are of size
 * 1000, so this function shrinks them to size 500.
 */
SLIP_info slip_sparse_collapse
(
    SLIP_matrix* A // matrix to be shrunk
);

/* Purpose: This function expands a SLIP LU matrix by doubling its size. It
 * merely expands x and i and does not initialize/allocate the values.
 */
SLIP_info slip_sparse_realloc
(
    SLIP_matrix* A // the matrix to be expanded
);

/* Purpose: This function computes the reach of column k of A on the graph of L
 * mathematically that is: xi = Reach(A(:,k))_G_L.
 */
void slip_reach    // compute the reach of column k of A on the graph of L
(
    int64_t *top,
    SLIP_matrix* L,         // matrix representing graph of L
    const SLIP_matrix* A,   // input matrix
    int64_t k,              // column of A of interest
    int64_t* xi,            // nonzero pattern
    const int64_t* pinv     // row permutation
)  ;

/* Purpose: This function performs the sparse REF triangular solve; that is,
 * (LD) x = A(:,k). The algorithm is described in the paper; however in essence
 * it computes the nonzero pattern xi, then performs a sequence of IPGE
 * operations on the nonzeros to obtain their final value. All operations are
 * gauranteed to be integral. There are various enhancements in this code used
 * to reduce the overall cost of the operations and minimize operations as much
 * as possible.
 */
SLIP_info slip_ref_triangular_solve // performs the sparse REF triangular solve
(
    int64_t *top_output,      // Output the beginning of nonzero pattern
    SLIP_matrix* L,           // partial L matrix
    const SLIP_matrix* A,     // input matrix
    int64_t k,                // iteration of algorithm
    int64_t* xi,              // nonzero pattern vector
    const int64_t* q,         // column permutation
    SLIP_matrix* rhos,        // sequence of pivots
    const int64_t* pinv,      // inverse row permutation
    const int64_t* row_perm,  // row permutation
    int64_t* h,               // history vector
    SLIP_matrix* x            // solution of system ==> kth column of L and U
);

// typecast a double value to int64, accounting for Infs and Nans
static inline int64_t slip_cast_double_to_int64 (double x)
{
    if (isnan (x))
    {
        return (0) ;
    }
    else if (x >= (double) INT64_MAX)
    {
        return (INT64_MAX) ;
    }
    else if (x <= (double) INT64_MIN)
    {
        return (INT64_MIN) ;
    }
    else
    {
        return ((int64_t) (x)) ;
    }
}

SLIP_info slip_cast_array
(
    void *Y,                // output array, of size n
    SLIP_type ytype,        // type of Y
    void *X,                // input array, of size n
    SLIP_type xtype,        // type of X
    int64_t n,              // size of Y and X
    mpq_t y_scale,          // scale factor applied if y is mpz_t
    mpq_t x_scale,          // scale factor applied if x is mpz_t
    const SLIP_options *option
) ;

SLIP_info slip_cast_matrix
(
    SLIP_matrix **Y_handle,     // nz-by-1 dense matrix to create
    SLIP_type Y_type,           // type of Y
    SLIP_matrix *A,             // matrix with nz entries
    const SLIP_options *option
) ;

// (void *) pointer to the values of A.  A must be non-NULL with a valid type
#define SLIP_X(A)                                                           \
    ((A->type == SLIP_MPZ  ) ? (void *) A->x.mpz   :                        \
    ((A->type == SLIP_MPQ  ) ? (void *) A->x.mpq   :                        \
    ((A->type == SLIP_MPFR ) ? (void *) A->x.mpfr  :                        \
    ((A->type == SLIP_INT64) ? (void *) A->x.int64 : (void *) A->x.fp64))))


// return an error if A->kind (csc, triplet, dense) is wrong
#define SLIP_REQUIRE_KIND(A,required_kind) \
    if (A == NULL || A->kind != required_kind) return (SLIP_INCORRECT_INPUT) ;

#define ASSERT_KIND(A,required_kind) \
    ASSERT (A != NULL && A->kind == required_kind)

// return an error if A->type (mpz, mpq, mpfr, int64, or double) is wrong
#define SLIP_REQUIRE_TYPE(A,required_type) \
    if (A == NULL || A->type != required_type) return (SLIP_INCORRECT_INPUT) ;

#define ASSERT_TYPE(A,required_type) \
    ASSERT (A != NULL && A->type == required_type)

// return an error if A->kind or A->type is wrong
#define SLIP_REQUIRE(A,required_kind,required_type)     \
    SLIP_REQUIRE_KIND (A,required_kind) ;               \
    SLIP_REQUIRE_TYPE (A,required_type) ;

#define ASSERT_MATRIX(A,required_kind,required_type)    \
    ASSERT_KIND (A,required_kind) ;                     \
    ASSERT_TYPE (A,required_type) ;

#endif

