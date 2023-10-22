// =============================================================================
// === spqr.hpp ================================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

// Internal definitions and non-user-callable routines.  This should not be
// included in the user's code.

#ifndef SPQR_INTERNAL_H
#define SPQR_INTERNAL_H

// -----------------------------------------------------------------------------
// include files
// -----------------------------------------------------------------------------

#define SUITESPARSE_BLAS_DEFINITIONS
#include "SuiteSparseQR.hpp"
#include "spqr_cholmod_wrappers.hpp"
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <cstring>

// -----------------------------------------------------------------------------
// debugging and printing control
// -----------------------------------------------------------------------------

// force debugging off
#ifndef NDEBUG
#define NDEBUG
#endif

// force printing off
#ifndef NPRINT
#define NPRINT
#endif

// uncomment the following line to turn on debugging (SPQR will be slow!)
/*
#undef NDEBUG
*/

// uncomment the following line to turn on printing (LOTS of output!)
/*
#undef NPRINT
*/

// uncomment the following line to turn on expensive debugging (very slow!)
/*
#define DEBUG_EXPENSIVE
*/

// -----------------------------------------------------------------------------
// basic macros
// -----------------------------------------------------------------------------

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define EMPTY (-1)
#define TRUE 1
#define FALSE 0 
#define IMPLIES(p,q) (!(p) || (q))

// NULL should already be defined, but ensure it is here.
#ifndef NULL
#define NULL ((void *) 0)
#endif

// column-major indexing; A[i,j] is A (INDEX (i,j,lda))
#define INDEX(i,j,lda) ((i) + ((j)*(lda)))

// FLIP is a "negation about -1", and is used to mark an integer i that is
// normally non-negative.  FLIP (EMPTY) is EMPTY.  FLIP of a number > EMPTY
// is negative, and FLIP of a number < EMTPY is positive.  FLIP (FLIP (i)) = i
// for all integers i.  UNFLIP (i) is >= EMPTY.
#define EMPTY (-1)
#define FLIP(i) (-(i)-2)
#define UNFLIP(i) (((i) < EMPTY) ? FLIP (i) : (i))

// -----------------------------------------------------------------------------
// additional include files
// -----------------------------------------------------------------------------

#ifdef MATLAB_MEX_FILE
#include "mex.h"
#endif

#define ITYPE CHOLMOD_LONG
#define DTYPE CHOLMOD_DOUBLE
#define ID "%" PRId64

// -----------------------------------------------------------------------------

#define ERROR(status,msg) \
    cholmod_l_error (status, __FILE__, __LINE__, msg, cc)

// Check a pointer and return if null.  Set status to invalid, unless the
// status is already "out of memory"
#define RETURN_IF_NULL(A,result) \
{ \
    if ((A) == NULL) \
    { \
	if (cc->status != CHOLMOD_OUT_OF_MEMORY) \
	{ \
	    ERROR (CHOLMOD_INVALID, NULL) ; \
	} \
	return (result) ; \
    } \
}

// Return if Common is NULL or invalid
#define RETURN_IF_NULL_COMMON(result) \
{ \
    if (cc == NULL) \
    { \
	return (result) ; \
    } \
}

#define RETURN_IF_XTYPE_INVALID(A,result) \
{ \
    if (A->xtype != xtype) \
    { \
        ERROR (CHOLMOD_INVALID, "invalid xtype") ; \
        return (result) ; \
    } \
}

// -----------------------------------------------------------------------------
// debugging and printing macros
// -----------------------------------------------------------------------------

#ifndef NDEBUG

    #ifdef MATLAB_MEX_FILE

        // #define ASSERT(e) mxAssert (e, "error: ")

        extern char spqr_mx_debug_string [200] ;
        char *spqr_mx_id (int line) ;

        #define ASSERT(e) \
            ((e) ? (void) 0 : \
            mexErrMsgIdAndTxt (spqr_mx_id (__LINE__), \
            "assert: (" #e ") file:"  __FILE__ ))

    #else

        #include <assert.h>
        #define ASSERT(e) assert (e)

    #endif

    #define DEBUG(e) e
    #ifdef DEBUG_EXPENSIVE
        #define DEBUG2(e) e
        #define ASSERT2(e) ASSERT(e)
    #else
        #define DEBUG2(e)
        #define ASSERT2(e)
    #endif

#else

    #define ASSERT(e)
    #define ASSERT2(e)
    #define DEBUG(e)
    #define DEBUG2(e)

#endif

#ifndef NPRINT

    #ifdef MATLAB_MEX_FILE
        #define PR(e) mexPrintf e
    #else
        #define PR(e) printf e
    #endif

    #define PRVAL(e) spqrDebug_print (e)

#else

    #define PR(e)
    #define PRVAL(e)

#endif

// -----------------------------------------------------------------------------
// For counting flops
// -----------------------------------------------------------------------------

#define FLOP_COUNT(f) { if (cc->SPQR_grain <= 1) cc->SPQR_flopcount += ((double) (f)) ; }
#define FLOP_COUNT2(f1,f2) FLOP_COUNT(((double) (f1)) * ((double) (f2)))

// =============================================================================
// === spqr_work ===============================================================
// =============================================================================

// workspace required for each stack in spqr_factorize and spqr_kernel
template <typename Entry, typename Int = int64_t> struct spqr_work
{
    Int *Stair1 ;          // size maxfn if H not kept
    Int *Cmap ;            // size maxfn
    Int *Fmap ;            // size n
    Entry *WTwork ;         // size (fchunk + (keepH ? 0:1)) * maxfn

    Entry *Stack_head ;     // head of Stack
    Entry *Stack_top ;      // top of Stack

    Int sumfrank ;         // sum of ranks of the fronts in this stack
    Int maxfrank ;         // largest rank of fronts in this stack

    // for computing the 2-norm of w, the vector of the dead column norms
    double wscale ;         // scale factor for norm (w (of this stack))
    double wssq ;           // sum-of-squares for norm (w (of this stack))
} ;


// =============================================================================
// === spqr_blob ===============================================================
// =============================================================================

// The spqr_blob is a collection of objects that the spqr_kernel requires.

template <typename Entry, typename Int = int64_t> struct spqr_blob
{
    double tol ;
    spqr_symbolic <Int> *QRsym ;
    spqr_numeric <Entry, Int> *QRnum ;
    spqr_work <Entry, Int> *Work ;
    Int *Cm ;
    Entry **Cblock ;
    Entry *Sx ;
    Int ntol ;
    Int fchunk ;
    cholmod_common *cc ;
} ;


// =============================================================================
// === SuiteSparseQR non-user-callable functions ===============================
// =============================================================================

template <typename Int = int64_t> spqr_symbolic <Int> *spqr_analyze
( 
    // inputs, not modified
    cholmod_sparse *A,
    int ordering,           // all ordering options available
    Int *Quser,            // user provided ordering, if given (may be NULL)

    int do_rank_detection,  // if TRUE, then rank deficient matrices may be
                            // considered during numerical factorization,
    // with tol >= 0 (tol < 0 is also allowed).  If FALSE, then the tol
    // parameter is ignored by the numerical factorization, and no rank
    // detection is performed.

    int keepH,                      // if nonzero, H is kept

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> spqr_numeric <Entry, Int> *spqr_factorize
(
    // input, optionally freed on output
    cholmod_sparse **Ahandle,

    // inputs, not modified
    Int freeA,                     // if TRUE, free A on output
    double tol,                     // for rank detection
    Int ntol,                      // apply tol only to first ntol columns
    spqr_symbolic <Int> *QRsym,

    // workspace and parameters
    cholmod_common *cc
) ;

// returns tol (-1 if error)
template <typename Entry, typename Int = int64_t> double spqr_tol
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> double spqr_maxcolnorm
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> void spqr_kernel
(
    Int task,
    spqr_blob <Entry, Int> *Blob
) ;

template <typename Entry, typename Int = int64_t> void spqr_parallel
(
    Int ntasks,
    int nthreads,
    spqr_blob <Entry, Int> *Blob
) ;

template <typename Int = int64_t> void spqr_freesym
(
    spqr_symbolic <Int> **QRsym_handle,

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> void spqr_freenum
(
    spqr_numeric <Entry, Int> **QRnum_handle,

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> void spqr_freefac
(
    SuiteSparseQR_factorization <Entry, Int> **QR_handle,

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Int = int64_t> void spqr_stranspose1
(
    // input, not modified
    cholmod_sparse *A,  // m-by-n
    Int *Qfill,        // size n, fill-reducing column permutation;
                        // Qfill [k] = j if the kth column of S is the jth
                        // column of A.  Identity permutation is used if
                        // Qfill is NULL.

    // output, contents not defined on input
    Int *Sp,           // size m+1, row pointers of S
    Int *Sj,           // size nz, column indices of S
    Int *PLinv,        // size m, inverse row permutation, PLinv [i] = k
    Int *Sleft,        // size n+2, Sleft [j] ... Sleft [j+1]-1 is the list of
                        // rows of S whose leftmost column index is j.  The list
                        // can be empty (that is, Sleft [j] == Sleft [j+1]).
                        // Sleft [n] is the number of non-empty rows of S, and
                        // Sleft [n+1] is always m.  That is, Sleft [n] ...
                        // Sleft [n+1]-1 gives the empty rows of S.

    // workspace, not defined on input or output
    Int *W             // size m
) ;


template <typename Entry, typename Int = int64_t> void spqr_stranspose2
(
    // input, not modified
    cholmod_sparse *A,  // m-by-n
    Int *Qfill,        // size n, fill-reducing column permutation;
                        // Qfill [k] = j
                        // if the kth column of S is the jth column of A.
                        // Identity permutation is used if Qfill is NULL.

    Int *Sp,           // size m+1, row pointers of S
    Int *PLinv,        // size m, inverse row permutation, PLinv [i] = k

    // output, contents not defined on input
    Entry *Sx,          // size nz, numerical values of S

    // workspace, not defined on input or output
    Int *W             // size m
) ;


// =============================================================================

#ifndef NDEBUG

template <typename Entry, typename Int = int64_t> void spqrDebug_dumpdense
(
    Entry *A,
    Int m,
    Int n,
    Int lda,
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> void spqrDebug_dumpsparse
(
    Int *Ap,
    Int *Ai,
    Entry *Ax,
    Int m,
    Int n,
    cholmod_common *cc
) ;

void spqrDebug_print (double x) ;
void spqrDebug_print (Complex x) ;

template <typename Int = int64_t> 
void spqrDebug_dump_Parent (Int n, Int *Parent, const char *filename) ;

template <typename Int = int64_t> Int spqrDebug_rhsize // returns # of entries in R+H block
(
    // input, not modified
    Int m,                 // # of rows in F
    Int n,                 // # of columns in F
    Int npiv,              // number of pivotal columns in F
    Int *Stair,            // size n; column j is dead if Stair [j] == 0.
                            // Only the first npiv columns can be dead.
    cholmod_common *cc
) ;
#endif

#ifdef DEBUG_EXPENSIVE
template <typename Int = int64_t> Int spqrDebug_listcount
(
    Int x, Int *List, Int len, Int what,
    cholmod_common *cc
) ;
#endif

// =============================================================================

template <typename Int = int64_t> Int spqr_fsize // returns # of rows of F
(
    // inputs, not modified
    Int f,
    Int *Super,            // size nf, from QRsym
    Int *Rp,               // size nf, from QRsym
    Int *Rj,               // size rjsize, from QRsym
    Int *Sleft,            // size n+2, from QRsym
    Int *Child,            // size nf, from QRsym
    Int *Childp,           // size nf+1, from QRsym
    Int *Cm,               // size nf, from QRwork

    // outputs, not defined on input
    Int *Fmap,             // size n, from QRwork
    Int *Stair             // size fn, from QRwork
) ;


template <typename Entry, typename Int = int64_t> void spqr_assemble
(
    // inputs, not modified
    Int f,                 // front to assemble F
    Int fm,                // number of rows of F
    int keepH,              // if TRUE, then construct row pattern of H
    Int *Super,
    Int *Rp,
    Int *Rj,
    Int *Sp,
    Int *Sj,
    Int *Sleft,
    Int *Child,
    Int *Childp,
    Entry *Sx,
    Int *Fmap,
    Int *Cm,
    Entry **Cblock,
#ifndef NDEBUG
    char *Rdead,
#endif
    Int *Hr,

    // input/output
    Int *Stair,
    Int *Hii,              // if keepH, construct list of row indices for F
    // input only
    Int *Hip,

    // outputs, not defined on input
    Entry *F,

    // workspace, not defined on input or output
    Int *Cmap
) ;

template <typename Entry, typename Int = int64_t> Int spqr_cpack // returns # of rows in C
(
    // input, not modified
    Int m,                 // # of rows in F
    Int n,                 // # of columns in F
    Int npiv,              // number of pivotal columns in F
    Int g,                 // the C block starts at F (g,npiv)

    // input, not modified unless the pack occurs in-place
    Entry *F,               // m-by-n frontal matrix in column-major order

    // output, contents not defined on input
    Entry *C                // packed columns of C, of size cm-by-cn in upper
                            // trapezoidal form.
) ;

template <typename Int = int64_t> Int spqr_fcsize // returns # of entries in C of current front F
(
    // input, not modified
    Int m,                 // # of rows in F
    Int n,                 // # of columns in F
    Int npiv,              // number of pivotal columns in F
    Int g                  // the C block starts at F (g,npiv)
) ;

template <typename Int = int64_t> Int spqr_csize // returns # of entries in C of a child
(
    // input, not modified
    Int c,                 // child c
    Int *Rp,               // size nf+1, pointers for pattern of R
    Int *Cm,               // size nf, Cm [c] = # of rows in child C
    Int *Super             // size nf, pivotal columns in each front
) ;

template <typename Entry, typename Int = int64_t> void spqr_rcount
(
    // inputs, not modified
    spqr_symbolic <Int> *QRsym,
    spqr_numeric <Entry, Int> *QRnum,

    Int n1rows,        // added to each row index of Ra and Rb
    Int econ,          // only get entries in rows n1rows to econ-1
    Int n2,            // Ra = R (:,0:n2-1), Rb = R (:,n2:n-1)
    int getT,           // if true, count Rb' instead of Rb

    // input/output
    Int *Ra,           // size n2; Ra [j] += nnz (R (:,j)) if j < n2
    Int *Rb,           // If getT is false: size n-n2 and
                        // Rb [j-n2] += nnz (R (:,j)) if j >= n2.
                        // If getT is true: size econ, and
                        // Rb [i] += nnz (R (i, n2:n-1))
    Int *Hp,           // size rjsize+1.  Column pointers for H.
                        // Only computed if H was kept during factorization.
                        // Only Hp [0..nh] is used.
    Int *p_nh          // number of Householder vectors (nh <= rjsize)
) ;

template <typename Entry, typename Int = int64_t> void spqr_rconvert
(
    // inputs, not modified
    spqr_symbolic <Int> *QRsym,
    spqr_numeric <Entry, Int> *QRnum,

    Int n1rows,        // added to each row index of Ra, Rb, and H
    Int econ,          // only get entries in rows n1rows to econ-1
    Int n2,            // Ra = R (:,0:n2-1), Rb = R (:,n2:n-1)
    int getT,           // if true, get Rb' instead of Rb

    // input/output
    Int *Rap,          // size n2+1; on input, Rap [j] is the column pointer
                        // for Ra.  Incremented on output by the number of
                        // entries added to column j of Ra.

    // output, not defined on input
    Int *Rai,          // size rnz1 = nnz(Ra); row indices of Ra
    Entry *Rax,         // size rnz; numerical values of Ra

    // input/output
    Int *Rbp,          // if getT is false:
                        // size (n-n2)+1; on input, Rbp [j] is the column
                        // pointer for Rb.  Incremented on output by the number
                        // of entries added to column j of Rb.
                        // if getT is true:
                        // size econ+1; on input, Rbp [i] is the row
                        // pointer for Rb.  Incremented on output by the number
                        // of entries added to row i of Rb.

    // output, not defined on input
    Int *Rbi,          // size rnz2 = nnz(Rb); indices of Rb
    Entry *Rbx,         // size rnz2; numerical values of Rb

    // input
    Int *H2p,          // size nh+1; H2p [j] is the column pointer for H.
                        // H2p, H2i, and H2x are ignored if H was not kept
                        // during factorization.  nh computed by rcount

    // output, not defined on input
    Int *H2i,           // size hnz = nnz(H); indices of H
    Entry *H2x,         // size hnz; numerical values of H
    Entry *H2Tau        // size nh; Householder coefficients
) ;

template <typename Entry, typename Int = int64_t> Int spqr_rhpack    // returns # of entries in R+H
(
    // input, not modified
    int keepH,              // if true, then H is packed
    Int m,                 // # of rows in F
    Int n,                 // # of columns in F
    Int npiv,              // number of pivotal columns in F
    Int *Stair,            // size npiv; column j is dead if Stair [j] == 0.
                            // Only the first npiv columns can be dead.

    // input, not modified (unless the pack occurs in-place)
    Entry *F,               // m-by-n frontal matrix in column-major order

    // output, contents not defined on input
    Entry *R,               // packed columns of R+H
    Int *p_rm              // number of rows in R block
) ;

template <typename Entry, typename Int = int64_t> void spqr_hpinv
(
    // input
    spqr_symbolic <Int> *QRsym,
    // input/output
    spqr_numeric <Entry, Int> *QRnum,
    // workspace
    Int *W              // size QRnum->m
) ;

template <typename Entry, typename Int = int64_t> int spqr_1colamd
(
    // inputs, not modified
    int ordering,           // all available, except 0:fixed and 3:given
                            // treated as 1:natural
    double tol,             // only accept singletons above tol
    Int bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // output arrays, neither allocated nor defined on input.

    Int **p_Q1fill,        // size n+bncols, fill-reducing
                            // or natural ordering

    Int **p_R1p,           // size n1rows+1, R1p [k] = # of nonzeros in kth
                            // row of R1.  NULL if n1cols == 0.
    Int **p_P1inv,         // size m, singleton row inverse permutation.
                            // If row i of A is the kth singleton row, then
                            // P1inv [i] = k.  NULL if n1cols is zero.

    cholmod_sparse **p_Y,   // on output, only the first n-n1cols+1 entries of
                            // Y->p are defined (if Y is not NULL), where
                            // Y = [A B] or Y = [A2 B2].  If B is empty and
                            // there are no column singletons, Y is NULL

    Int *p_n1cols,         // number of column singletons found
    Int *p_n1rows,         // number of corresponding rows found

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> int spqr_1fixed
(
    // inputs, not modified
    double tol,             // only accept singletons above tol
    Int bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // output arrays, neither allocated nor defined on input.

    Int **p_R1p,           // size n1rows+1, R1p [k] = # of nonzeros in kth
                            // row of R1.  NULL if n1cols == 0.
    Int **p_P1inv,         // size m, singleton row inverse permutation.
                            // If row i of A is the kth singleton row, then
                            // P1inv [i] = k.  NULL if n1cols is zero.

    cholmod_sparse **p_Y,   // on output, only the first n-n1cols+1 entries of
                            // Y->p are defined (if Y is not NULL), where
                            // Y = [A B] or Y = [A2 B2].  If B is empty and
                            // there are no column singletons, Y is NULL

    Int *p_n1cols,         // number of column singletons found
    Int *p_n1rows,         // number of corresponding rows found

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> 
SuiteSparseQR_factorization <Entry, Int> *spqr_1factor
(
    // inputs, not modified
    int ordering,           // all ordering options available
    double tol,             // only accept singletons above tol
    Int bncols,            // number of columns of B
    int keepH,              // if TRUE, keep the Householder vectors
    cholmod_sparse *A,      // m-by-n sparse matrix
    Int ldb,               // leading dimension of B, if dense
    Int *Bp,               // size bncols+1, column pointers of B
    Int *Bi,               // size bnz = Bp [bncols], row indices of B
    Entry *Bx,              // size bnz, numerical values of B

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Int = int64_t> Int spqr_cumsum // returns total sum
(
    // input, not modified
    Int n,

    // input/output
    Int *X                 // size n+1. X = cumsum ([0 X])
) ;

template <typename Int = int64_t> void spqr_shift
(
    // input, not modified
    Int n,

    // input/output
    Int *X                 // size n+1
) ;

template <typename Entry, typename Int = int64_t> void spqr_larftb
(
    // inputs, not modified (V is modified and then restored on output)
    int method,     // 0,1,2,3
    Int m,         // C is m-by-n
    Int n,
    Int k,         // V is v-by-k
                    // for methods 0 and 1, v = m,
                    // for methods 2 and 3, v = n
    Int ldc,       // leading dimension of C
    Int ldv,       // leading dimension of V
    Entry *V,       // V is v-by-k, unit lower triangular (diag not stored)
    Entry *Tau,     // size k, the k Householder coefficients

    // input/output
    Entry *C,       // C is m-by-n, with leading dimension ldc

    // workspace, not defined on input or output
    Entry *W,       // for methods 0,1: size k*k + n*k
                    // for methods 2,3: size k*k + m*k
    cholmod_common *cc
) ;

template <typename Int = int64_t> int spqr_happly_work
(
    // input
    int method,     // 0,1,2,3 

    Int m,         // X is m-by-n
    Int n,

    // FUTURE : make H cholmod_sparse:
    Int nh,        // number of Householder vectors
    Int *Hp,       // size nh+1, column pointers for H
    Int hchunk, 

    // outputs; sizes of workspaces needed
    Int *p_vmax, 
    Int *p_vsize, 
    Int *p_csize
) ;

template <typename Entry, typename Int = int64_t> void spqr_happly
(
    // input
    int method,     // 0,1,2,3 

    Int m,         // X is m-by-n
    Int n,

    Int nh,        // number of Householder vectors
    Int *Hp,       // size nh+1, column pointers for H
    Int *Hi,       // size hnz = Hp [nh], row indices of H
    Entry *Hx,      // size hnz, Householder values.  Note that the first
                    // entry in each column must be equal to 1.0

    Entry *Tau,     // size nh

    // input/output
    Entry *X,       // size m-by-n with leading dimension m

    // workspace
    Int vmax,
    Int hchunk,
    Int *Wi,       // size vmax
    Int *Wmap,     // size MAX(mh,1) where H is mh-by-nh
    Entry *C,       // size csize
    Entry *V,       // size vsize
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> void spqr_panel
(
    // input
    int method,
    Int m,
    Int n,
    Int v,
    Int h,             // number of Householder vectors in the panel
    Int *Vi,           // Vi [0:v-1] defines the pattern of the panel
    Entry *V,           // v-by-h, panel of Householder vectors
    Entry *Tau,         // size h, Householder coefficients for the panel
    Int ldx,

    // input/output
    Entry *X,           // m-by-n with leading dimension ldx

    // workspace
    Entry *C,           // method 0,1: v-by-n;  method 2,3: m-by-v
    Entry *W,           // method 0,1: k*k+n*k; method 2,3: k*k+m*k

    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> int spqr_append       // TRUE if OK, FALSE otherwise
(
    // inputs, not modified
    Entry *X,       // size m-by-1
    Int *P,        // size m, or NULL; permutation to apply to X.
                    // P [k] = i if row k of A is row i of X

    // input/output
    cholmod_sparse *A,    // size m-by-n2 where n2 > n
    Int *p_n,       // number of columns of A; increased by one

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> Int spqr_trapezoidal // rank of R, or EMPTY
(
    // inputs, not modified

    Int n,         // R is m-by-n (m is not needed here; can be economy R)
    Int *Rp,       // size n+1, column pointers of R
    Int *Ri,       // size rnz = Rp [n], row indices of R
    Entry *Rx,      // size rnz, numerical values of R

    Int bncols,    // number of columns of B

    Int *Qfill,    // size n+bncols, fill-reducing ordering.  Qfill [k] = j if
                    // the jth column of A is the kth column of R.  If Qfill is
                    // NULL, then it is assumed to be the identity
                    // permutation.

    int skip_if_trapezoidal,        // if R is already in trapezoidal form,
                                    // and skip_if_trapezoidal is TRUE, then
                                    // the matrix T is not created.

    // outputs, not allocated on input
    Int **p_Tp,    // size n+1, column pointers of T
    Int **p_Ti,    // size rnz, row indices of T
    Entry **p_Tx,   // size rnz, numerical values of T

    Int **p_Qtrap, // size n+bncols, modified Qfill

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> int spqr_type (void) ;

template <typename Int = int64_t> void *spqr_malloc (size_t n, size_t size, cholmod_common *Common) ;
template <typename Int = int64_t> void *spqr_calloc (size_t n, size_t size, cholmod_common *Common) ;
template <typename Int = int64_t> void *spqr_free (size_t n, size_t size, void *p, cholmod_common *Common) ;

template <typename Int = int64_t> void *spqr_realloc	/* returns pointer to reallocated block */
(
    /* ---- input ---- */
    size_t nnew,	/* requested # of items in reallocated block */
    size_t size,	/* size of each item */
    /* ---- in/out --- */
    void *p,		/* block of memory to realloc */
    size_t *n,		/* current size on input, nnew on output if successful*/
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int = int64_t> cholmod_sparse *spqr_allocate_sparse 
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of A */
    size_t ncol,	/* # of columns of A */
    size_t nzmax,	/* max # of nonzeros of A */
    int sorted,		/* TRUE if columns of A sorted, FALSE otherwise */
    int packed,		/* TRUE if A will be packed, FALSE otherwise */
    int stype,		/* stype of A */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int = int64_t> int spqr_free_sparse
(
    /* ---- in/out --- */
    cholmod_sparse **A,	/* matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int = int64_t> int spqr_reallocate_sparse
(
    /* ---- input ---- */
    size_t nznew,	/* new # of entries in A */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* matrix to reallocate */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int = int64_t> cholmod_sparse *spqr_speye
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of A */
    size_t ncol,	/* # of columns of A */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int = int64_t> cholmod_dense *spqr_allocate_dense
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    size_t d,		/* leading dimension */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int = int64_t> int spqr_free_dense
(
    /* ---- in/out --- */
    cholmod_dense **X,	/* dense matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int = int64_t> cholmod_factor *spqr_allocate_factor
(
    /* ---- input ---- */
    size_t n,		/* L is n-by-n */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int = int64_t> int spqr_free_factor
(
    /* ---- in/out --- */
    cholmod_factor **L,	/* factor to free, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int = int64_t> int spqr_allocate_work
(
    /* ---- input ---- */
    size_t nrow,	/* size: Common->Flag (nrow), Common->Head (nrow+1) */
    size_t iworksize,	/* size of Common->Iwork */
    size_t xworksize,	/* size of Common->Xwork */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int = int64_t> 
int spqr_amd(cholmod_sparse *A, Int *fset, size_t fsize, Int *Perm, cholmod_common *Common) ;

template <typename Int = int64_t> 
int spqr_metis(cholmod_sparse *A, Int *fset, size_t fsize, int postorder, Int *Perm, cholmod_common *Common) ;


template <typename Int = int64_t>
cholmod_sparse *spqr_transpose(cholmod_sparse *A, int values, cholmod_common *Common) ;

template <typename Int = int64_t>
cholmod_factor *spqr_analyze_p2
(
    /* ---- input ---- */
    int for_whom,       /* FOR_SPQR     (0): for SPQR but not GPU-accelerated
                           FOR_CHOLESKY (1): for Cholesky (GPU or not)
                           FOR_SPQRGPU  (2): for SPQR with GPU acceleration */
    cholmod_sparse *A,	/* matrix to order and analyze */
    Int *UserPerm,	/* user-provided permutation, size A->nrow */
    Int *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int = int64_t>
int spqr_colamd
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    Int *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int postorder,	/* if TRUE, follow with a coletree postorder */
    /* ---- output --- */
    Int *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int = int64_t>
int64_t spqr_nnz
(
    cholmod_sparse *A,
    cholmod_common *Common
) ;

template <typename Int = int64_t> Int spqr_postorder	/* return # of nodes postordered */
(
    /* ---- input ---- */
    Int *Parent,	/* size n. Parent [j] = p if p is the parent of j */
    size_t n,
    Int *Weight_p,	/* size n, optional. Weight [j] is weight of node j */
    /* ---- output --- */
    Int *Post,	/* size n. Post [k] = j is kth in postordered tree */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Entry, typename Int = int64_t> void spqr_rsolve
(
    // inputs
    SuiteSparseQR_factorization <Entry, Int> *QR,
    int use_Q1fill,

    Int nrhs,              // number of columns of B
    Int ldb,               // leading dimension of B
    Entry *B,               // size m-by-nrhs with leading dimesion ldb

    // output
    Entry *X,               // size n-by-nrhs with leading dimension n

    // workspace
    Entry **Rcolp,
    Int *Rlive,
    Entry *W,

    cholmod_common *cc
) ;

// returns rank of F, or 0 on error
template <typename Entry, typename Int = int64_t> Int spqr_front
(
    // input, not modified
    Int m,             // F is m-by-n with leading dimension m
    Int n,
    Int npiv,          // number of pivot columns
    double tol,         // a column is flagged as dead if its norm is <= tol
    Int ntol,          // apply tol only to first ntol pivot columns
    Int fchunk,        // block size for compact WY Householder reflections,
                        // treated as 1 if fchunk <= 1

    // input/output
    Entry *F,           // frontal matrix F of size m-by-n
    Int *Stair,        // size n, entries F (Stair[k]:m-1, k) are all zero,
                        // and remain zero on output.
    char *Rdead,        // size npiv; all zero on input.  If k is dead,
                        // Rdead [k] is set to 1

    // output, not defined on input
    Entry *Tau,         // size n, Householder coefficients

    // workspace, undefined on input and output
    Entry *W,           // size b*(n+b), where b = min (fchunk,n,m)

    // input/output
    double *wscale,
    double *wssq,

    cholmod_common *cc
) ;

template <typename Int = int64_t> cholmod_sparse *spqr_dense_to_sparse
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to copy */
    int values,		/* TRUE if values to be copied, FALSE otherwise */
    /* --------------- */
    cholmod_common *Common
) ;
template <typename Int = int64_t> cholmod_dense *spqr_sparse_to_dense
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Entry, typename Int = int64_t> int spqr_rmap
(
    SuiteSparseQR_factorization <Entry, Int> *QR,
    cholmod_common *cc
) ;

// =============================================================================
// === spqrgpu features ========================================================
// =============================================================================

#ifdef SUITESPARSE_CUDA
#include "spqrgpu.hpp"
#endif

// =============================================================================
// === spqr_conj ===============================================================
// =============================================================================

inline double spqr_conj (double x)
{
    return (x) ;
}

inline Complex spqr_conj (Complex x)
{
    return (std::conj (x)) ;
}


// =============================================================================
// === spqr_abs ================================================================
// =============================================================================

inline double spqr_abs (double x, cholmod_common *cc)       // cc is unused
{
    return (fabs (x)) ;
}

inline double spqr_abs (Complex x, cholmod_common *cc)
{
    return (SuiteSparse_config_hypot (x.real ( ), x.imag ( ))) ;
}


// =============================================================================
// === spqr_divide =============================================================
// =============================================================================

inline double spqr_divide (double a, double b, cholmod_common *cc)  // cc unused
{
    return (a/b) ;
}

inline Complex spqr_divide (Complex a, Complex b, cholmod_common *cc)
{
    double creal, cimag ;
    SuiteSparse_config_divcomplex
        (a.real(), a.imag(), b.real(), b.imag(), &creal, &cimag) ;
    return (Complex (creal, cimag)) ;
}


// =============================================================================
// === spqr_add ================================================================
// =============================================================================

// Add two non-negative Int's, and return the result.  Checks for Int
// overflow and sets ok to FALSE if it occurs.

template <typename Int = int64_t> inline Int spqr_add (Int a, Int b, int *ok)
{
    Int c = a + b ;
    if (c < 0)
    {
        (*ok) = FALSE ;
        return (EMPTY) ;
    }
    return (c) ;
}


// =============================================================================
// === spqr_mult ===============================================================
// =============================================================================

// Multiply two positive Int's, and return the result.  Checks for Int
// overflow and sets ok to FALSE if it occurs.

template <typename Int = int64_t> inline Int spqr_mult (Int a, Int b, int *ok)
{
    Int c = a * b ;
    if (((double) c) != ((double) a) * ((double) b))
    {
        (*ok) = FALSE ;
        return (EMPTY) ;
    }
    return (c) ;
}

//------------------------------------------------------------------------------
// test coverage
//------------------------------------------------------------------------------

// SuiteSparse_metis has been modified from the original METIS 5.1.0.  It uses
// the SuiteSparse_config function pointers for malloc/calloc/realloc/free, so
// that it can use the same memory manager functions as the rest of
// SuiteSparse.  However, during test coverage in SPQR/Tcov, the call to
// malloc inside SuiteSparse_metis pretends to fail, to test SPQR's memory
// handling.  This causes METIS to terminate the program.  To avoid this, METIS
// is allowed to use the standard ANSI C11 malloc/calloc/realloc/free functions
// during testing.

#ifdef TEST_COVERAGE

    //--------------------------------------------------------------------------
    // SPQR during test coverage in SPQR/Tcov.
    //--------------------------------------------------------------------------

    void normal_memory_handler (cholmod_common *cc, bool free_work) ;
    void test_memory_handler (cholmod_common *cc, bool free_work) ;
    extern int64_t my_tries, my_punt, save_my_tries, save_my_punt ;

    #define TEST_COVERAGE_PAUSE                                     \
    {                                                               \
        save_my_tries = my_tries ;                                  \
        save_my_punt  = my_punt  ;                                  \
        normal_memory_handler (cc, false) ;                         \
    }

    #define TEST_COVERAGE_RESUME                                    \
    {                                                               \
        test_memory_handler (cc, false) ;                           \
        my_tries = save_my_tries ;                                  \
        my_punt  = save_my_punt  ;                                  \
    }

#else

    //--------------------------------------------------------------------------
    // SPQR in production: no change to SuiteSparse_config
    //--------------------------------------------------------------------------


    #define TEST_COVERAGE_PAUSE
    #define TEST_COVERAGE_RESUME

#endif
#endif

