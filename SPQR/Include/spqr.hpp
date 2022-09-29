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

#include "SuiteSparseQR.hpp"
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <cstring>

#include <complex>
typedef std::complex<double> Complex ;

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
    if (cc->itype != ITYPE || cc->dtype != DTYPE) \
    { \
	cc->status = CHOLMOD_INVALID ; \
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

#define FLOP_COUNT(f) { if (cc->SPQR_grain <= 1) cc->SPQR_flopcount += (f) ; }

// =============================================================================
// === spqr_work ===============================================================
// =============================================================================

// workspace required for each stack in spqr_factorize and spqr_kernel
template <typename Entry> struct spqr_work
{
    int64_t *Stair1 ;          // size maxfn if H not kept
    int64_t *Cmap ;            // size maxfn
    int64_t *Fmap ;            // size n
    Entry *WTwork ;         // size (fchunk + (keepH ? 0:1)) * maxfn

    Entry *Stack_head ;     // head of Stack
    Entry *Stack_top ;      // top of Stack

    int64_t sumfrank ;         // sum of ranks of the fronts in this stack
    int64_t maxfrank ;         // largest rank of fronts in this stack

    // for computing the 2-norm of w, the vector of the dead column norms
    double wscale ;         // scale factor for norm (w (of this stack))
    double wssq ;           // sum-of-squares for norm (w (of this stack))
} ;


// =============================================================================
// === spqr_blob ===============================================================
// =============================================================================

// The spqr_blob is a collection of objects that the spqr_kernel requires.

template <typename Entry> struct spqr_blob
{
    double tol ;
    spqr_symbolic *QRsym ;
    spqr_numeric <Entry> *QRnum ;
    spqr_work <Entry> *Work ;
    int64_t *Cm ;
    Entry **Cblock ;
    Entry *Sx ;
    int64_t ntol ;
    int64_t fchunk ;
    cholmod_common *cc ;
} ;


// =============================================================================
// === SuiteSparseQR non-user-callable functions ===============================
// =============================================================================

spqr_symbolic *spqr_analyze
( 
    // inputs, not modified
    cholmod_sparse *A,
    int ordering,           // all ordering options available
    int64_t *Quser,            // user provided ordering, if given (may be NULL)

    int do_rank_detection,  // if TRUE, then rank deficient matrices may be
                            // considered during numerical factorization,
    // with tol >= 0 (tol < 0 is also allowed).  If FALSE, then the tol
    // parameter is ignored by the numerical factorization, and no rank
    // detection is performed.

    int keepH,                      // if nonzero, H is kept

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> spqr_numeric <Entry> *spqr_factorize
(
    // input, optionally freed on output
    cholmod_sparse **Ahandle,

    // inputs, not modified
    int64_t freeA,                     // if TRUE, free A on output
    double tol,                     // for rank detection
    int64_t ntol,                      // apply tol only to first ntol columns
    spqr_symbolic *QRsym,

    // workspace and parameters
    cholmod_common *cc
) ;

// returns tol (-1 if error)
template <typename Entry> double spqr_tol
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> double spqr_maxcolnorm
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> void spqr_kernel
(
    int64_t task,
    spqr_blob <Entry> *Blob
) ;

template <typename Entry> void spqr_parallel
(
    int64_t ntasks,
    int nthreads,
    spqr_blob <Entry> *Blob
) ;

void spqr_freesym
(
    spqr_symbolic **QRsym_handle,

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> void spqr_freenum
(
    spqr_numeric <Entry> **QRnum_handle,

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> void spqr_freefac
(
    SuiteSparseQR_factorization <Entry> **QR_handle,

    // workspace and parameters
    cholmod_common *cc
) ;

void spqr_stranspose1
(
    // input, not modified
    cholmod_sparse *A,  // m-by-n
    int64_t *Qfill,        // size n, fill-reducing column permutation;
                        // Qfill [k] = j if the kth column of S is the jth
                        // column of A.  Identity permutation is used if
                        // Qfill is NULL.

    // output, contents not defined on input
    int64_t *Sp,           // size m+1, row pointers of S
    int64_t *Sj,           // size nz, column indices of S
    int64_t *PLinv,        // size m, inverse row permutation, PLinv [i] = k
    int64_t *Sleft,        // size n+2, Sleft [j] ... Sleft [j+1]-1 is the list of
                        // rows of S whose leftmost column index is j.  The list
                        // can be empty (that is, Sleft [j] == Sleft [j+1]).
                        // Sleft [n] is the number of non-empty rows of S, and
                        // Sleft [n+1] is always m.  That is, Sleft [n] ...
                        // Sleft [n+1]-1 gives the empty rows of S.

    // workspace, not defined on input or output
    int64_t *W             // size m
) ;


template <typename Entry> void spqr_stranspose2
(
    // input, not modified
    cholmod_sparse *A,  // m-by-n
    int64_t *Qfill,        // size n, fill-reducing column permutation;
                        // Qfill [k] = j
                        // if the kth column of S is the jth column of A.
                        // Identity permutation is used if Qfill is NULL.

    int64_t *Sp,           // size m+1, row pointers of S
    int64_t *PLinv,        // size m, inverse row permutation, PLinv [i] = k

    // output, contents not defined on input
    Entry *Sx,          // size nz, numerical values of S

    // workspace, not defined on input or output
    int64_t *W             // size m
) ;


// =============================================================================

#ifndef NDEBUG

template <typename Entry> void spqrDebug_dumpdense
(
    Entry *A,
    int64_t m,
    int64_t n,
    int64_t lda,
    cholmod_common *cc
) ;

template <typename Entry> void spqrDebug_dumpsparse
(
    int64_t *Ap,
    int64_t *Ai,
    Entry *Ax,
    int64_t m,
    int64_t n,
    cholmod_common *cc
) ;

void spqrDebug_print (double x) ;
void spqrDebug_print (Complex x) ;

void spqrDebug_dump_Parent (int64_t n, int64_t *Parent, const char *filename) ;

int64_t spqrDebug_rhsize             // returns # of entries in R+H block
(
    // input, not modified
    int64_t m,                 // # of rows in F
    int64_t n,                 // # of columns in F
    int64_t npiv,              // number of pivotal columns in F
    int64_t *Stair,            // size n; column j is dead if Stair [j] == 0.
                            // Only the first npiv columns can be dead.
    cholmod_common *cc
) ;
#endif

#ifdef DEBUG_EXPENSIVE
int64_t spqrDebug_listcount
(
    int64_t x, int64_t *List, int64_t len, int64_t what,
    cholmod_common *cc
) ;
#endif

// =============================================================================

int64_t spqr_fsize     // returns # of rows of F
(
    // inputs, not modified
    int64_t f,
    int64_t *Super,            // size nf, from QRsym
    int64_t *Rp,               // size nf, from QRsym
    int64_t *Rj,               // size rjsize, from QRsym
    int64_t *Sleft,            // size n+2, from QRsym
    int64_t *Child,            // size nf, from QRsym
    int64_t *Childp,           // size nf+1, from QRsym
    int64_t *Cm,               // size nf, from QRwork

    // outputs, not defined on input
    int64_t *Fmap,             // size n, from QRwork
    int64_t *Stair             // size fn, from QRwork
) ;


template <typename Entry> void spqr_assemble
(
    // inputs, not modified
    int64_t f,                 // front to assemble F
    int64_t fm,                // number of rows of F
    int keepH,              // if TRUE, then construct row pattern of H
    int64_t *Super,
    int64_t *Rp,
    int64_t *Rj,
    int64_t *Sp,
    int64_t *Sj,
    int64_t *Sleft,
    int64_t *Child,
    int64_t *Childp,
    Entry *Sx,
    int64_t *Fmap,
    int64_t *Cm,
    Entry **Cblock,
#ifndef NDEBUG
    char *Rdead,
#endif
    int64_t *Hr,

    // input/output
    int64_t *Stair,
    int64_t *Hii,              // if keepH, construct list of row indices for F
    // input only
    int64_t *Hip,

    // outputs, not defined on input
    Entry *F,

    // workspace, not defined on input or output
    int64_t *Cmap
) ;

template <typename Entry> int64_t spqr_cpack     // returns # of rows in C
(
    // input, not modified
    int64_t m,                 // # of rows in F
    int64_t n,                 // # of columns in F
    int64_t npiv,              // number of pivotal columns in F
    int64_t g,                 // the C block starts at F (g,npiv)

    // input, not modified unless the pack occurs in-place
    Entry *F,               // m-by-n frontal matrix in column-major order

    // output, contents not defined on input
    Entry *C                // packed columns of C, of size cm-by-cn in upper
                            // trapezoidal form.
) ;

int64_t spqr_fcsize    // returns # of entries in C of current front F
(
    // input, not modified
    int64_t m,                 // # of rows in F
    int64_t n,                 // # of columns in F
    int64_t npiv,              // number of pivotal columns in F
    int64_t g                  // the C block starts at F (g,npiv)
) ;

int64_t spqr_csize     // returns # of entries in C of a child
(
    // input, not modified
    int64_t c,                 // child c
    int64_t *Rp,               // size nf+1, pointers for pattern of R
    int64_t *Cm,               // size nf, Cm [c] = # of rows in child C
    int64_t *Super             // size nf, pivotal columns in each front
) ;

template <typename Entry> void spqr_rcount
(
    // inputs, not modified
    spqr_symbolic *QRsym,
    spqr_numeric <Entry> *QRnum,

    int64_t n1rows,        // added to each row index of Ra and Rb
    int64_t econ,          // only get entries in rows n1rows to econ-1
    int64_t n2,            // Ra = R (:,0:n2-1), Rb = R (:,n2:n-1)
    int getT,           // if true, count Rb' instead of Rb

    // input/output
    int64_t *Ra,           // size n2; Ra [j] += nnz (R (:,j)) if j < n2
    int64_t *Rb,           // If getT is false: size n-n2 and
                        // Rb [j-n2] += nnz (R (:,j)) if j >= n2.
                        // If getT is true: size econ, and
                        // Rb [i] += nnz (R (i, n2:n-1))
    int64_t *Hp,           // size rjsize+1.  Column pointers for H.
                        // Only computed if H was kept during factorization.
                        // Only Hp [0..nh] is used.
    int64_t *p_nh          // number of Householder vectors (nh <= rjsize)
) ;

template <typename Entry> void spqr_rconvert
(
    // inputs, not modified
    spqr_symbolic *QRsym,
    spqr_numeric <Entry> *QRnum,

    int64_t n1rows,        // added to each row index of Ra, Rb, and H
    int64_t econ,          // only get entries in rows n1rows to econ-1
    int64_t n2,            // Ra = R (:,0:n2-1), Rb = R (:,n2:n-1)
    int getT,           // if true, get Rb' instead of Rb

    // input/output
    int64_t *Rap,          // size n2+1; on input, Rap [j] is the column pointer
                        // for Ra.  Incremented on output by the number of
                        // entries added to column j of Ra.

    // output, not defined on input
    int64_t *Rai,          // size rnz1 = nnz(Ra); row indices of Ra
    Entry *Rax,         // size rnz; numerical values of Ra

    // input/output
    int64_t *Rbp,          // if getT is false:
                        // size (n-n2)+1; on input, Rbp [j] is the column
                        // pointer for Rb.  Incremented on output by the number
                        // of entries added to column j of Rb.
                        // if getT is true:
                        // size econ+1; on input, Rbp [i] is the row
                        // pointer for Rb.  Incremented on output by the number
                        // of entries added to row i of Rb.

    // output, not defined on input
    int64_t *Rbi,          // size rnz2 = nnz(Rb); indices of Rb
    Entry *Rbx,         // size rnz2; numerical values of Rb

    // input
    int64_t *H2p,          // size nh+1; H2p [j] is the column pointer for H.
                        // H2p, H2i, and H2x are ignored if H was not kept
                        // during factorization.  nh computed by rcount

    // output, not defined on input
    int64_t *H2i,           // size hnz = nnz(H); indices of H
    Entry *H2x,         // size hnz; numerical values of H
    Entry *H2Tau        // size nh; Householder coefficients
) ;

template <typename Entry> int64_t spqr_rhpack    // returns # of entries in R+H
(
    // input, not modified
    int keepH,              // if true, then H is packed
    int64_t m,                 // # of rows in F
    int64_t n,                 // # of columns in F
    int64_t npiv,              // number of pivotal columns in F
    int64_t *Stair,            // size npiv; column j is dead if Stair [j] == 0.
                            // Only the first npiv columns can be dead.

    // input, not modified (unless the pack occurs in-place)
    Entry *F,               // m-by-n frontal matrix in column-major order

    // output, contents not defined on input
    Entry *R,               // packed columns of R+H
    int64_t *p_rm              // number of rows in R block
) ;

template <typename Entry> void spqr_hpinv
(
    // input
    spqr_symbolic *QRsym,
    // input/output
    spqr_numeric <Entry> *QRnum,
    // workspace
    int64_t *W              // size QRnum->m
) ;

template <typename Entry> int spqr_1colamd
(
    // inputs, not modified
    int ordering,           // all available, except 0:fixed and 3:given
                            // treated as 1:natural
    double tol,             // only accept singletons above tol
    int64_t bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // output arrays, neither allocated nor defined on input.

    int64_t **p_Q1fill,        // size n+bncols, fill-reducing
                            // or natural ordering

    int64_t **p_R1p,           // size n1rows+1, R1p [k] = # of nonzeros in kth
                            // row of R1.  NULL if n1cols == 0.
    int64_t **p_P1inv,         // size m, singleton row inverse permutation.
                            // If row i of A is the kth singleton row, then
                            // P1inv [i] = k.  NULL if n1cols is zero.

    cholmod_sparse **p_Y,   // on output, only the first n-n1cols+1 entries of
                            // Y->p are defined (if Y is not NULL), where
                            // Y = [A B] or Y = [A2 B2].  If B is empty and
                            // there are no column singletons, Y is NULL

    int64_t *p_n1cols,         // number of column singletons found
    int64_t *p_n1rows,         // number of corresponding rows found

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> int spqr_1fixed
(
    // inputs, not modified
    double tol,             // only accept singletons above tol
    int64_t bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // output arrays, neither allocated nor defined on input.

    int64_t **p_R1p,           // size n1rows+1, R1p [k] = # of nonzeros in kth
                            // row of R1.  NULL if n1cols == 0.
    int64_t **p_P1inv,         // size m, singleton row inverse permutation.
                            // If row i of A is the kth singleton row, then
                            // P1inv [i] = k.  NULL if n1cols is zero.

    cholmod_sparse **p_Y,   // on output, only the first n-n1cols+1 entries of
                            // Y->p are defined (if Y is not NULL), where
                            // Y = [A B] or Y = [A2 B2].  If B is empty and
                            // there are no column singletons, Y is NULL

    int64_t *p_n1cols,         // number of column singletons found
    int64_t *p_n1rows,         // number of corresponding rows found

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> SuiteSparseQR_factorization <Entry> *spqr_1factor
(
    // inputs, not modified
    int ordering,           // all ordering options available
    double tol,             // only accept singletons above tol
    int64_t bncols,            // number of columns of B
    int keepH,              // if TRUE, keep the Householder vectors
    cholmod_sparse *A,      // m-by-n sparse matrix
    int64_t ldb,               // leading dimension of B, if dense
    int64_t *Bp,               // size bncols+1, column pointers of B
    int64_t *Bi,               // size bnz = Bp [bncols], row indices of B
    Entry *Bx,              // size bnz, numerical values of B

    // workspace and parameters
    cholmod_common *cc
) ;

int64_t spqr_cumsum            // returns total sum
(
    // input, not modified
    int64_t n,

    // input/output
    int64_t *X                 // size n+1. X = cumsum ([0 X])
) ;

void spqr_shift
(
    // input, not modified
    int64_t n,

    // input/output
    int64_t *X                 // size n+1
) ;

template <typename Entry> void spqr_larftb
(
    // inputs, not modified (V is modified and then restored on output)
    int method,     // 0,1,2,3
    int64_t m,         // C is m-by-n
    int64_t n,
    int64_t k,         // V is v-by-k
                    // for methods 0 and 1, v = m,
                    // for methods 2 and 3, v = n
    int64_t ldc,       // leading dimension of C
    int64_t ldv,       // leading dimension of V
    Entry *V,       // V is v-by-k, unit lower triangular (diag not stored)
    Entry *Tau,     // size k, the k Householder coefficients

    // input/output
    Entry *C,       // C is m-by-n, with leading dimension ldc

    // workspace, not defined on input or output
    Entry *W,       // for methods 0,1: size k*k + n*k
                    // for methods 2,3: size k*k + m*k
    cholmod_common *cc
) ;

int spqr_happly_work
(
    // input
    int method,     // 0,1,2,3 

    int64_t m,         // X is m-by-n
    int64_t n,

    // FUTURE : make H cholmod_sparse:
    int64_t nh,        // number of Householder vectors
    int64_t *Hp,       // size nh+1, column pointers for H
    int64_t hchunk, 

    // outputs; sizes of workspaces needed
    int64_t *p_vmax, 
    int64_t *p_vsize, 
    int64_t *p_csize
) ;

template <typename Entry> void spqr_happly
(
    // input
    int method,     // 0,1,2,3 

    int64_t m,         // X is m-by-n
    int64_t n,

    int64_t nh,        // number of Householder vectors
    int64_t *Hp,       // size nh+1, column pointers for H
    int64_t *Hi,       // size hnz = Hp [nh], row indices of H
    Entry *Hx,      // size hnz, Householder values.  Note that the first
                    // entry in each column must be equal to 1.0

    Entry *Tau,     // size nh

    // input/output
    Entry *X,       // size m-by-n with leading dimension m

    // workspace
    int64_t vmax,
    int64_t hchunk,
    int64_t *Wi,       // size vmax
    int64_t *Wmap,     // size MAX(mh,1) where H is mh-by-nh
    Entry *C,       // size csize
    Entry *V,       // size vsize
    cholmod_common *cc
) ;

template <typename Entry> void spqr_panel
(
    // input
    int method,
    int64_t m,
    int64_t n,
    int64_t v,
    int64_t h,             // number of Householder vectors in the panel
    int64_t *Vi,           // Vi [0:v-1] defines the pattern of the panel
    Entry *V,           // v-by-h, panel of Householder vectors
    Entry *Tau,         // size h, Householder coefficients for the panel
    int64_t ldx,

    // input/output
    Entry *X,           // m-by-n with leading dimension ldx

    // workspace
    Entry *C,           // method 0,1: v-by-n;  method 2,3: m-by-v
    Entry *W,           // method 0,1: k*k+n*k; method 2,3: k*k+m*k

    cholmod_common *cc
) ;

template <typename Entry> int spqr_append       // TRUE if OK, FALSE otherwise
(
    // inputs, not modified
    Entry *X,       // size m-by-1
    int64_t *P,        // size m, or NULL; permutation to apply to X.
                    // P [k] = i if row k of A is row i of X

    // input/output
    cholmod_sparse *A,    // size m-by-n2 where n2 > n
    int64_t *p_n,       // number of columns of A; increased by one

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> int64_t spqr_trapezoidal       // rank of R, or EMPTY
(
    // inputs, not modified

    int64_t n,         // R is m-by-n (m is not needed here; can be economy R)
    int64_t *Rp,       // size n+1, column pointers of R
    int64_t *Ri,       // size rnz = Rp [n], row indices of R
    Entry *Rx,      // size rnz, numerical values of R

    int64_t bncols,    // number of columns of B

    int64_t *Qfill,    // size n+bncols, fill-reducing ordering.  Qfill [k] = j if
                    // the jth column of A is the kth column of R.  If Qfill is
                    // NULL, then it is assumed to be the identity
                    // permutation.

    int skip_if_trapezoidal,        // if R is already in trapezoidal form,
                                    // and skip_if_trapezoidal is TRUE, then
                                    // the matrix T is not created.

    // outputs, not allocated on input
    int64_t **p_Tp,    // size n+1, column pointers of T
    int64_t **p_Ti,    // size rnz, row indices of T
    Entry **p_Tx,   // size rnz, numerical values of T

    int64_t **p_Qtrap, // size n+bncols, modified Qfill

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> int spqr_type (void) ;

template <typename Entry> void spqr_rsolve
(
    // inputs
    SuiteSparseQR_factorization <Entry> *QR,
    int use_Q1fill,

    int64_t nrhs,              // number of columns of B
    int64_t ldb,               // leading dimension of B
    Entry *B,               // size m-by-nrhs with leading dimesion ldb

    // output
    Entry *X,               // size n-by-nrhs with leading dimension n

    // workspace
    Entry **Rcolp,
    int64_t *Rlive,
    Entry *W,

    cholmod_common *cc
) ;

// returns rank of F, or 0 on error
template <typename Entry> int64_t spqr_front
(
    // input, not modified
    int64_t m,             // F is m-by-n with leading dimension m
    int64_t n,
    int64_t npiv,          // number of pivot columns
    double tol,         // a column is flagged as dead if its norm is <= tol
    int64_t ntol,          // apply tol only to first ntol pivot columns
    int64_t fchunk,        // block size for compact WY Householder reflections,
                        // treated as 1 if fchunk <= 1

    // input/output
    Entry *F,           // frontal matrix F of size m-by-n
    int64_t *Stair,        // size n, entries F (Stair[k]:m-1, k) are all zero,
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

template <typename Entry> int spqr_rmap
(
    SuiteSparseQR_factorization <Entry> *QR,
    cholmod_common *cc
) ;

// =============================================================================
// === spqrgpu features ========================================================
// =============================================================================

#ifdef GPU_BLAS
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
    return (SuiteSparse_config.hypot_func (x.real ( ), x.imag ( ))) ;
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
    SuiteSparse_config.divcomplex_func
        (a.real(), a.imag(), b.real(), b.imag(), &creal, &cimag) ;
    return (Complex (creal, cimag)) ;
}


// =============================================================================
// === spqr_add ================================================================
// =============================================================================

// Add two non-negative int64_t's, and return the result.  Checks for int64_t
// overflow and sets ok to FALSE if it occurs.

inline int64_t spqr_add (int64_t a, int64_t b, int *ok)
{
    int64_t c = a + b ;
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

// Multiply two positive int64_t's, and return the result.  Checks for int64_t
// overflow and sets ok to FALSE if it occurs.

inline int64_t spqr_mult (int64_t a, int64_t b, int *ok)
{
    int64_t c = a * b ;
    if (((double) c) != ((double) a) * ((double) b))
    {
        (*ok) = FALSE ;
        return (EMPTY) ;
    }
    return (c) ;
}

#endif
