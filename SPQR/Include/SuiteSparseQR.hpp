// =============================================================================
// === SuiteSparseQR.hpp =======================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

// User include file for C++ programs.

#ifndef SUITESPARSEQR_H
#define SUITESPARSEQR_H

// -----------------------------------------------------------------------------
// include files
// -----------------------------------------------------------------------------

#ifdef SUITESPARSE_CUDA
#include <cublas_v2.h>
#endif
#define SUITESPARSE_GPU_EXTERN_ON
extern "C"
{
#include "SuiteSparseQR_definitions.h"
#include "cholmod.h"
}
#undef SUITESPARSE_GPU_EXTERN_ON

#include <complex>
typedef std::complex<double> Complex ;

// =============================================================================
// === spqr_gpu ================================================================
// =============================================================================

template <typename Int = int64_t> struct spqr_gpu_impl
{
    Int *RimapOffsets;      // Stores front offsets into Rimap
    Int RimapSize;          // Allocated space for in Rimap

    Int *RjmapOffsets;      // Stores front offsets into Rjmap
    Int RjmapSize;          // Allocated space for Rjmap

    Int numStages;          // # of Stages required to factorize
    Int *Stagingp;          // Pointers into Post for boundaries
    Int *StageMap;          // Mapping of front to stage #
    size_t *FSize;                       // Total size of fronts in a stage
    size_t *RSize;                       // Total size of R+C for a stage
    size_t *SSize;                       // Total size of S for a stage
    Int *FOffsets;          // F Offsets relative to a base
    Int *ROffsets;          // R Offsets relative to a base
    Int *SOffsets;          // S Offsets relative to a base
};

extern template struct spqr_gpu_impl<int32_t> ;
extern template struct spqr_gpu_impl<int64_t> ;

typedef spqr_gpu_impl<int32_t> spqr_int_gpu ;
typedef spqr_gpu_impl<int64_t> spqr_gpu ;

// =============================================================================
// === spqr_symbolic ===========================================================
// =============================================================================

// The contents of this object do not change during numeric factorization.  The
// Symbolic object depends only on the pattern of the input matrix, and not its
// values.  These contents also do not change with column pivoting for rank
// detection.  This makes parallelism easier to manage, since all threads can
// have access to this object without synchronization.
//
// The total size of the Symbolic object is (10 + 2*m + anz + 2*n + 5*nf + rnz)
// int64_t's, where the user's input A matrix is m-by-n with anz nonzeros, nf
// <= MIN(m,n) is the number of frontal matrices, and rnz <= nnz(R) is the
// number of column indices used to represent the supernodal form of R (one
// int64_t per non-pivotal column index in the leading row of each block of R).

template <typename Int = int64_t> struct spqr_symbolic
{

    // -------------------------------------------------------------------------
    // row-form of the input matrix and its permutations
    // -------------------------------------------------------------------------

    // During symbolic analysis, the nonzero pattern of S = A(P,Q) is
    // constructed, where A is the user's input matrix.  Its numerical values
    // are also constructed, but they do not become part of the Symbolic
    // object.  The matrix S is stored in row-oriented form.  The rows of S are
    // sorted according to their leftmost column index (via PLinv).  Column
    // indices in each row of S are in strictly ascending order, even though
    // the input matrix A need not be sorted.

    Int m, n, anz ; // S is m-by-n with anz entries

    Int *Sp ;       // size m+1, row pointers of S

    Int *Sj ;       // size anz = Sp [n], column indices of S

    Int *Qfill ;    // size n, fill-reducing column permutation.
                        // Qfill [k] = j if column k of A is column j of S.

    Int *PLinv ;    // size m, inverse row permutation that places
                        // S=A(P,Q) in increasing order of leftmost column
                        // index.  PLinv [i] = k if row i of A is row k of S.

    Int *Sleft ;    // size n+2.  The list of rows of S whose
            // leftmost column index is j is given by
            // Sleft [j] ... Sleft [j+1]-1.  This can be empty (that is, Sleft
            // [j] can equal Sleft [j+1]).  Sleft [n] is the number of
            // non-empty rows of S, and Sleft [n+1] == m.  That is, Sleft [n]
            // ... Sleft [n+1]-1 gives the empty rows of S, if any.

    // -------------------------------------------------------------------------
    // frontal matrices: pattern and tree
    // -------------------------------------------------------------------------

    // Each frontal matrix is fm-by-fn, with fnpiv pivot columns.  The fn
    // column indices are given by a set of size fnpiv pivot columns, defined
    // by Super, followed by the pattern Rj [ Rp[f] ...  Rp[f+1]-1 ].

    // The row indices of the front are not kept.  If the Householder vectors
    // are not kept, the row indices are not needed.  If the Householder
    // vectors are kept, the row indices are computed dynamically during
    // numerical factorization.

    Int nf ;        // number of frontal matrices; nf <= MIN (m,n)
    Int maxfn ;     // max # of columns in any front

    // parent, child, and childp define the row merge tree or etree (A'A)
    Int *Parent ;   // size nf+1
    Int *Child ;    // size nf+1
    Int *Childp ;   // size nf+2

    // The parent of a front f is Parent [f], or EMPTY if f=nf.
    // A list of children of f can be obtained in the list
    // Child [Childp [f] ... Childp [f+1]-1].

    // Node nf in the tree is a placeholder; it does not represent a frontal
    // matrix.  All roots of the frontal "tree" (may be a forest) have the
    // placeholder node nf as their parent.  Thus, the tree of nodes 0:nf is
    // truly a tree, with just one parent (node nf).

    Int *Super ;    // size nf+1.  Super [f] gives the first
        // pivot column in the front F.  This refers to a column of S.  The
        // number of expected pivot columns in F is thus
        // Super [f+1] - Super [f].

    Int *Rp ;       // size nf+1
    Int *Rj ;       // size rjsize; compressed supernodal form of R

    Int *Post ;     // size nf+1, post ordering of frontal tree.
                        // f=Post[k] gives the kth node in the postordered tree

    Int rjsize ;    // size of Rj

    Int do_rank_detection ; // TRUE: allow for tol >= 0.
                                         // FALSE: ignore tol

    // the rest depends on whether or not rank-detection is allowed:
    Int maxstack  ; // max stack size (sequential case)
    Int hisize ;    // size of Hii

    Int keepH ;     // TRUE if H is present

    Int *Hip ;      // size nf+1.  If H is kept, the row indices
        // of frontal matrix f are in Hii [Hip [f] ... Hip [f] + Hm [f]],
        // where Hii and Hm are stored in the numeric object.

        // There is one block row of R per frontal matrix.
        // The fn column indices of R are given by Rj [Rp [f] ... Rp [f+1]-1],
        // where the first fp column indices are Super [f] ... Super [f+1]-1.
        // The remaining column indices in Rj [...] are non-pivotal, and are
        // in the range Super [f+1] to n.  The number of rows of R is at
        // most fp, but can be less if dead columns appear in the matrix.
        // The number of columns in the contribution block C is always
        // cn = fn - fp, where fn = Rp [f+1] - Rp [f].

    Int ntasks ;    // number of tasks in task graph
    Int ns ;        // number of stacks

    // -------------------------------------------------------------------------
    // the rest of the QR symbolic object is present only if ntasks > 1
    // -------------------------------------------------------------------------

    // Task tree (nodes 0:ntasks), including placeholder node
    Int *TaskChildp ;       // size ntasks+2
    Int *TaskChild ;        // size ntasks+1

    Int *TaskStack ;        // size ntasks+1

    // list of fronts for each task
    Int *TaskFront ;        // size nf+1
    Int *TaskFrontp  ;      // size ntasks+2

    Int *On_stack  ;        // size nf+1, front f is on
                                         // stack On_stack [f]

    // size of each stack
    Int *Stack_maxstack ;   // size ns+2

    // number of rows for each front
    Int *Fm ;               // size nf+1

    // number of rows in the contribution block of each front
    Int *Cm ;               // size nf+1

    // from CHOLMOD's supernodal analysis, needed for GPU factorization
    size_t maxcsize ;
    size_t maxesize ;
    Int *ColCount ;
    // int64_t *px ;

    // -------------------------------------------------------------------------
    // GPU structure
    // -------------------------------------------------------------------------

    // This is NULL if the GPU is not in use.  The GPU must be enabled at
    // compile time (-DSUITESPARSE_CUDA enables the GPU).  If the Householder vectors
    // are requested or if rank detection is requested, then the GPU is
    // disabled.

    spqr_gpu_impl <Int> *QRgpu ;

} ;


// =============================================================================
// === spqr_numeric ============================================================
// =============================================================================

// The Numeric object contains the numerical values of the triangular/
// trapezoidal factor R, and optionally the Householder vectors H if they
// are kept.

template <typename Entry, typename Int = int64_t> struct spqr_numeric
{

    // -------------------------------------------------------------------------
    // Numeric R factor
    // -------------------------------------------------------------------------

    Entry **Rblock ;    // size nf.  R [f] is an (Entry *) pointer to the
                        // R block for front F.  It is an upper trapezoidal
                        // of size Rm(f)-by-Rn(f), but only the upper
                        // triangular part is stored in column-packed format.

    Entry **Stacks ;    // size ns; an array of stacks holding the R and H
                        // factors and the current frontal matrix F at the head.
                        // This is followed by empty space, then the C blocks of
                        // prior frontal matrices at the bottom.  When the
                        // factorization is complete, only the R and H part at
                        // the head of each stack is left.

    Int *Stack_size ;   // size ns; Stack_size [s] is the size
                                     // of Stacks [s]

    Int hisize ;        // size of Hii

    Int n ;             // A is m-by-n
    Int m ;
    Int nf ;            // number of frontal matrices
    Int ntasks ;        // # of tasks in task graph actually used
    Int ns ;            // number of stacks actually used
    Int maxstack ;      // size of sequential stack, if used

    // -------------------------------------------------------------------------
    // for rank detection and m < n case
    // -------------------------------------------------------------------------

    char *Rdead ;       // size n, Rdead [k] = 1 if k is a dead pivot column,
                        // Rdead [k] = 0 otherwise.  If no columns are dead,
                        // this is NULL.  If m < n, then at least m-n columns
                        // will be dead.

    Int rank ;      // number of live pivot columns
    Int rank1 ;     // number of live pivot columns in first ntol
                                 // columns of A

    Int maxfrank ;  // max number of rows in any R block

    double norm_E_fro ; // 2-norm of w, the vector of dead column 2-norms

    // -------------------------------------------------------------------------
    // for keeping Householder vectors
    // -------------------------------------------------------------------------

    // The factorization is R = (H_s * ... * H_2 * H_1) * P_H
    // where P_H is the permutation HPinv, and H_1, ... H_s are the Householder
    // vectors (s = rjsize).

    Int keepH ;     // TRUE if H is present

    Int rjsize ;    // size of Hstair and HTau

    Int *HStair ;   // size rjsize.  The list Hstair [Rp [f] ...
                        // Rp [f+1]-1] gives the staircase for front F

    Entry *HTau ;       // size rjsize.  The list HTau [Rp [f] ... Rp [f+1]-1]
                        // gives the Householder coefficients for front F

    Int *Hii ;      // size hisize, row indices of H.

    Int *HPinv ;    // size m.  HPinv [i] = k if row i of A and H
                        // is row k of R.  This permutation includes
                        // QRsym->PLinv, and the permutation constructed via
                        // pivotal row ordering during factorization.

    Int *Hm ;       // size nf, Hm [f] = # of rows in front F
    Int *Hr ;       // size nf, Hr [f] = # of rows in R block of
                                 // front F
    Int maxfm ;     // max (Hm [0:nf-1]), computed only if H kept

} ;
extern template struct spqr_numeric <double, int32_t>;
extern template struct spqr_numeric <Complex, int32_t>;

extern template struct spqr_numeric <double, int64_t>;
extern template struct spqr_numeric <Complex, int64_t>;

// =============================================================================
// === SuiteSparseQR_factorization =============================================
// =============================================================================

// A combined symbolic+numeric QR factorization of A or [A B],
// with singletons

template <typename Entry, typename Int = int64_t> struct SuiteSparseQR_factorization
{

    // QR factorization of A or [A Binput] after singletons have been removed
    double tol ;        // tol used
    spqr_symbolic <Int> *QRsym ;
    spqr_numeric <Entry, Int> *QRnum ;

    // singletons, in compressed-row form; R is n1rows-by-n
    Int *R1p ;      // size n1rows+1
    Int *R1j ;
    Entry *R1x ;
    Int r1nz ;      // nnz (R1)

    // combined singleton and fill-reducing permutation
    Int *Q1fill ;
    Int *P1inv ;
    Int *HP1inv ;   // NULL if n1cols == 0, in which case
                        // QRnum->HPinv serves in its place.

    // Rmap and RmapInv are NULL if QR->rank == A->ncol
    Int *Rmap ;     // size n.  Rmap [j] = k if column j of R is
                        // the kth live column and where k < QR->rank;
                        // otherwise, if j is a dead column, then
                        // k >= QR->rank.

    Int *RmapInv ;

    Int n1rows ;    // number of singleton rows of [A B]
    Int n1cols ;    // number of singleton columns of [A B]

    Int narows ;    // number of rows of A
    Int nacols ;    // number of columns of A
    Int bncols ;    // number of columns of B
    Int rank ;      // rank estimate of A (n1rows + QRnum->rank1),
                                 // ranges from 0 to min(m,n)

    int allow_tol ;     // if TRUE, do rank detection
} ;


// =============================================================================
// === Simple user-callable SuiteSparseQR functions ============================
// =============================================================================

//  SuiteSparseQR           Sparse QR factorization and solve
//  SuiteSparseQR_qmult     Q'*X, Q*X, X*Q', or X*Q for X full or sparse

// returns rank(A) estimate, or EMPTY on failure
template <typename Entry, typename Int = int64_t> Int SuiteSparseQR
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // only accept singletons above tol

    Int econ,  // number of rows of C and R to return; a value
                            // less than the rank r of A is treated as r, and
                            // a value greater than m is treated as m.

    int getCTX,             // if 0: return Z = C of size econ-by-bncols
                            // if 1: return Z = C' of size bncols-by-econ
                            // if 2: return Z = X of size econ-by-bncols

    cholmod_sparse *A,      // m-by-n sparse matrix

    // B is either sparse or dense.  If Bsparse is non-NULL, B is sparse and
    // Bdense is ignored.  If Bsparse is NULL and Bdense is non-NULL, then B is
    // dense.  B is not present if both are NULL.
    cholmod_sparse *Bsparse,
    cholmod_dense *Bdense,

    // output arrays, neither allocated nor defined on input.

    // Z is the matrix C, C', or X
    cholmod_sparse **Zsparse,
    cholmod_dense  **Zdense,
    cholmod_sparse **R,     // the R factor
    Int **E,   // size n; fill-reducing ordering of A.
    cholmod_sparse **H,     // the Householder vectors (m-by-nh)
    Int **HPinv,// size m; row permutation for H
    cholmod_dense **HTau,   // size nh, Householder coefficients

    // workspace and parameters
    cholmod_common *cc
) ;

// X = A\dense(B)
template <typename Entry, typename Int = int64_t> cholmod_dense *SuiteSparseQR
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;

// X = A\dense(B) using default ordering and tolerance
template <typename Entry, typename Int = int64_t> cholmod_dense *SuiteSparseQR
(
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;

// X = A\sparse(B)
template <typename Entry, typename Int = int64_t> cholmod_sparse *SuiteSparseQR
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;

// [Q,R,E] = qr(A), returning Q as a sparse matrix
template <typename Entry, typename Int = int64_t> Int SuiteSparseQR
    // returns rank(A) estimate
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,
    Int econ,
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **Q,     // m-by-e sparse matrix where e=max(econ,rank(A))
    cholmod_sparse **R,     // e-by-n sparse matrix
    Int **E,   // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;

// [Q,R,E] = qr(A), discarding Q
template <typename Entry, typename Int = int64_t> Int SuiteSparseQR
    // returns rank(A) estimate
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,
    Int econ,
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **R,     // e-by-n sparse matrix
    Int **E,   // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;

// [C,R,E] = qr(A,B), where C and B are dense
template <typename Entry, typename Int = int64_t> Int SuiteSparseQR
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // only accept singletons above tol
    Int econ,  // number of rows of C and R to return
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs dense matrix
    // outputs
    cholmod_dense  **C,     // C = Q'*B, an e-by-nrhs dense matrix
    cholmod_sparse **R,     // e-by-n sparse matrix where e=max(econ,rank(A))
    Int **E,   // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;

// [C,R,E] = qr(A,B), where C and B are sparse
template <typename Entry, typename Int = int64_t> Int SuiteSparseQR
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // only accept singletons above tol
    Int econ,  // number of rows of C and R to return
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-nrhs sparse matrix
    // outputs
    cholmod_sparse **C,     // C = Q'*B, an e-by-nrhs sparse matrix
    cholmod_sparse **R,     // e-by-n sparse matrix where e=max(econ,rank(A))
    Int **E,   // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;

// [Q,R,E] = qr(A) where Q is returned in Householder form
template <typename Entry, typename Int = int64_t> Int SuiteSparseQR
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // only accept singletons above tol
    Int econ,  // number of rows of C and R to return
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **R,     // the R factor
    Int **E,   // permutation of 0:n-1, NULL if identity
    cholmod_sparse **H,     // the Householder vectors (m-by-nh)
    Int **HPinv,// size m; row permutation for H
    cholmod_dense **HTau,   // size nh, Householder coefficients
    cholmod_common *cc      // workspace and parameters
) ;

// =============================================================================
// === SuiteSparseQR_qmult =====================================================
// =============================================================================

// This function takes as input the matrix Q in Householder form, as returned
// by SuiteSparseQR (... H, HPinv, HTau, cc) above.

// returns Y of size m-by-n (NULL on failure)
template <typename Entry, typename Int = int64_t> cholmod_dense *SuiteSparseQR_qmult
(
    // inputs, no modified
    int method,         // 0,1,2,3
    cholmod_sparse *H,      // either m-by-nh or n-by-nh
    cholmod_dense *HTau,    // size 1-by-nh
    Int *HPinv,// size mh
    cholmod_dense *Xdense,  // size m-by-n

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> cholmod_sparse *SuiteSparseQR_qmult
(
    // inputs, no modified
    int method,             // 0,1,2,3
    cholmod_sparse *H,      // either m-by-nh or n-by-nh
    cholmod_dense *HTau,    // size 1-by-nh
    Int *HPinv,// size mh
    cholmod_sparse *X,

    // workspace and parameters
    cholmod_common *cc
) ;

// =============================================================================
// === Expert user-callable SuiteSparseQR functions ============================
// =============================================================================

#ifndef NEXPERT

// These functions are "expert" routines, allowing reuse of the QR
// factorization for different right-hand-sides.  They also allow the user to
// find the minimum 2-norm solution to an undertermined system of equations.

template <typename Entry, typename Int = int64_t>
SuiteSparseQR_factorization <Entry, Int> *SuiteSparseQR_factorize
(
    // inputs, not modified:
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // treat columns with 2-norm <= tol as zero
    cholmod_sparse *A,      // sparse matrix to factorize
    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> cholmod_dense *SuiteSparseQR_solve    // returns X
(
    // inputs, not modified:
    int system,                 // which system to solve
    SuiteSparseQR_factorization <Entry, Int> *QR, // of an m-by-n sparse matrix A
    cholmod_dense *B,           // right-hand-side, m-by-nrhs or n-by-nrhs
    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> cholmod_sparse *SuiteSparseQR_solve    // returns X
(
    // inputs, not modified:
    int system,                 // which system to solve (0,1,2,3)
    SuiteSparseQR_factorization <Entry, Int> *QR, // of an m-by-n sparse matrix A
    cholmod_sparse *Bsparse,    // right-hand-side, m-by-nrhs or n-by-nrhs
    // workspace and parameters
    cholmod_common *cc
) ;

// returns Y of size m-by-n, or NULL on failure
template <typename Entry, typename Int = int64_t> cholmod_dense *SuiteSparseQR_qmult
(
    // inputs, not modified
    int method,                 // 0,1,2,3 (same as SuiteSparseQR_qmult)
    SuiteSparseQR_factorization <Entry, Int> *QR, // of an m-by-n sparse matrix A
    cholmod_dense *Xdense,      // size m-by-n with leading dimension ldx
    // workspace and parameters
    cholmod_common *cc
) ;

// returns Y of size m-by-n, or NULL on failure
template <typename Entry, typename Int = int64_t> cholmod_sparse *SuiteSparseQR_qmult
(
    // inputs, not modified
    int method,                 // 0,1,2,3
    SuiteSparseQR_factorization <Entry, Int> *QR, // of an m-by-n sparse matrix A
    cholmod_sparse *Xsparse,    // size m-by-n
    // workspace and parameters
    cholmod_common *cc
) ;

// free the QR object
template <typename Entry, typename Int = int64_t> int SuiteSparseQR_free
(
    SuiteSparseQR_factorization <Entry, Int> **QR, // of an m-by-n sparse matrix A
    cholmod_common *cc
) ;

// find the min 2-norm solution to a sparse linear system
template <typename Entry, typename Int = int64_t> cholmod_dense *SuiteSparseQR_min2norm
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,
    cholmod_sparse *A,
    cholmod_dense *B,
    cholmod_common *cc
) ;

template <typename Entry, typename Int = int64_t> cholmod_sparse *SuiteSparseQR_min2norm
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,
    cholmod_sparse *A,
    cholmod_sparse *B,
    cholmod_common *cc
) ;

// symbolic QR factorization; no singletons exploited
template <typename Entry, typename Int = int64_t>
SuiteSparseQR_factorization <Entry, Int> *SuiteSparseQR_symbolic
(
    // inputs:
    int ordering,           // all, except 3:given treated as 0:fixed
    int allow_tol,          // if FALSE, tol is ignored by the numeric
                            // factorization, and no rank detection is performed
    cholmod_sparse *A,      // sparse matrix to factorize (A->x ignored)
    cholmod_common *cc      // workspace and parameters
) ;

// numeric QR factorization;
template <typename Entry, typename Int = int64_t> int SuiteSparseQR_numeric
(
    // inputs:
    double tol,             // treat columns with 2-norm <= tol as zero
    cholmod_sparse *A,      // sparse matrix to factorize
    // input/output
    SuiteSparseQR_factorization <Entry, Int> *QR,
    cholmod_common *cc      // workspace and parameters
) ;

// forward declare template instantiations

extern template int SuiteSparseQR_numeric <double, int32_t>
(
    double tol, cholmod_sparse *A,
    SuiteSparseQR_factorization <double, int32_t> *QR, cholmod_common *cc
) ;

extern template int SuiteSparseQR_numeric <Complex, int32_t>
(
    double tol, cholmod_sparse *A,
    SuiteSparseQR_factorization <Complex, int32_t> *QR, cholmod_common *cc
) ;

extern template int SuiteSparseQR_numeric <double, int64_t>
(
    double tol, cholmod_sparse *A,
    SuiteSparseQR_factorization <double, int64_t> *QR, cholmod_common *cc
) ;

extern template int SuiteSparseQR_numeric <Complex, int64_t>
(
    double tol, cholmod_sparse *A,
    SuiteSparseQR_factorization <Complex, int64_t> *QR, cholmod_common *cc
) ;
#endif

#endif
