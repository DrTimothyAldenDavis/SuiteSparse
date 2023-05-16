// =============================================================================
// === SuiteSparseQR ===========================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

//  QR factorization of a sparse matrix, and optionally solve a least squares
//  problem, Q*R = A*E where A is m-by-n, E is a permutation matrix, R is
//  upper triangular if A is full rank, and Q is an orthogonal matrix.
//  R is upper trapezoidal or a "squeezed" upper trapezoidal matrix if A is
//  found to be rank deficient.
//
//  All output arrays are optional.  If you pass NULL for the C, R, E, X,
//  and/or H array handles, those arrays will not be returned.
//
//  The Z output (either sparse or dense) is either C, C', or X where
//  C = Q'*B and X = E*(R\(Q'*B)).  The latter gives the result of X=A\B,
//  which is the least-squares solution if m > n.
//
//  To return full-sized results, set econ = m.  Then C and R will have m rows,
//  and C' will have m columns.
//
//  To return economy-sized results, set econ = n.  Then C and R will have k
//  rows and C' will have k columns, where k = min(m,n).
//
//  To return rank-sized results, set econ = 0.  Then C and R will have k rows
//  and C' will have k columns, where k = r = the estimated rank of A.
//
//  In all three cases, k = max (min (m, econ), r).
//
//  To compute Q, pass in B = speye (m), and then set getCTX to 1
//  (Q is then returned as Zsparse).
//
//  The Householder representation of Q is returned in H, HTau, and HPinv.
//  E is a permutation vector represention of the column permutation, for
//  the factorization Q*R=A*E.


#include "spqr.hpp"

template int32_t SuiteSparseQR <double, int32_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int32_t econ,              // number of rows of C and R to return; a value
                            // less than the rank r of A is treated as r, and
                            // a value greater than m is treated as m.

    int getCTX,             // if 0: return Z = C of size econ-by-bncols
                            // if 1: return Z = C' of size bncols-by-econ
                            // if 2: return Z = X of size econ-by-bncols

    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *Bsparse,
    cholmod_dense *Bdense,

    // output arrays, neither allocated nor defined on input.

    // Z is the matrix C, C', or X
    cholmod_sparse **p_Zsparse,
    cholmod_dense  **p_Zdense,
    cholmod_sparse **p_R,   // the R factor
    int32_t **p_E,             // size n; fill-reducing ordering of A.
    cholmod_sparse **p_H,   // the Householder vectors (m-by-nh)
    int32_t **p_HPinv,         // size m; row permutation for H
    cholmod_dense **p_HTau, // size 1-by-nh, Householder coefficients

    // workspace and parameters
    cholmod_common *cc
) ;

template int64_t SuiteSparseQR <double, int64_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int64_t econ,              // number of rows of C and R to return; a value
                            // less than the rank r of A is treated as r, and
                            // a value greater than m is treated as m.

    int getCTX,             // if 0: return Z = C of size econ-by-bncols
                            // if 1: return Z = C' of size bncols-by-econ
                            // if 2: return Z = X of size econ-by-bncols

    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *Bsparse,
    cholmod_dense *Bdense,

    // output arrays, neither allocated nor defined on input.

    // Z is the matrix C, C', or X
    cholmod_sparse **p_Zsparse,
    cholmod_dense  **p_Zdense,
    cholmod_sparse **p_R,   // the R factor
    int64_t **p_E,             // size n; fill-reducing ordering of A.
    cholmod_sparse **p_H,   // the Householder vectors (m-by-nh)
    int64_t **p_HPinv,         // size m; row permutation for H
    cholmod_dense **p_HTau, // size 1-by-nh, Householder coefficients

    // workspace and parameters
    cholmod_common *cc
) ;

template int32_t SuiteSparseQR <Complex, int32_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int32_t econ,              // number of rows of C and R to return; a value
                            // less than the rank r of A is treated as r, and
                            // a value greater than m is treated as m.

    int getCTX,             // if 0: return Z = C of size econ-by-bncols
                            // if 1: return Z = C' of size bncols-by-econ
                            // if 2: return Z = X of size econ-by-bncols

    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *Bsparse,
    cholmod_dense *Bdense,

    // output arrays, neither allocated nor defined on input.

    // Z is the matrix C, C', or X
    cholmod_sparse **p_Zsparse,
    cholmod_dense  **p_Zdense,
    cholmod_sparse **p_R,   // the R factor
    int32_t **p_E,             // size n; fill-reducing ordering of A.
    cholmod_sparse **p_H,   // the Householder vectors (m-by-nh)
    int32_t **p_HPinv,         // size m; row permutation for H
    cholmod_dense **p_HTau, // size 1-by-nh, Householder coefficients

    // workspace and parameters
    cholmod_common *cc
) ;

template int64_t SuiteSparseQR <Complex, int64_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int64_t econ,              // number of rows of C and R to return; a value
                            // less than the rank r of A is treated as r, and
                            // a value greater than m is treated as m.

    int getCTX,             // if 0: return Z = C of size econ-by-bncols
                            // if 1: return Z = C' of size bncols-by-econ
                            // if 2: return Z = X of size econ-by-bncols

    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *Bsparse,
    cholmod_dense *Bdense,

    // output arrays, neither allocated nor defined on input.

    // Z is the matrix C, C', or X
    cholmod_sparse **p_Zsparse,
    cholmod_dense  **p_Zdense,
    cholmod_sparse **p_R,   // the R factor
    int64_t **p_E,             // size n; fill-reducing ordering of A.
    cholmod_sparse **p_H,   // the Householder vectors (m-by-nh)
    int64_t **p_HPinv,         // size m; row permutation for H
    cholmod_dense **p_HTau, // size 1-by-nh, Householder coefficients

    // workspace and parameters
    cholmod_common *cc
) ;

template cholmod_dense *SuiteSparseQR <double, int32_t>
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;
template cholmod_dense *SuiteSparseQR <double, int64_t>
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;

template cholmod_dense *SuiteSparseQR <Complex, int32_t>
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;
template cholmod_dense *SuiteSparseQR <Complex, int64_t>
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;

template cholmod_dense *SuiteSparseQR <double, int32_t>
(
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;
template cholmod_dense *SuiteSparseQR <double, int64_t>
(
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;

template cholmod_dense *SuiteSparseQR <Complex, int32_t>
(
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;
template cholmod_dense *SuiteSparseQR <Complex, int64_t>
(
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;

template cholmod_sparse *SuiteSparseQR <double, int32_t>
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;

template cholmod_sparse *SuiteSparseQR <double, int64_t>
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;

template cholmod_sparse *SuiteSparseQR <Complex, int32_t>
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;

template cholmod_sparse *SuiteSparseQR <Complex, int64_t>
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
) ;

template int32_t SuiteSparseQR <Complex, int32_t>     // returns rank(A) estimate
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int32_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **Q,     // m-by-e sparse matrix
    cholmod_sparse **R,     // e-by-n sparse matrix
    int32_t **E,               // permutation of 0:n-1
    cholmod_common *cc      // workspace and parameters
) ;
template int64_t SuiteSparseQR <Complex, int64_t>     // returns rank(A) estimate
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int64_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **Q,     // m-by-e sparse matrix
    cholmod_sparse **R,     // e-by-n sparse matrix
    int64_t **E,               // permutation of 0:n-1
    cholmod_common *cc      // workspace and parameters
) ;

template int32_t SuiteSparseQR <double, int32_t>     // returns rank(A) estimate
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int32_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **Q,     // m-by-e sparse matrix where e=max(econ,rank(A))
    cholmod_sparse **R,     // e-by-n sparse matrix
    int32_t **E,               // permutation of 0:n-1
    cholmod_common *cc      // workspace and parameters
) ;
template int64_t SuiteSparseQR <double, int64_t>     // returns rank(A) estimate
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int64_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **Q,     // m-by-e sparse matrix where e=max(econ,rank(A))
    cholmod_sparse **R,     // e-by-n sparse matrix
    int64_t **E,               // permutation of 0:n-1
    cholmod_common *cc      // workspace and parameters
) ;

template int32_t SuiteSparseQR <Complex, int32_t>     // returns rank(A) estimate
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int32_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **R,     // e-by-n sparse matrix
    int32_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;
template int64_t SuiteSparseQR <Complex, int64_t>     // returns rank(A) estimate
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int64_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **R,     // e-by-n sparse matrix
    int64_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;

template int32_t SuiteSparseQR <double, int32_t>     // returns rank(A) estimate
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int32_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **R,     // e-by-n sparse matrix
    int32_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;
template int64_t SuiteSparseQR <double, int64_t>     // returns rank(A) estimate
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int64_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **R,     // e-by-n sparse matrix
    int64_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;

template int32_t SuiteSparseQR <double, int32_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int32_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs dense matrix
    // outputs
    cholmod_dense  **C,     // C = Q'*B, an e-by-nrhs dense matrix
    cholmod_sparse **R,     // e-by-n sparse matrix where e=max(econ,rank(A))
    int32_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;
template int64_t SuiteSparseQR <double, int64_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int64_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs dense matrix
    // outputs
    cholmod_dense  **C,     // C = Q'*B, an e-by-nrhs dense matrix
    cholmod_sparse **R,     // e-by-n sparse matrix where e=max(econ,rank(A))
    int64_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;

template int32_t SuiteSparseQR <Complex, int32_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int32_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs dense matrix
    // outputs
    cholmod_dense  **C,     // C = Q'*B, an e-by-nrhs dense matrix
    cholmod_sparse **R,     // e-by-n sparse matrix where e=max(econ,rank(A))
    int32_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;
template int64_t SuiteSparseQR <Complex, int64_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int64_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs dense matrix
    // outputs
    cholmod_dense  **C,     // C = Q'*B, an e-by-nrhs dense matrix
    cholmod_sparse **R,     // e-by-n sparse matrix where e=max(econ,rank(A))
    int64_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;

// -----------------------------------------------------------------------------
// [C,R,E] = qr(A,B) where C and B are both sparse
// -----------------------------------------------------------------------------

// returns rank(A) estimate if successful, EMPTY otherwise
template <typename Entry, typename Int> Int SuiteSparseQR
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    Int econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-nrhs sparse matrix
    // outputs
    cholmod_sparse **C,     // C = Q'*B, an e-by-nrhs sparse matrix
    cholmod_sparse **R,     // e-by-n sparse matrix where e=max(econ,rank(A))
    Int **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
)
{
    return (SuiteSparseQR <Entry, Int> (ordering, tol, econ, 0, A, B, NULL,
        C, NULL, R, E, NULL, NULL, NULL, cc)) ;
}

template int32_t SuiteSparseQR <double, int32_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int32_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-nrhs sparse matrix
    // outputs
    cholmod_sparse **C,     // C = Q'*B, an e-by-nrhs sparse matrix
    cholmod_sparse **R,     // e-by-n sparse matrix where e=max(econ,rank(A))
    int32_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;
template int64_t SuiteSparseQR <double, int64_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int64_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-nrhs sparse matrix
    // outputs
    cholmod_sparse **C,     // C = Q'*B, an e-by-nrhs sparse matrix
    cholmod_sparse **R,     // e-by-n sparse matrix where e=max(econ,rank(A))
    int64_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;


template int32_t SuiteSparseQR <Complex, int32_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int32_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-nrhs sparse matrix
    // outputs
    cholmod_sparse **C,     // C = Q'*B, an e-by-nrhs sparse matrix
    cholmod_sparse **R,     // e-by-n sparse matrix where e=max(econ,rank(A))
    int32_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;
template int64_t SuiteSparseQR <Complex, int64_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int64_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-nrhs sparse matrix
    // outputs
    cholmod_sparse **C,     // C = Q'*B, an e-by-nrhs sparse matrix
    cholmod_sparse **R,     // e-by-n sparse matrix where e=max(econ,rank(A))
    int64_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
) ;

// -----------------------------------------------------------------------------
// [Q,R,E] = qr(A) where Q is returned in Householder form
// -----------------------------------------------------------------------------

// returns rank(A) estimate if successful, EMPTY otherwise
template <typename Entry, typename Int> Int SuiteSparseQR
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    Int econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **R,     // the R factor
    Int **E,               // permutation of 0:n-1, NULL if identity
    cholmod_sparse **H,     // the Householder vectors (m-by-nh)
    Int **HPinv,           // size m; row permutation for H
    cholmod_dense **HTau,   // size 1-by-nh, Householder coefficients
    cholmod_common *cc      // workspace and parameters
)
{
    return (SuiteSparseQR <Entry, Int> (ordering, tol, econ, EMPTY, A,
        NULL, NULL, NULL, NULL, R, E, H, HPinv, HTau, cc)) ;
}

template int32_t SuiteSparseQR <double, int32_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int32_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **R,     // the R factor
    int32_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_sparse **H,     // the Householder vectors (m-by-nh)
    int32_t **HPinv,           // size m; row permutation for H
    cholmod_dense **HTau,   // size 1-by-nh, Householder coefficients
    cholmod_common *cc      // workspace and parameters
) ;

template int64_t SuiteSparseQR <double, int64_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int64_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **R,     // the R factor
    int64_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_sparse **H,     // the Householder vectors (m-by-nh)
    int64_t **HPinv,           // size m; row permutation for H
    cholmod_dense **HTau,   // size 1-by-nh, Householder coefficients
    cholmod_common *cc      // workspace and parameters
) ;

template int32_t SuiteSparseQR <Complex, int32_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int32_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **R,     // the R factor
    int32_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_sparse **H,     // the Householder vectors (m-by-nh)
    int32_t **HPinv,           // size m; row permutation for H
    cholmod_dense **HTau,   // size 1-by-nh, Householder coefficients
    cholmod_common *cc      // workspace and parameters
) ;

template int64_t SuiteSparseQR <Complex, int64_t>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    int64_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **R,     // the R factor
    int64_t **E,               // permutation of 0:n-1, NULL if identity
    cholmod_sparse **H,     // the Householder vectors (m-by-nh)
    int64_t **HPinv,           // size m; row permutation for H
    cholmod_dense **HTau,   // size 1-by-nh, Householder coefficients
    cholmod_common *cc      // workspace and parameters
) ;