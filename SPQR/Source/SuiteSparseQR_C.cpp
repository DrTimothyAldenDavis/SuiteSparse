// =============================================================================
// === SuiteSparseQR_C =========================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// This C++ file provides a set of C-callable wrappers so that a C program can
// call SuiteSparseQR.

#include "spqr.hpp"
#include "SuiteSparseQR_C.h"

extern "C" {

// =============================================================================
// === SuiteSparseQR_C =========================================================
// =============================================================================

// Primary sparse QR function, with all inputs/outputs available.  The primary
// uses of this function are to perform any one of the the MATLAB equivalent
// statements:
//
//      X = A\B                 % where B is sparse or dense
//      [C,R,E] = qr (A,B)      % where Q*R=A*E and C=Q'*B
//      [Q,R,E] = qr (A)        % with Q in Householder form (H, HPinv, HTau)
//      [Q,R,E] = qr (A)        % where Q is discarded (the "Q-less" QR)
//      R = qr (A)              % as above, but with E=I
//
// To obtain the factor Q in sparse matrix form, use SuiteSparseQR_C_QR instead.

SuiteSparse_long SuiteSparseQR_C  // returns rank(A) estimate, (-1) if failure
(
    // inputs:
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as 0
    SuiteSparse_long econ,  // e = max(min(m,econ),rank(A))
    int getCTX,             // 0: Z=C (e-by-k), 1: Z=C', 2: Z=X (e-by-k)
    cholmod_sparse *A,      // m-by-n sparse matrix to factorize
    cholmod_sparse *Bsparse,// sparse m-by-k B
    cholmod_dense  *Bdense, // dense  m-by-k B
    // outputs:
    cholmod_sparse **Zsparse,   // sparse Z
    cholmod_dense  **Zdense,    // dense Z
    cholmod_sparse **R,     // R factor, e-by-n
    SuiteSparse_long **E,   // size n column permutation, NULL if identity
    cholmod_sparse **H,     // m-by-nh Householder vectors
    SuiteSparse_long **HPinv, // size m row permutation
    cholmod_dense **HTau,   // 1-by-nh Householder coefficients
    cholmod_common *cc      // workspace and parameters
)
{
    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_NULL (A, EMPTY) ;
    cc->status = CHOLMOD_OK ;

    return ((A->xtype == CHOLMOD_REAL) ?
        SuiteSparseQR <double, SuiteSparse_long>  (ordering, tol, econ, getCTX, A, Bsparse,
            Bdense, Zsparse, Zdense, R, E, H, HPinv, HTau, cc) :
        SuiteSparseQR <Complex, SuiteSparse_long> (ordering, tol, econ, getCTX, A, Bsparse,
            Bdense, Zsparse, Zdense, R, E, H, HPinv, HTau, cc)) ;
}

int32_t SuiteSparseQR_i_C    // returns rank(A) estimate, (-1) if failure
(
    // inputs:
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as 0
    int32_t econ,              // e = max(min(m,econ),rank(A))
    int getCTX,             // 0: Z=C (e-by-k), 1: Z=C', 2: Z=X (e-by-k)
    cholmod_sparse *A,      // m-by-n sparse matrix to factorize
    cholmod_sparse *Bsparse,// sparse m-by-k B
    cholmod_dense  *Bdense, // dense  m-by-k B
    // outputs:
    cholmod_sparse **Zsparse,   // sparse Z
    cholmod_dense  **Zdense,    // dense Z
    cholmod_sparse **R,     // R factor, e-by-n
    int32_t **E,               // size n column permutation, NULL if identity
    cholmod_sparse **H,     // m-by-nh Householder vectors
    int32_t **HPinv,           // size m row permutation
    cholmod_dense **HTau,   // 1-by-nh Householder coefficients
    cholmod_common *cc      // workspace and parameters
)
{
    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_NULL (A, EMPTY) ;
    cc->status = CHOLMOD_OK ;

    return ((A->xtype == CHOLMOD_REAL) ?
        SuiteSparseQR <double, int32_t>  (ordering, tol, econ, getCTX, A, Bsparse,
            Bdense, Zsparse, Zdense, R, E, H, HPinv, HTau, cc) :
        SuiteSparseQR <Complex, int32_t> (ordering, tol, econ, getCTX, A, Bsparse,
            Bdense, Zsparse, Zdense, R, E, H, HPinv, HTau, cc)) ;
}

// =============================================================================
// === SuiteSparseQR_C_QR ======================================================
// =============================================================================

// [Q,R,E] = qr(A), returning Q as a sparse matrix

SuiteSparse_long SuiteSparseQR_C_QR          // returns rank(A) estimate, (-1) if failure
(
    // inputs:
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as 0
    SuiteSparse_long econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix to factorize
    // outputs:
    cholmod_sparse **Q,     // m-by-e sparse matrix
    cholmod_sparse **R,     // e-by-n sparse matrix
    SuiteSparse_long **E,               // size n column permutation, NULL if identity
    cholmod_common *cc      // workspace and parameters
)
{
    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_NULL (A, EMPTY) ;
    cc->status = CHOLMOD_OK ;

    return ((A->xtype == CHOLMOD_REAL) ?
        SuiteSparseQR <double>  (ordering, tol, econ, A, Q, R, E, cc) :
        SuiteSparseQR <Complex> (ordering, tol, econ, A, Q, R, E, cc)) ;
}

int32_t SuiteSparseQR_i_C_QR          // returns rank(A) estimate, (-1) if failure
(
    // inputs:
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as 0
    int32_t econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix to factorize
    // outputs:
    cholmod_sparse **Q,     // m-by-e sparse matrix
    cholmod_sparse **R,     // e-by-n sparse matrix
    int32_t **E,               // size n column permutation, NULL if identity
    cholmod_common *cc      // workspace and parameters
)
{
    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_NULL (A, EMPTY) ;
    cc->status = CHOLMOD_OK ;

    return ((A->xtype == CHOLMOD_REAL) ?
        SuiteSparseQR <double, int32_t>  (ordering, tol, econ, A, Q, R, E, cc) :
        SuiteSparseQR <Complex, int32_t> (ordering, tol, econ, A, Q, R, E, cc)) ;
}
// =============================================================================
// === SuiteSparseQR_C_backslash ===============================================
// =============================================================================

// X = A\B where B is dense
cholmod_dense *SuiteSparseQR_C_backslash    // returns X, NULL if failure
(
    // inputs:
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as 0
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-k
    cholmod_common *cc      // workspace and parameters
)
{
    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (A, NULL) ;
    RETURN_IF_NULL (B, NULL) ;
    cc->status = CHOLMOD_OK ;
    if (A->itype == CHOLMOD_INT)
    {
        return ((A->xtype == CHOLMOD_REAL) ?
        SuiteSparseQR <double, int32_t>  (ordering, tol, A, B, cc) :
        SuiteSparseQR <Complex, int32_t> (ordering, tol, A, B, cc)) ;
    }
    else
    {
        return ((A->xtype == CHOLMOD_REAL) ?
        SuiteSparseQR <double, SuiteSparse_long>  (ordering, tol, A, B, cc) :
        SuiteSparseQR <Complex, SuiteSparse_long> (ordering, tol, A, B, cc)) ;
    }
}


// =============================================================================
// === SuiteSparseQR_C_backslash_default =======================================
// =============================================================================

// X = A\B where B is dense, using default ordering and tol
cholmod_dense *SuiteSparseQR_C_backslash_default   // returns X, NULL if failure
(
    // inputs:
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-k
    cholmod_common *cc      // workspace and parameters
)
{
    return (SuiteSparseQR_C_backslash (SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL,
        A, B, cc)) ;
}

// =============================================================================
// === SuiteSparseQR_C_backslash_sparse ========================================
// =============================================================================

// X = A\B where B is sparse
cholmod_sparse *SuiteSparseQR_C_backslash_sparse   // returns X, NULL if failure
(
    // inputs:
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as 0
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-k
    cholmod_common *cc      // workspace and parameters
)
{
    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (A, NULL) ;
    RETURN_IF_NULL (B, NULL) ;
    cc->status = CHOLMOD_OK ;
    if (A->itype == CHOLMOD_INT)
    {
        return ((A->xtype == CHOLMOD_REAL) ?
            SuiteSparseQR <double, int32_t>  (ordering, tol, A, B, cc) :
            SuiteSparseQR <Complex, int32_t> (ordering, tol, A, B, cc)) ;
    }
    else
    {
        return ((A->xtype == CHOLMOD_REAL) ?
            SuiteSparseQR <double, SuiteSparse_long>  (ordering, tol, A, B, cc) :
            SuiteSparseQR <Complex, SuiteSparse_long> (ordering, tol, A, B, cc)) ;
    }
}

#ifndef NEXPERT

// =============================================================================
// === C wrappers for expert routines ==========================================
// =============================================================================

// =============================================================================
// === SuiteSparseQR_C_factorize ===============================================
// =============================================================================

SuiteSparseQR_C_factorization *SuiteSparseQR_C_factorize
(
    // inputs:
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as 0
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_common *cc      // workspace and parameters
)
{
    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (A, NULL) ;
    cc->status = CHOLMOD_OK ;

    SuiteSparseQR_C_factorization *QR ;
    QR = (SuiteSparseQR_C_factorization *) (A->itype == CHOLMOD_INT ?
        spqr_malloc <int32_t> (1, sizeof (SuiteSparseQR_C_factorization), cc) :
        spqr_malloc <SuiteSparse_long> (1, sizeof (SuiteSparseQR_C_factorization), cc)) ;
        
    if (cc->status < CHOLMOD_OK)
    {
        return (NULL) ;
    }
    QR->xtype = A->xtype ;
    QR->itype = A->itype ;

    if (A->itype == CHOLMOD_INT)
    {
        QR->factors = (A->xtype == CHOLMOD_REAL) ?
        ((void *) SuiteSparseQR_factorize <double, int32_t>  (ordering, tol, A, cc)) :
        ((void *) SuiteSparseQR_factorize <Complex, int32_t> (ordering, tol, A, cc)) ;
    }
    else
    {
        QR->factors = (A->xtype == CHOLMOD_REAL) ?
        ((void *) SuiteSparseQR_factorize <double, SuiteSparse_long>  (ordering, tol, A, cc)) :
        ((void *) SuiteSparseQR_factorize <Complex, SuiteSparse_long> (ordering, tol, A, cc)) ;
    }

    if (cc->status < CHOLMOD_OK)
    {
        SuiteSparseQR_C_free (&QR, cc) ;
    }
    return (QR) ;
}

// =============================================================================
// === SuiteSparseQR_C_symbolic ================================================
// =============================================================================

SuiteSparseQR_C_factorization *SuiteSparseQR_C_symbolic
(
    // inputs:
    int ordering,           // all, except 3:given treated as 0:fixed
    int allow_tol,          // if FALSE, tol is ignored by the numeric
                            // factorization, and no rank detection is performed
    cholmod_sparse *A,      // sparse matrix to factorize (A->x ignored)
    cholmod_common *cc      // workspace and parameters
)
{
    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (A, NULL) ;
    cc->status = CHOLMOD_OK ;

    SuiteSparseQR_C_factorization *QR ;
    QR = (SuiteSparseQR_C_factorization *) (A->itype == CHOLMOD_INT ?
        spqr_malloc <int32_t> (1, sizeof (SuiteSparseQR_C_factorization), cc) :
        spqr_malloc <SuiteSparse_long> (1, sizeof (SuiteSparseQR_C_factorization), cc)) ;
    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        return (NULL) ;
    }
    QR->xtype = A->xtype ;
    QR->itype = A->itype ;
    if (A->itype == CHOLMOD_INT)
    {
        QR->factors = (A->xtype == CHOLMOD_REAL) ?
        ((void *) SuiteSparseQR_symbolic <double, int32_t>  (ordering, allow_tol, A, cc)) :
        ((void *) SuiteSparseQR_symbolic <Complex, int32_t> (ordering, allow_tol, A, cc)) ;
    }
    else
    {
        QR->factors = (A->xtype == CHOLMOD_REAL) ?
        ((void *) SuiteSparseQR_symbolic <double, SuiteSparse_long>  (ordering, allow_tol, A, cc)) :
        ((void *) SuiteSparseQR_symbolic <Complex, SuiteSparse_long> (ordering, allow_tol, A, cc)) ;
    }
    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        SuiteSparseQR_C_free (&QR, cc) ;
    }
    return (QR) ;
}

// =============================================================================
// === SuiteSparseQR_C_numeric =================================================
// =============================================================================

// numeric QR factorization; must be preceded by a call to
// SuiteSparseQR_C_symbolic.

int SuiteSparseQR_C_numeric // returns TRUE if successful, FALSE otherwise
(
    // inputs:
    double tol,             // treat columns with 2-norm <= tol as zero
    cholmod_sparse *A,      // sparse matrix to factorize
    // input/output:
    SuiteSparseQR_C_factorization *QR,
    cholmod_common *cc      // workspace and parameters
)
{
    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (A, FALSE) ;
    RETURN_IF_NULL (QR, FALSE) ;
    cc->status = CHOLMOD_OK ;

    if (QR->xtype == CHOLMOD_REAL)
    {
        if (QR->itype == CHOLMOD_INT)
        {
            SuiteSparseQR_factorization <double, int32_t> *QR2 ;
            QR2 = (SuiteSparseQR_factorization <double, int32_t> *) (QR->factors) ;
            SuiteSparseQR_numeric (tol, A, QR2, cc) ;
        }
        else
        {
            SuiteSparseQR_factorization <double, SuiteSparse_long> *QR2 ;
            QR2 = (SuiteSparseQR_factorization <double, SuiteSparse_long> *) (QR->factors) ;
            SuiteSparseQR_numeric (tol, A, QR2, cc) ;
        }
    }
    else
    {
        if (QR->itype == CHOLMOD_INT)
        {
            SuiteSparseQR_factorization <Complex, int32_t> *QR2 ;
            QR2 = (SuiteSparseQR_factorization <Complex, int32_t> *) (QR->factors) ;
            SuiteSparseQR_numeric (tol, A, QR2, cc) ;
        }
        else
        {
            SuiteSparseQR_factorization <Complex, SuiteSparse_long> *QR2 ;
            QR2 = (SuiteSparseQR_factorization <Complex, SuiteSparse_long> *) (QR->factors) ;
            SuiteSparseQR_numeric (tol, A, QR2, cc) ;
        }
    }
    return (TRUE) ;
}

// =============================================================================
// === SuiteSparseQR_C_free ====================================================
// =============================================================================

int SuiteSparseQR_C_free
(
    // input/output:
    SuiteSparseQR_C_factorization **QR_handle,
    cholmod_common *cc
)
{
    RETURN_IF_NULL_COMMON (FALSE) ;

    SuiteSparseQR_C_factorization *QR ;
    if (QR_handle == NULL || *QR_handle == NULL) return (TRUE) ;
    QR = *QR_handle ;
    if (QR->xtype == CHOLMOD_REAL)
    {
        if (QR->itype == CHOLMOD_INT)
        {
            SuiteSparseQR_factorization <double, int32_t> *QR2 ;
            QR2 = (SuiteSparseQR_factorization <double, int32_t> *) (QR->factors) ;
            spqr_freefac (&QR2, cc) ;
        }
        else
        {
            SuiteSparseQR_factorization <double, SuiteSparse_long> *QR2 ;
            QR2 = (SuiteSparseQR_factorization <double, SuiteSparse_long> *) (QR->factors) ;
            spqr_freefac (&QR2, cc) ;
        }
    }
    else
    {
        if (QR->itype == CHOLMOD_INT)
        {
            SuiteSparseQR_factorization <Complex, int32_t> *QR2 ;
            QR2 = (SuiteSparseQR_factorization <Complex, int32_t> *) (QR->factors) ;
            spqr_freefac (&QR2, cc) ;
        }
        else
        {
            SuiteSparseQR_factorization <Complex, SuiteSparse_long> *QR2 ;
            QR2 = (SuiteSparseQR_factorization <Complex, SuiteSparse_long> *) (QR->factors) ;
            spqr_freefac (&QR2, cc) ;
        }
    }
    if (QR->itype == CHOLMOD_INT)
    {
        spqr_free <int32_t> (1, sizeof (SuiteSparseQR_C_factorization), QR, cc) ;
    }
    else
    {
        spqr_free <SuiteSparse_long> (1, sizeof (SuiteSparseQR_C_factorization), QR, cc) ;
    }
    *QR_handle = NULL ;
    return (TRUE) ;
}

// =============================================================================
// === SuiteSparseQR_C_solve ===================================================
// =============================================================================

// Solve an upper or lower triangular system using R from the QR factorization
//
// system=SPQR_RX_EQUALS_B    (0): X = R\B         B is m-by-k and X is n-by-k
// system=SPQR_RETX_EQUALS_B  (1): X = E*(R\B)     as above, E is a permutation
// system=SPQR_RTX_EQUALS_B   (2): X = R'\B        B is n-by-k and X is m-by-k
// system=SPQR_RTX_EQUALS_ETB (3): X = R'\(E'*B)   as above, E is a permutation

cholmod_dense* SuiteSparseQR_C_solve    // returnx X, or NULL if failure
(
    // inputs:
    int system,                 // which system to solve
    SuiteSparseQR_C_factorization *QR,  // of an m-by-n sparse matrix A
    cholmod_dense *B,           // right-hand-side, m-by-k or n-by-k
    cholmod_common *cc          // workspace and parameters
)
{
    RETURN_IF_NULL (QR, NULL) ;
    if (QR->xtype == CHOLMOD_REAL)
    {
        if (QR->itype == CHOLMOD_INT)
        {
            return SuiteSparseQR_solve (system,
            (SuiteSparseQR_factorization <double, int32_t>  *) QR->factors, B, cc) ;
        }
        else
        {
            return SuiteSparseQR_solve (system,
            (SuiteSparseQR_factorization <double, SuiteSparse_long>  *) QR->factors, B, cc) ;
        }
    }
    else
    {
        if (QR->itype == CHOLMOD_INT)
        {
            return SuiteSparseQR_solve (system,
            (SuiteSparseQR_factorization <Complex, int32_t>  *) QR->factors, B, cc) ;
        }
        else
        {
            return SuiteSparseQR_solve (system,
            (SuiteSparseQR_factorization <Complex, SuiteSparse_long>  *) QR->factors, B, cc) ;
        }
    }
}

// =============================================================================
// === SuiteSparseQR_C_qmult ===================================================
// =============================================================================

// Applies Q in Householder form (as stored in the QR factorization object
// returned by SuiteSparseQR_C_factorize) to a dense matrix X.
//
//  method SPQR_QTX (0): Y = Q'*X
//  method SPQR_QX  (1): Y = Q*X
//  method SPQR_XQT (2): Y = X*Q'
//  method SPQR_XQ  (3): Y = X*Q

// returns Y of size m-by-n, or NULL on failure
cholmod_dense *SuiteSparseQR_C_qmult
(
    // inputs:
    int method,                 // 0,1,2,3
    SuiteSparseQR_C_factorization *QR,  // of an m-by-n sparse matrix A
    cholmod_dense *X,           // size m-by-n with leading dimension ldx
    cholmod_common *cc          // workspace and parameters
)
{
    RETURN_IF_NULL (QR, NULL) ;
    if (QR->xtype == CHOLMOD_REAL)
    {
        if (QR->itype == CHOLMOD_INT)
        {
            return SuiteSparseQR_qmult (method,
            (SuiteSparseQR_factorization <double, int32_t>  *) QR->factors, X, cc) ;
        }
        else
        {
            return SuiteSparseQR_qmult (method,
            (SuiteSparseQR_factorization <double, SuiteSparse_long>  *) QR->factors, X, cc) ;
        }
    }
    else
    {
        if (QR->itype == CHOLMOD_INT)
        {
            return SuiteSparseQR_qmult (method,
            (SuiteSparseQR_factorization <Complex, int32_t>  *) QR->factors, X, cc) ;
        }
        else
        {
            return SuiteSparseQR_qmult (method,
            (SuiteSparseQR_factorization <Complex, SuiteSparse_long>  *) QR->factors, X, cc) ;
        }
    }
}

#endif

// =============================================================================
}
