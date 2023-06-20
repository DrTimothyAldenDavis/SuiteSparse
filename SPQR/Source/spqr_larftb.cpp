// =============================================================================
// === spqr_larftb =============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Apply a set of Householder reflections to a matrix.  Given the vectors
// V and coefficients Tau, construct the matrix T and then apply the updates.
// In MATLAB (1-based indexing), this function computes the following:

/*
    function C = larftb (C, V, Tau, method)
    [v k] = size (V) ;
    [m n] = size (C) ;
    % construct T for the compact WY representation
    V = tril (V,-1) + eye (v,k) ;
    T = zeros (k,k) ;
    T (1,1) = Tau (1) ;
    for j = 2:k
        tau = Tau (j) ;
        z = -tau * V (:, 1:j-1)' * V (:,j) ;
        T (1:j-1,j) = T (1:j-1,1:j-1) * z ;
        T (j,j) = tau ;
    end
    % apply the updates
    if (method == 0)
        C = C - V * T' * V' * C ;       % method 0: Left, Transpose
    elseif (method == 1)
        C = C - V * T * V' * C ;        % method 1: Left, No Transpose
    elseif (method == 2)
        C = C - C * V * T' * V' ;       % method 2: Right, Transpose
    elseif (method == 3)
        C = C - C * V * T * V' ;        % method 3: Right, No Transpose
    end
*/

#include "spqr.hpp"

template <typename Int> inline void spqr_private_larft (char direct, char storev, Int n, Int k,
    double *V, Int ldv, double *Tau, double *T, Int ldt,
    cholmod_common *cc)
{
    SUITESPARSE_LAPACK_dlarft (&direct, &storev, n, k, V, ldv, Tau, T, ldt,
        cc->blas_ok) ;
}

template <typename Int> inline void spqr_private_larft (char direct, char storev, Int n, Int k,
    Complex *V, Int ldv, Complex *Tau, Complex *T, Int ldt,
    cholmod_common *cc)
{
    SUITESPARSE_LAPACK_zlarft (&direct, &storev, n, k, V, ldv, Tau, T, ldt,
        cc->blas_ok) ;
}


template <typename Int> inline void spqr_private_larfb (char side, char trans, char direct, char storev,
    Int m, Int n, Int k, double *V, Int ldv, double *T,
    Int ldt, double *C, Int ldc, double *Work, Int ldwork,
    cholmod_common *cc)
{
    SUITESPARSE_LAPACK_dlarfb (&side, &trans, &direct, &storev, m, n, k,
        V, ldv, T, ldt, C, ldc, Work, ldwork, cc->blas_ok) ;
}


template <typename Int> inline void spqr_private_larfb (char side, char trans, char direct, char storev,
    Int m, Int n, Int k, Complex *V, Int ldv, Complex *T,
    Int ldt, Complex *C, Int ldc, Complex *Work, Int ldwork,
    cholmod_common *cc)
{
    char tr = (trans == 'T') ? 'C' : 'N' ;      // change T to C
    SUITESPARSE_LAPACK_zlarfb (&side, &tr, &direct, &storev, m, n, k,
        V, ldv, T, ldt, C, ldc, Work, ldwork, cc->blas_ok) ;
}


// =============================================================================

template <typename Entry, typename Int> void spqr_larftb
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
)
{
    Entry *T, *Work ;

    // -------------------------------------------------------------------------
    // check inputs and split up workspace
    // -------------------------------------------------------------------------

    if (m <= 0 || n <= 0 || k <= 0)
    {
        return ; // nothing to do
    }

    T = W ;             // triangular k-by-k matrix for block reflector
    Work = W + k*k ;    // workspace of size n*k or m*k for larfb

    // -------------------------------------------------------------------------
    // construct and apply the k-by-k upper triangular matrix T
    // -------------------------------------------------------------------------

    // larft and larfb are always used "Forward" and "Columnwise"

    if (method == SPQR_QTX)
    {
        ASSERT (m >= k) ;
        spqr_private_larft ('F', 'C', m, k, V, ldv, Tau, T, k, cc) ;
        // Left, Transpose, Forward, Columwise:
        spqr_private_larfb ('L', 'T', 'F', 'C', m, n, k, V, ldv, T, k, C, ldc,
            Work, n, cc) ;
    }
    else if (method == SPQR_QX)
    {
        ASSERT (m >= k) ;
        spqr_private_larft ('F', 'C', m, k, V, ldv, Tau, T, k, cc) ;
        // Left, No Transpose, Forward, Columwise:
        spqr_private_larfb ('L', 'N', 'F', 'C', m, n, k, V, ldv, T, k, C, ldc,
            Work, n, cc) ;
    }
    else if (method == SPQR_XQT)
    {
        ASSERT (n >= k) ;
        spqr_private_larft ('F', 'C', n, k, V, ldv, Tau, T, k, cc) ;
        // Right, Transpose, Forward, Columwise:
        spqr_private_larfb ('R', 'T', 'F', 'C', m, n, k, V, ldv, T, k, C, ldc,
            Work, m, cc) ;
    }
    else if (method == SPQR_XQ)
    {
        ASSERT (n >= k) ;
        spqr_private_larft ('F', 'C', n, k, V, ldv, Tau, T, k, cc) ;
        // Right, No Transpose, Forward, Columwise:
        spqr_private_larfb ('R', 'N', 'F', 'C', m, n, k, V, ldv, T, k, C, ldc,
            Work, m, cc) ;
    }
}

template void spqr_larftb <double, int32_t>
(
    // inputs, not modified (V is modified and then restored on output)
    int method,     // 0,1,2,3
    int32_t m,         // C is m-by-n
    int32_t n,
    int32_t k,         // V is v-by-k
                    // for methods 0 and 1, v = m,
                    // for methods 2 and 3, v = n
    int32_t ldc,       // leading dimension of C
    int32_t ldv,       // leading dimension of V
    double *V,       // V is v-by-k, unit lower triangular (diag not stored)
    double *Tau,     // size k, the k Householder coefficients

    // input/output
    double *C,       // C is m-by-n, with leading dimension ldc

    // workspace, not defined on input or output
    double *W,       // for methods 0,1: size k*k + n*k
                    // for methods 2,3: size k*k + m*k
    cholmod_common *cc
) ;
template void spqr_larftb <Complex, int32_t>
(
    // inputs, not modified (V is modified and then restored on output)
    int method,     // 0,1,2,3
    int32_t m,         // C is m-by-n
    int32_t n,
    int32_t k,         // V is v-by-k
                    // for methods 0 and 1, v = m,
                    // for methods 2 and 3, v = n
    int32_t ldc,       // leading dimension of C
    int32_t ldv,       // leading dimension of V
    Complex *V,       // V is v-by-k, unit lower triangular (diag not stored)
    Complex *Tau,     // size k, the k Householder coefficients

    // input/output
    Complex *C,       // C is m-by-n, with leading dimension ldc

    // workspace, not defined on input or output
    Complex *W,       // for methods 0,1: size k*k + n*k
                    // for methods 2,3: size k*k + m*k
    cholmod_common *cc
) ;
template void spqr_larftb <double, int64_t>
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
    double *V,       // V is v-by-k, unit lower triangular (diag not stored)
    double *Tau,     // size k, the k Householder coefficients

    // input/output
    double *C,       // C is m-by-n, with leading dimension ldc

    // workspace, not defined on input or output
    double *W,       // for methods 0,1: size k*k + n*k
                    // for methods 2,3: size k*k + m*k
    cholmod_common *cc
) ;
template void spqr_larftb <Complex, int64_t>
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
    Complex *V,       // V is v-by-k, unit lower triangular (diag not stored)
    Complex *Tau,     // size k, the k Householder coefficients

    // input/output
    Complex *C,       // C is m-by-n, with leading dimension ldc

    // workspace, not defined on input or output
    Complex *W,       // for methods 0,1: size k*k + n*k
                    // for methods 2,3: size k*k + m*k
    cholmod_common *cc
) ;
