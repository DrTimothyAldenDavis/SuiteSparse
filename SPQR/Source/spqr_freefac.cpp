// =============================================================================
// === spqr_freefac ============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "spqr.hpp"

// Frees the contents of the QR factor object.

template <typename Entry, typename Int> void spqr_freefac
(
    SuiteSparseQR_factorization <Entry, Int> **QR_handle,

    // workspace and parameters
    cholmod_common *cc
)
{
    SuiteSparseQR_factorization <Entry, Int> *QR ;
    Int n, m, bncols, n1rows, r1nz ;

    if (QR_handle == NULL || *QR_handle == NULL)
    {
        // nothing to do; caller probably ran out of memory
        return ;
    }
    QR = *QR_handle ;

    n      = QR->nacols ;
    m      = QR->narows ;
    bncols = QR->bncols ;
    n1rows = QR->n1rows ;
    r1nz   = QR->r1nz ;

    spqr_freenum (& (QR->QRnum), cc) ;
    spqr_freesym (& (QR->QRsym), cc) ;

    spqr_free <Int> (n+bncols, sizeof (Int),  QR->Q1fill,  cc) ; 
    spqr_free <Int> (m,        sizeof (Int),  QR->P1inv,   cc) ;
    spqr_free <Int> (m,        sizeof (Int),  QR->HP1inv,  cc) ;
    spqr_free <Int> (n1rows+1, sizeof (Int),  QR->R1p,     cc) ;
    spqr_free <Int> (r1nz,     sizeof (Int),  QR->R1j,     cc) ;
    spqr_free <Int> (r1nz,     sizeof (Entry), QR->R1x,     cc) ;
    spqr_free <Int> (n,        sizeof (Int),  QR->Rmap,    cc) ;
    spqr_free <Int> (n,        sizeof (Int),  QR->RmapInv, cc) ;

    spqr_free <Int> (1, sizeof (SuiteSparseQR_factorization <Entry, Int>), QR, cc) ;
    *QR_handle = NULL ;
}

// =============================================================================
template void spqr_freefac <double, int32_t>
(
    SuiteSparseQR_factorization <double, int32_t> **QR_handle,

    // workspace and parameters
    cholmod_common *cc
) ;
template void spqr_freefac <Complex, int32_t>
(
    SuiteSparseQR_factorization <Complex, int32_t> **QR_handle,

    // workspace and parameters
    cholmod_common *cc
) ;
template void spqr_freefac <double, int64_t>
(
    SuiteSparseQR_factorization <double, int64_t> **QR_handle,

    // workspace and parameters
    cholmod_common *cc
) ;
template void spqr_freefac <Complex, int64_t>
(
    SuiteSparseQR_factorization <Complex, int64_t> **QR_handle,

    // workspace and parameters
    cholmod_common *cc
) ;
