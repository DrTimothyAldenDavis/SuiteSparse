// =============================================================================
// === spqr_freefac ============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Frees the contents of the QR factor object.

#include "spqr.hpp"

template <typename Entry> void spqr_freefac
(
    SuiteSparseQR_factorization <Entry> **QR_handle,

    // workspace and parameters
    cholmod_common *cc
)

{
    SuiteSparseQR_factorization <Entry> *QR ;
    int64_t n, m, bncols, n1rows, r1nz ;

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

    cholmod_l_free (n+bncols, sizeof (int64_t),  QR->Q1fill,  cc) ; 
    cholmod_l_free (m,        sizeof (int64_t),  QR->P1inv,   cc) ;
    cholmod_l_free (m,        sizeof (int64_t),  QR->HP1inv,  cc) ;
    cholmod_l_free (n1rows+1, sizeof (int64_t),  QR->R1p,     cc) ;
    cholmod_l_free (r1nz,     sizeof (int64_t),  QR->R1j,     cc) ;
    cholmod_l_free (r1nz,     sizeof (Entry), QR->R1x,     cc) ;
    cholmod_l_free (n,        sizeof (int64_t),  QR->Rmap,    cc) ;
    cholmod_l_free (n,        sizeof (int64_t),  QR->RmapInv, cc) ;

    cholmod_l_free (1, sizeof (SuiteSparseQR_factorization <Entry>), QR, cc) ;
    *QR_handle = NULL ;
}

// =============================================================================

template void spqr_freefac <double>
(
    SuiteSparseQR_factorization <double> **QR_handle,

    // workspace and parameters
    cholmod_common *cc
) ;

// =============================================================================

template void spqr_freefac <Complex>
(
    SuiteSparseQR_factorization <Complex> **QR_handle,

    // workspace and parameters
    cholmod_common *cc
) ;
