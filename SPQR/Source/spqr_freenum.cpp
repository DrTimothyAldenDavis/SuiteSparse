// =============================================================================
// === spqr_freenum ============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "spqr.hpp"

// Frees the contents of the QR Numeric object

template <typename Entry, typename Int> void spqr_freenum
(
    spqr_numeric <Entry, Int> **QRnum_handle,

    // workspace and parameters
    cholmod_common *cc
)
{
    spqr_numeric <Entry, Int> *QRnum ;
    Int nf, n, m, rjsize, hisize, ns, stack, maxstack ;

    if (QRnum_handle == NULL || *QRnum_handle == NULL)
    {
        // nothing to do; caller probably ran out of memory
        return ;
    }
    QRnum = *QRnum_handle ;

    n  = QRnum->n ;
    m  = QRnum->m ;
    nf = QRnum->nf ;
    rjsize = QRnum->rjsize ;
    hisize = QRnum->hisize ;
    ns = QRnum->ns ;
    maxstack = QRnum->maxstack ;

    spqr_free <Int> (nf, sizeof (Entry *), QRnum->Rblock, cc) ;
    spqr_free <Int> (n,  sizeof (char),    QRnum->Rdead,  cc) ;

    if (QRnum->keepH)
    {
        // QRnum->H* items are present only if H is kept
        spqr_free <Int> (rjsize, sizeof (Int),  QRnum->HStair,  cc) ;
        spqr_free <Int> (rjsize, sizeof (Entry), QRnum->HTau,    cc) ;
        spqr_free <Int> (nf,     sizeof (Int),  QRnum->Hm,      cc) ;
        spqr_free <Int> (nf,     sizeof (Int),  QRnum->Hr,      cc) ;
        spqr_free <Int> (hisize, sizeof (Int),  QRnum->Hii,     cc) ;
        spqr_free <Int> (m,      sizeof (Int),  QRnum->HPinv,   cc) ;
    }

    // free each stack
    if (QRnum->Stacks != NULL)
    {
        Int *Stack_size = QRnum->Stack_size ;
        for (stack = 0 ; stack < ns ; stack++)
        {
            size_t s = Stack_size ? (Stack_size [stack]) : maxstack ;
            spqr_free <Int> (s, sizeof (Entry), QRnum->Stacks [stack], cc) ;
        }
    }
    spqr_free <Int> (ns, sizeof (Entry *), QRnum->Stacks, cc) ;
    spqr_free <Int> (ns, sizeof (Int), QRnum->Stack_size, cc) ;

    spqr_free <Int> (1, sizeof (spqr_numeric<Entry, Int>), QRnum, cc) ;
    *QRnum_handle = NULL ;
}

template void spqr_freenum <double, int32_t>
(
    spqr_numeric <double, int32_t> **QRnum_handle,

    // workspace and parameters
    cholmod_common *cc
) ;
template void spqr_freenum <double, int64_t>
(
    spqr_numeric <double, int64_t> **QRnum_handle,

    // workspace and parameters
    cholmod_common *cc
) ;
template void spqr_freenum <Complex, int32_t>
(
    spqr_numeric <Complex, int32_t> **QRnum_handle,

    // workspace and parameters
    cholmod_common *cc
) ;
template void spqr_freenum <Complex, int64_t>
(
    spqr_numeric <Complex, int64_t> **QRnum_handle,

    // workspace and parameters
    cholmod_common *cc
) ;
// =============================================================================
