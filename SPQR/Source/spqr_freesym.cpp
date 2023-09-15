// =============================================================================
// === spqr_freesym ============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Frees the contents of the QR Symbolic object

#include "spqr.hpp"

template <typename Int> void spqr_freesym
(
    spqr_symbolic <Int> **QRsym_handle,

    // workspace and parameters
    cholmod_common *cc
)
{
    spqr_symbolic <Int> *QRsym ;
    spqr_gpu_impl <Int> *QRgpu ;
    Int m, n, anz, nf, rjsize, ns, ntasks ;

    if (QRsym_handle == NULL || *QRsym_handle == NULL)
    {
        // nothing to do; caller probably ran out of memory
        return ;
    }
    QRsym = *QRsym_handle  ;

    m = QRsym->m ;
    n = QRsym->n ;
    nf = QRsym->nf ;
    anz = QRsym->anz ;
    rjsize = QRsym->rjsize ;

    spqr_free <Int> (n,      sizeof (Int), QRsym->Qfill, cc) ;
    spqr_free <Int> (nf+1,   sizeof (Int), QRsym->Super, cc) ;
    spqr_free <Int> (nf+1,   sizeof (Int), QRsym->Rp, cc) ;
    spqr_free <Int> (rjsize, sizeof (Int), QRsym->Rj, cc) ;
    spqr_free <Int> (nf+1,   sizeof (Int), QRsym->Parent, cc) ;
    spqr_free <Int> (nf+2,   sizeof (Int), QRsym->Childp, cc) ;
    spqr_free <Int> (nf+1,   sizeof (Int), QRsym->Child, cc) ;
    spqr_free <Int> (nf+1,   sizeof (Int), QRsym->Post, cc) ;
    spqr_free <Int> (m,      sizeof (Int), QRsym->PLinv, cc) ;
    spqr_free <Int> (n+2,    sizeof (Int), QRsym->Sleft, cc) ;
    spqr_free <Int> (m+1,    sizeof (Int), QRsym->Sp, cc) ;
    spqr_free <Int> (anz,    sizeof (Int), QRsym->Sj, cc) ;

    spqr_free <Int> (nf+1,   sizeof (Int), QRsym->Hip, cc) ;

    spqr_free <Int> (nf+1,   sizeof (Int), QRsym->Fm, cc) ;
    spqr_free <Int> (nf+1,   sizeof (Int), QRsym->Cm, cc) ;

    spqr_free <Int> (n,      sizeof (Int), QRsym->ColCount, cc) ;

    // gpu metadata
    QRgpu = QRsym->QRgpu;
    if(QRgpu)
    {
        spqr_free <Int> (nf,   sizeof (Int),   QRgpu->RimapOffsets, cc) ;
        spqr_free <Int> (nf,   sizeof (Int),   QRgpu->RjmapOffsets, cc) ;
        spqr_free <Int> (nf+2, sizeof (Int),   QRgpu->Stagingp, cc) ;
        spqr_free <Int> (nf,   sizeof (Int),   QRgpu->StageMap, cc) ;
        spqr_free <Int> (nf+1, sizeof (size_t), QRgpu->FSize, cc) ;
        spqr_free <Int> (nf+1, sizeof (size_t), QRgpu->RSize, cc) ;
        spqr_free <Int> (nf+1, sizeof (size_t), QRgpu->SSize, cc) ;
        spqr_free <Int> (nf,   sizeof (Int),   QRgpu->FOffsets, cc) ;
        spqr_free <Int> (nf,   sizeof (Int),   QRgpu->ROffsets, cc) ;
        spqr_free <Int> (nf,   sizeof (Int),   QRgpu->SOffsets, cc) ;

        spqr_free <Int> (1, sizeof (spqr_gpu), QRgpu, cc) ;
    }

    // parallel analysis
    ntasks = QRsym->ntasks ;
    spqr_free <Int> (ntasks+2, sizeof (Int), QRsym->TaskChildp, cc) ;
    spqr_free <Int> (ntasks+1, sizeof (Int), QRsym->TaskChild, cc) ;
    spqr_free <Int> (nf+1,     sizeof (Int), QRsym->TaskFront, cc) ;
    spqr_free <Int> (ntasks+2, sizeof (Int), QRsym->TaskFrontp, cc) ;
    spqr_free <Int> (ntasks+1, sizeof (Int), QRsym->TaskStack, cc) ;
    spqr_free <Int> (nf+1,     sizeof (Int), QRsym->On_stack, cc) ;

    ns = QRsym->ns ;
    spqr_free <Int> (ns+2,     sizeof (Int), QRsym->Stack_maxstack, cc) ;

    spqr_free <Int> (1, sizeof (spqr_symbolic <Int>), QRsym, cc) ;

    *QRsym_handle = NULL ;
}

template void spqr_freesym <int32_t>
(
    spqr_symbolic <int32_t> **QRsym_handle,

    // workspace and parameters
    cholmod_common *cc
) ;
template void spqr_freesym <int64_t>
(
    spqr_symbolic <int64_t> **QRsym_handle,

    // workspace and parameters
    cholmod_common *cc
) ;
