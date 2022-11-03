// =============================================================================
// === spqr_freesym ============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Frees the contents of the QR Symbolic object

#include "spqr.hpp"

void spqr_freesym
(
    spqr_symbolic **QRsym_handle,

    // workspace and parameters
    cholmod_common *cc
)
{
    spqr_symbolic *QRsym ;
    spqr_gpu *QRgpu ;
    int64_t m, n, anz, nf, rjsize, ns, ntasks ;

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

    cholmod_l_free (n,      sizeof (int64_t), QRsym->Qfill, cc) ;
    cholmod_l_free (nf+1,   sizeof (int64_t), QRsym->Super, cc) ;
    cholmod_l_free (nf+1,   sizeof (int64_t), QRsym->Rp, cc) ;
    cholmod_l_free (rjsize, sizeof (int64_t), QRsym->Rj, cc) ;
    cholmod_l_free (nf+1,   sizeof (int64_t), QRsym->Parent, cc) ;
    cholmod_l_free (nf+2,   sizeof (int64_t), QRsym->Childp, cc) ;
    cholmod_l_free (nf+1,   sizeof (int64_t), QRsym->Child, cc) ;
    cholmod_l_free (nf+1,   sizeof (int64_t), QRsym->Post, cc) ;
    cholmod_l_free (m,      sizeof (int64_t), QRsym->PLinv, cc) ;
    cholmod_l_free (n+2,    sizeof (int64_t), QRsym->Sleft, cc) ;
    cholmod_l_free (m+1,    sizeof (int64_t), QRsym->Sp, cc) ;
    cholmod_l_free (anz,    sizeof (int64_t), QRsym->Sj, cc) ;

    cholmod_l_free (nf+1,   sizeof (int64_t), QRsym->Hip, cc) ;

    cholmod_l_free (nf+1,   sizeof (int64_t), QRsym->Fm, cc) ;
    cholmod_l_free (nf+1,   sizeof (int64_t), QRsym->Cm, cc) ;

    cholmod_l_free (n,      sizeof (int64_t), QRsym->ColCount, cc) ;

    // gpu metadata
    QRgpu = QRsym->QRgpu;
    if(QRgpu)
    {
        cholmod_l_free (nf,   sizeof (int64_t),   QRgpu->RimapOffsets, cc) ;
        cholmod_l_free (nf,   sizeof (int64_t),   QRgpu->RjmapOffsets, cc) ;
        cholmod_l_free (nf+2, sizeof (int64_t),   QRgpu->Stagingp, cc) ;
        cholmod_l_free (nf,   sizeof (int64_t),   QRgpu->StageMap, cc) ;
        cholmod_l_free (nf+1, sizeof (size_t), QRgpu->FSize, cc) ;
        cholmod_l_free (nf+1, sizeof (size_t), QRgpu->RSize, cc) ;
        cholmod_l_free (nf+1, sizeof (size_t), QRgpu->SSize, cc) ;
        cholmod_l_free (nf,   sizeof (int64_t),   QRgpu->FOffsets, cc) ;
        cholmod_l_free (nf,   sizeof (int64_t),   QRgpu->ROffsets, cc) ;
        cholmod_l_free (nf,   sizeof (int64_t),   QRgpu->SOffsets, cc) ;

        cholmod_l_free (1, sizeof (spqr_gpu), QRgpu, cc) ;
    }

    // parallel analysis
    ntasks = QRsym->ntasks ;
    cholmod_l_free (ntasks+2, sizeof (int64_t), QRsym->TaskChildp, cc) ;
    cholmod_l_free (ntasks+1, sizeof (int64_t), QRsym->TaskChild, cc) ;
    cholmod_l_free (nf+1,     sizeof (int64_t), QRsym->TaskFront, cc) ;
    cholmod_l_free (ntasks+2, sizeof (int64_t), QRsym->TaskFrontp, cc) ;
    cholmod_l_free (ntasks+1, sizeof (int64_t), QRsym->TaskStack, cc) ;
    cholmod_l_free (nf+1,     sizeof (int64_t), QRsym->On_stack, cc) ;

    ns = QRsym->ns ;
    cholmod_l_free (ns+2,     sizeof (int64_t), QRsym->Stack_maxstack, cc) ;

    cholmod_l_free (1, sizeof (spqr_symbolic), QRsym, cc) ;

    *QRsym_handle = NULL ;
}
