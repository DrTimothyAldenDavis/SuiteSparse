// =============================================================================
// === spqr_freesym ============================================================
// =============================================================================

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

    cholmod_l_free (n,      sizeof (Int), QRsym->Qfill, cc) ;
    cholmod_l_free (nf+1,   sizeof (Int), QRsym->Super, cc) ;
    cholmod_l_free (nf+1,   sizeof (Int), QRsym->Rp, cc) ;
    cholmod_l_free (rjsize, sizeof (Int), QRsym->Rj, cc) ;
    cholmod_l_free (nf+1,   sizeof (Int), QRsym->Parent, cc) ;
    cholmod_l_free (nf+2,   sizeof (Int), QRsym->Childp, cc) ;
    cholmod_l_free (nf+1,   sizeof (Int), QRsym->Child, cc) ;
    cholmod_l_free (nf+1,   sizeof (Int), QRsym->Post, cc) ;
    cholmod_l_free (m,      sizeof (Int), QRsym->PLinv, cc) ;
    cholmod_l_free (n+2,    sizeof (Int), QRsym->Sleft, cc) ;
    cholmod_l_free (m+1,    sizeof (Int), QRsym->Sp, cc) ;
    cholmod_l_free (anz,    sizeof (Int), QRsym->Sj, cc) ;

    cholmod_l_free (nf+1,   sizeof (Int), QRsym->Hip, cc) ;

    // parallel analysis
    ntasks = QRsym->ntasks ;
    cholmod_l_free (ntasks+2, sizeof (Int), QRsym->TaskChildp, cc) ;
    cholmod_l_free (ntasks+1, sizeof (Int), QRsym->TaskChild, cc) ;
    cholmod_l_free (nf+1,     sizeof (Int), QRsym->TaskFront, cc) ;
    cholmod_l_free (ntasks+2, sizeof (Int), QRsym->TaskFrontp, cc) ;
    cholmod_l_free (ntasks+1, sizeof (Int), QRsym->TaskStack, cc) ;
    cholmod_l_free (nf+1,     sizeof (Int), QRsym->On_stack, cc) ;

    ns = QRsym->ns ;
    cholmod_l_free (ns+2,     sizeof (Int), QRsym->Stack_maxstack, cc) ;

    cholmod_l_free (1, sizeof (spqr_symbolic), QRsym, cc) ;

    *QRsym_handle = NULL ;
}
