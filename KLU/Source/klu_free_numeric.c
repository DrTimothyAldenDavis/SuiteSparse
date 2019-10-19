/* ========================================================================== */
/* === klu_free_numeric ===================================================== */
/* ========================================================================== */

/* Free the KLU Numeric object. */

#include "klu_internal.h"

int KLU_free_numeric
(
    klu_numeric **NumericHandle,
    klu_common	*Common
)
{
    klu_numeric *Numeric ;
    int **Lbip, **Ubip, **Lblen, **Ublen ;
    Unit **LUbx, **Udiag ;
    int block ;

    if (Common == NULL)
    {
	return (FALSE) ;
    }
    if (NumericHandle == NULL)
    {
	return (TRUE) ;
    }
    Numeric = *NumericHandle ;
    if (Numeric == NULL)
    {
	return (TRUE) ;
    }

    klu_free (Numeric->Pnum, Common) ;
    klu_free (Numeric->Offp, Common) ;
    klu_free (Numeric->Offi, Common) ;
    klu_free (Numeric->Offx, Common) ;
    klu_free (Numeric->Singleton, Common) ;
    klu_free (Numeric->Rs, Common) ;
    klu_free (Numeric->Pinv, Common) ;
    klu_free (Numeric->Work, Common) ;

    Lbip  = Numeric->Lbip ;
    Lblen = Numeric->Lblen ;
    Ubip  = Numeric->Ubip ;
    Ublen = Numeric->Ublen ;
    LUbx  = (Unit **) Numeric->LUbx ;
    Udiag = (Unit **) Numeric->Udiag ;
    for (block = 0 ; block < Numeric->nblocks ; block++)
    {
	if (Lbip  != (int **) NULL)  klu_free (Lbip  [block], Common) ;
	if (Lblen != (int **) NULL)  klu_free (Lblen [block], Common) ;
	if (Ubip  != (int **) NULL)  klu_free (Ubip  [block], Common) ;
	if (Ublen != (int **) NULL)  klu_free (Ublen [block], Common) ;
	if (LUbx  != (Unit **) NULL) klu_free (LUbx  [block], Common) ;
	if (Udiag != (Unit **) NULL) klu_free (Udiag [block], Common) ;
    }

    klu_free (Numeric->Lbip,  Common) ;
    klu_free (Numeric->Lblen, Common) ;
    klu_free (Numeric->Ubip,  Common) ;
    klu_free (Numeric->Ublen, Common) ;
    klu_free (Numeric->LUbx,  Common) ;
    klu_free (Numeric->Udiag, Common) ;

    klu_free (Numeric, Common) ;

    *NumericHandle = NULL ;
    return (TRUE) ;
}
