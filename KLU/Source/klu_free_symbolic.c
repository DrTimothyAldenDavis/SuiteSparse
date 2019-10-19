/* ========================================================================== */
/* === klu_free_symbolic ==================================================== */
/* ========================================================================== */

/* Free the KLU Symbolic object. */

#include "klu_internal.h"

int klu_free_symbolic
(
    klu_symbolic **SymbolicHandle,
    klu_common	 *Common
)
{
    klu_symbolic *Symbolic ;
    if (Common == NULL)
    {
	return (FALSE) ;
    }
    if (SymbolicHandle == NULL)
    {
	return (TRUE) ;
    }
    Symbolic = *SymbolicHandle ;
    klu_free (Symbolic->P, Common) ;
    klu_free (Symbolic->Q, Common) ;
    klu_free (Symbolic->R, Common) ;
    klu_free (Symbolic->Lnz, Common) ;
    klu_free (Symbolic, Common) ;
    *SymbolicHandle = NULL ;
    return (TRUE) ;
}
