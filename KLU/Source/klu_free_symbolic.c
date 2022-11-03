//------------------------------------------------------------------------------
// KLU/Source/klu_free_symbolic: free the KLU symbolic analysis
//------------------------------------------------------------------------------

// KLU, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
// Authors: Timothy A. Davis and Ekanathan Palamadai.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

/* Free the KLU Symbolic object. */

#include "klu_internal.h"

int KLU_free_symbolic
(
    KLU_symbolic **SymbolicHandle,
    KLU_common   *Common
)
{
    KLU_symbolic *Symbolic ;
    Int n ;
    if (Common == NULL)
    {
        return (FALSE) ;
    }
    if (SymbolicHandle == NULL || *SymbolicHandle == NULL)
    {
        return (TRUE) ;
    }
    Symbolic = *SymbolicHandle ;
    n = Symbolic->n ;
    KLU_free (Symbolic->P, n, sizeof (Int), Common) ;
    KLU_free (Symbolic->Q, n, sizeof (Int), Common) ;
    KLU_free (Symbolic->R, n+1, sizeof (Int), Common) ;
    KLU_free (Symbolic->Lnz, n, sizeof (double), Common) ;
    KLU_free (Symbolic, 1, sizeof (KLU_symbolic), Common) ;
    *SymbolicHandle = NULL ;
    return (TRUE) ;
}
