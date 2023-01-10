//------------------------------------------------------------------------------
// UMFPACK/Source/umfpack_symbolic: symbolic analysis
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    User-callable.  Performs a symbolic factorization.
    See umfpack.h for details.
*/

#include "umf_internal.h"

int UMFPACK_symbolic
(
    Int n_row,
    Int n_col,
    const Int Ap [ ],
    const Int Ai [ ],
    const double Ax [ ],
#ifdef COMPLEX
    const double Az [ ],
#endif
    void **SymbolicHandle,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
)
{
    return (UMFPACK_qsymbolic (n_row, n_col, Ap, Ai, Ax,
#ifdef COMPLEX
        Az,
#endif
        (Int *) NULL, SymbolicHandle, Control, Info)) ;
}
