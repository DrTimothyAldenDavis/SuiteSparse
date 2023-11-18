//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_transpose: transpose a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

cholmod_sparse *CHOLMOD(transpose)
(
    // input:
    cholmod_sparse *A,  // input matrix
    int mode,           // 2: numerical (conj)
                        // 1: numerical (non-conj.)
                        // <= 0: pattern (with diag)
    cholmod_common *Common
)
{
    return (CHOLMOD(ptranspose) (A, mode, /* Perm: */ NULL,
        /* fset: */ NULL, /* fsize: */ 0, Common)) ;
}

