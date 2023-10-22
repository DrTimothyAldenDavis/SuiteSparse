//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_transpose_unsym_worker: C = A', A(:,f)', or A(p,f)'
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_transpose_unsym_worker)
(
    cholmod_sparse *A,  // input matrix
    Int *fset,          // a list of column indices in range 0:A->ncol-1
    Int nf,             // # of entries in fset
    cholmod_sparse *C,  // output matrix, must be allocated on input
    Int *Wi             // workspace of size nrow
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int  *Ap  = (Int  *) A->p ;
    Int  *Ai  = (Int  *) A->i ;
    Int  *Anz = (Int  *) A->nz ;
    Real *Ax  = (Real *) A->x ;
    Real *Az  = (Real *) A->z ;
    Int ncol = A->ncol ;

    Int  *Cp  = (Int  *) C->p ;
    Int  *Ci  = (Int  *) C->i ;
    Real *Cx  = (Real *) C->x ;
    Real *Cz  = (Real *) C->z ;

    //--------------------------------------------------------------------------
    // compute the pattern and values of C
    //--------------------------------------------------------------------------

    if (fset != NULL)
    {
        if (A->packed)
        {
            // C = A (p,f)' or A(:,f)' where A is packed
            #define PACKED
            #define FSET
            #include "t_cholmod_transpose_unsym_template.c"
        }
        else
        {
            // C = A (p,f)' or A(:,f)' where A is unpacked
            #define FSET
            #include "t_cholmod_transpose_unsym_template.c"
        }
    }
    else
    {
        if (A->packed)
        {
            // C = A (p,:)' or A' where A is packed
            #define PACKED
            #include "t_cholmod_transpose_unsym_template.c"
        }
        else
        {
            // C = A (p,:)' or A' where A is unpacked
            #include "t_cholmod_transpose_unsym_template.c"
        }
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX
#undef NCONJUGATE

