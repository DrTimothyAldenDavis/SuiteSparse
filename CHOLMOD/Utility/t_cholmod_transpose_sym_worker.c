//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_transpose_sym_worker: C = A' or A(p,p)'
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_transpose_sym_worker)
(
    cholmod_sparse *C,  // output matrix of size n-by-n
    cholmod_sparse *A,  // input matrix of size n-by-n
    Int *Pinv,          // size n, inverse permutation, or NULL if none
    Int *Wi             // size n workspace; column pointers of C on input
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int n = A->ncol ;

    Int  *Ap  = (Int  *) A->p ;
    Int  *Ai  = (Int  *) A->i ;
    Int  *Anz = (Int  *) A->nz ;
    Real *Ax  = (Real *) A->x ;
    Real *Az  = (Real *) A->z ;

    Int  *Cp  = (Int  *) C->p ;
    Int  *Ci  = (Int  *) C->i ;
    Real *Cx  = (Real *) C->x ;
    Real *Cz  = (Real *) C->z ;

    //--------------------------------------------------------------------------
    // compute pattern and values of C
    //--------------------------------------------------------------------------

    #define NUMERIC
    #include "t_cholmod_transpose_sym_template.c"

}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX
#undef NCONJUGATE

