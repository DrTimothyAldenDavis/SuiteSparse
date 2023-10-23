//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_aat_worker: C = A*A' or A(:,f)*A*(:,f)'
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static void TEMPLATE (cholmod_aat_worker)
(
    cholmod_sparse *C,  // output matrix
    cholmod_sparse *A,  // input matrix, not modified
    cholmod_sparse *F,  // input matrix, not modified, F = A' or A(:,f)'
    bool ignore_diag,   // if true, ignore diagonal
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A->stype == 0) ;
    ASSERT (C->stype == 0) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int n = A->nrow ;
    Int  *Ap  = (Int  *) A->p ;
    Int  *Anz = (Int  *) A->nz ;
    Int  *Ai  = (Int  *) A->i ;
    Real *Ax  = (Real *) A->x ;
    Real *Az  = (Real *) A->z ;
    bool packed = A->packed ;

    Int  *Fp  = (Int  *) F->p ;
    Int  *Fi  = (Int  *) F->i ;
    Real *Fx  = (Real *) F->x ;
    Real *Fz  = (Real *) F->z ;
    ASSERT (F->packed) ;

    Int  *Cp = (Int  *) C->p ;
    Int  *Ci = (Int  *) C->i ;
    Real *Cx = (Real *) C->x ;
    Real *Cz = (Real *) C->z ;
    ASSERT (C->packed) ;

    //--------------------------------------------------------------------------
    // get workspace
    //--------------------------------------------------------------------------

    // W is all negative, because jmark in t_cholmod_aat is always negative.

    Int *W = Common->Iwork ;    // size n, all < 0
    #ifndef NDEBUG
    for (Int i = 0 ; i < n ; i++)
    {
        ASSERT (W [i] < 0) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // C = A*A' or A(:,f)*A(:,f)'
    //--------------------------------------------------------------------------

    Int pc = 0 ;

    for (Int j = 0 ; j < n ; j++)
    {

        //----------------------------------------------------------------------
        // log the start of C(:,j)
        //----------------------------------------------------------------------

        Int pc_start = pc ;
        Cp [j] = pc ;

        //----------------------------------------------------------------------
        // compute C(:,j) = A*F(:,j)
        //----------------------------------------------------------------------

        for (Int pf = Fp [j] ; pf < Fp [j+1] ; pf++)
        {

            //------------------------------------------------------------------
            // get the F(t,j) entry
            //------------------------------------------------------------------

            Int t = Fi [pf] ;
            Real fx [2] = {0,0} ;
            Real fz [1] = {0} ;
            ASSIGN (fx, fz, 0, Fx, Fz, pf) ;

            //------------------------------------------------------------------
            // C(:,j) += A(:,t)*F(t,j)
            //------------------------------------------------------------------

            Int p = Ap [t] ;
            Int pend = (packed) ? (Ap [t+1]) : (p + Anz [t]) ;
            for ( ; p < pend ; p++)
            {
                // get the A(i,t) entry
                Int i = Ai [p] ;
                if (ignore_diag && i == j) continue ;
                Int pi = W [i] ;
                if (pi < pc_start)
                {
                    // C(i,j) is a new entry; log its position
                    Ci [pc] = i ;
                    W [i] = pc ;
                    // C(i,j) = A(i,t) * F(t,j)
                    MULT (Cx, Cz, pc, Ax, Az, p, fx, fz, 0) ;
                    pc++ ;
                }
                else
                {
                    // C(i,j) exists in C(:,j) at position pi
                    // C(i,j) += A(i,t) * F(t,j)
                    MULTADD (Cx, Cz, pi, Ax, Az, p, fx, fz, 0) ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // log the end of the last column of C
    //--------------------------------------------------------------------------

    Cp [n] = pc ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

