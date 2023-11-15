//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_add_worker: C = alpha*A + beta*B
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

//------------------------------------------------------------------------------
// cholmod_add: C = alpha*A + beta*B
//------------------------------------------------------------------------------

static void TEMPLATE (cholmod_add_worker)
(
    cholmod_sparse *C,
    cholmod_sparse *A,
    cholmod_sparse *B,
    double alpha [2],
    double beta [2],
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    ASSERT (A->sorted) ;
    ASSERT (B->sorted) ;
    ASSERT (C->sorted) ;
    ASSERT (A->stype == B->stype) ;
    ASSERT (A->stype == C->stype) ;
    ASSERT (C->packed) ;

    bool upper = (A->stype > 0) ;
    bool lower = (A->stype < 0) ;

    Int  *Ap  = (Int *) A->p ;
    Int  *Anz = (Int *) A->nz ;
    Int  *Ai  = (Int *) A->i ;
    bool apacked = A->packed ;
    size_t ncol = A->ncol ;

    Int  *Bp  = (Int *) B->p ;
    Int  *Bnz = (Int *) B->nz ;
    Int  *Bi  = (Int *) B->i ;
    bool bpacked = B->packed ;

    Int *Cp = (Int *) C->p ;
    Int *Ci = (Int *) C->i ;

    #ifndef PATTERN
        Real *Ax = (Real *) A->x ;
        Real *Bx = (Real *) B->x ;
        Real *Cx = (Real *) C->x ;
        Real alphax [2], betax [2] ;
        alphax [0] = (Real) alpha [0] ;
        betax  [0] = (Real) beta  [0] ;
        alphax [1] = 0 ;
        betax  [1] = 0 ;
    #endif

    #ifdef COMPLEX
        alphax [1] = (Real) alpha [1] ;
        betax  [1] = (Real) beta  [1] ;
    #endif

    #ifdef ZOMPLEX
        Real *Az = (Real *) A->z ;
        Real *Bz = (Real *) B->z ;
        Real *Cz = (Real *) C->z ;
        Real alphaz [1], betaz [1] ;
        alphaz [0] = (Real) alpha [1] ;
        betaz  [0] = (Real) beta  [1] ;
    #endif

    //--------------------------------------------------------------------------
    // C = alpha*A + beta*B
    //--------------------------------------------------------------------------

    Int cnz = 0 ;

    for (Int j = 0 ; j < ncol ; j++)
    {

        //----------------------------------------------------------------------
        // log the start of the jth column of C
        //----------------------------------------------------------------------

        Cp [j] = cnz ;

        //----------------------------------------------------------------------
        // get A(:,j) and B(:,j)
        //----------------------------------------------------------------------

        Int pa = Ap [j] ;
        Int paend = (apacked) ? (Ap [j+1]) : (pa + Anz [j]) ;

        Int pb = Bp [j] ;
        Int pbend = (bpacked) ? (Bp [j+1]) : (pb + Bnz [j]) ;

        //----------------------------------------------------------------------
        // C(:,j) = alpha*A(:,j) + beta*B(:,j)
        //----------------------------------------------------------------------

        while (pa < paend || pb < pbend)
        {
            // get A(i,j)
            Int iA = (pa < paend) ? Ai [pa] : Int_max ;

            // get B(i,j)
            Int iB = (pb < pbend) ? Bi [pb] : Int_max ;

            // skip if C(i,j) is in the ignored part of the matrix
            Int i = MIN (iA, iB) ;
            if ((upper && i > j) || (lower && i < j)) continue ;

            // compute C(i,j)
            ASSERT (cnz < C->nzmax) ;
            Ci [cnz] = i ;
            if (iA < iB)
            {
                // B(:,j) is not present, so C(i,j) = alpha*A(i,j)
                MULT (Cx, Cz, cnz, alphax, alphaz, 0, Ax, Az, pa) ;
                pa++ ;
            }
            else if (iA > iB)
            {
                // A(:,j) is not present, so C(i,j) = beta*B(i,j)
                MULT (Cx, Cz, cnz, betax, betaz, 0, Bx, Bz, pb) ;
                pb++ ;
            }
            else
            {
                // C(i,j) = alpha*A(i,j) + beta*B(i,j)
                MULT    (Cx, Cz, cnz, alphax, alphaz, 0, Ax, Az, pa) ;
                MULTADD (Cx, Cz, cnz, betax,  betaz,  0, Bx, Bz, pb) ;
                pa++ ;
                pb++ ;
            }
            cnz++ ;
        }
    }

    //--------------------------------------------------------------------------
    // log the end of the last column of C
    //--------------------------------------------------------------------------

    Cp [ncol] = cnz ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

