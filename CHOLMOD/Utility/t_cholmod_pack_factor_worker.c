//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_pack_factor_worker: pack a simplicial factorization
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

//------------------------------------------------------------------------------
// cholmod_pack_factor_worker
//------------------------------------------------------------------------------

static void TEMPLATE (cholmod_pack_factor_worker)
(
    cholmod_factor *L,      // factor to pack
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int n = L->n ;
    Int  *Lp    = (Int  *) L->p ;
    Int  *Li    = (Int  *) L->i ;
    Real *Lx    = (Real *) L->x ;
    Real *Lz    = (Real *) L->z ;
    Int  *Lnz   = (Int  *) L->nz ;
    Int  *Lnext = (Int  *) L->next ;

    Int slack = Common->grow2 ;

    //--------------------------------------------------------------------------
    // pack, traversing the link list of columns of L
    //--------------------------------------------------------------------------

    Int j = Lnext [n+1] ;   // first column in the list is Lnext [n+1]
    Int pnew = 0 ;          // next column can move to pnew 

    while (j != n)          // j=n is the fictious placeholder at end of list
    {

        //----------------------------------------------------------------------
	// get column j, entries currently in Li and Lx [pold...pold+lnzj-1]
        //----------------------------------------------------------------------

	Int pold = Lp [j] ;     // start of column j in L->i and L->j
	Int lnzj = Lnz [j] ;    // # of entries in column j
	ASSERT (lnzj > 0) ;

        //----------------------------------------------------------------------
        // pack column j, if possible
        //----------------------------------------------------------------------

	if (pnew < pold)
	{
            // Li,Lx [pnew...pnew+lnz-1] = Li,Lx [pold...pold+lnz-1]
	    for (Int k = 0 ; k < lnzj ; k++)
	    {
                // move L(i,j) from position pold+k to position pnew+k
		Li [pnew + k] = Li [pold + k] ;
                ASSIGN (Lx, Lz, pnew + k, Lx, Lz, pold + k) ;
            }
            // log the new position of the first entry of L(:,j)
            Lp [j] = pnew ;
        }

        //----------------------------------------------------------------------
        // add some empty space at the end of column j 
        //----------------------------------------------------------------------

        Int desired_space = lnzj + slack ;  // add slack space to column j
        Int max_space = n - j ;             // no need for more than this space
        Int total_space = MIN (desired_space, max_space) ;

        //----------------------------------------------------------------------
        // next column will move to position pnew, if possible
        //----------------------------------------------------------------------

        Int jnext = Lnext [j] ;             // jnext = next column in the list
        Int pnext = Lp [jnext] ;            // next column jnext starts here
        Int pthis = Lp [j] + total_space ;  // one past the end of column j
	pnew = MIN (pthis, pnext) ;         // next column can move to pnew
        j = jnext ;                         // move to the next column
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

