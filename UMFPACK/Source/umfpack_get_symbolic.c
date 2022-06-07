/* ========================================================================== */
/* === UMFPACK_get_symbolic ================================================= */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Gets the symbolic information held in the Symbolic object.
    See umfpack_get_symbolic.h for a more detailed description.
*/

#include "umf_internal.h"
#include "umf_valid_symbolic.h"

GLOBAL Int UMFPACK_get_symbolic
(
    Int *p_n_row,
    Int *p_n_col,
    Int *p_n1,			/* number of singletons */
    Int *p_nz,
    Int *p_nfr,
    Int *p_nchains,
    Int P [ ],
    Int Q [ ],
 //   Int Diag_map [],
    Int Front_npivcol [ ],
    Int Front_parent [ ],
    Int Front_1strow [ ],
    Int Front_leftmostdesc [ ],
    Int Chain_start [ ],
    Int Chain_maxrows [ ],
    Int Chain_maxcols [ ],
    void *SymbolicHandle
)
{
    SymbolicType *Symbolic ;
    Int k, n_row, n_col, n1, nfr, nchains, *p ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    Symbolic = (SymbolicType *) SymbolicHandle ;
    if (!UMF_valid_symbolic (Symbolic))
    {
	return (UMFPACK_ERROR_invalid_Symbolic_object) ;
    }

    /* ---------------------------------------------------------------------- */
    /* get contents of Symbolic */
    /* ---------------------------------------------------------------------- */

    n_row = Symbolic->n_row ;
    n_col = Symbolic->n_col ;
    n1 = Symbolic->n1 ;
    nfr = Symbolic->nfr ;
    nchains = Symbolic->nchains ;

    if (p_n_row)
    {
	*p_n_row = n_row ;
    }

    if (p_n_col)
    {
	*p_n_col = n_col ;
    }

    if (p_n1)
    {
	*p_n1 = n1 ;
    }

    if (p_nz)
    {
	*p_nz = Symbolic->nz ;
    }

    if (p_nfr)
    {
	*p_nfr = nfr ;
    }

    if (p_nchains)
    {
	*p_nchains = nchains ;
    }

    if (P != (Int *) NULL)
    {
	Int *Rperm_init, *Diagonal_map ;
	Rperm_init = Symbolic->Rperm_init ;
	Diagonal_map = Symbolic->Diagonal_map ;
	if (Diagonal_map != (Int *) NULL)
	{
	    ASSERT (n_row == n_col) ;
        //printf ("Diagonal_map is present\n") ;
	    //for (k = 0 ; k < n1 ; k++)
	    //{
        //    P [k] = Rperm_init [k] ;
	    //}
	    ///* next pivot rows are found in the diagonal map */
	    //for (k = n1 ; k < n_row ; k++)
	    //{
        //        Int knew = Diagonal_map [k] ;
        //        ASSERT (knew >= n1) ;
        //        // FIXME: remove this:
        //        //if (knew < n1) { printf ("Hey!!! this broke\n") ; abort ( ) ; }
        //        P [k] = Rperm_init [knew] ;
	    //}
        //for (k = 0 ; k < n_row ; k++)
        //    Diag_map [k] = Diagonal_map [k];
	}
    else
    {
     //   Diag_map[0] = -1;
    }
    /* there is no diagonal map.  */
    for (k = 0 ; k < n_row ; k++)
    {
        P [k] = Rperm_init [k] ;
    }
    
//      printf("\nInside UMFPACK P=:\n");
//	    for (k = 0 ; k < n_row ; k++)
//                printf("%ld ", P [k]);
//        printf("\n");
    }

    if (Q != (Int *) NULL)
    {
	p = Symbolic->Cperm_init ;
	for (k = 0 ; k < n_col ; k++)
	{
	    Q [k] = p [k] ;
	}
    }

    if (Front_npivcol != (Int *) NULL)
    {
	p = Symbolic->Front_npivcol ;
	for (k = 0 ; k <= nfr ; k++)
	{
	    Front_npivcol [k] = p [k] ;
	}
    }

    if (Front_parent != (Int *) NULL)
    {
	p = Symbolic->Front_parent ;
	for (k = 0 ; k <= nfr ; k++)
	{
	    Front_parent [k] = p [k] ;
	}
    }

    if (Front_1strow != (Int *) NULL)
    {
	p = Symbolic->Front_1strow ;
	for (k = 0 ; k <= nfr ; k++)
	{
	    Front_1strow [k] = p [k] ;
	}
    }

    if (Front_leftmostdesc != (Int *) NULL)
    {
	p = Symbolic->Front_leftmostdesc ;
	for (k = 0 ; k <= nfr ; k++)
	{
	    Front_leftmostdesc [k] = p [k] ;
	}
    }

    if (Chain_start != (Int *) NULL)
    {
	p = Symbolic->Chain_start ;
	for (k = 0 ; k <= nchains ; k++)
	{
	    Chain_start [k] = p [k] ;
	}
    }

    if (Chain_maxrows != (Int *) NULL)
    {
	p = Symbolic->Chain_maxrows ;
	for (k = 0 ; k < nchains ; k++)
	{
	    Chain_maxrows [k] = p [k] ;
	}
	Chain_maxrows [nchains] = 0 ;
    }

    if (Chain_maxcols != (Int *) NULL)
    {
	p = Symbolic->Chain_maxcols ;
	for (k = 0 ; k < nchains ; k++)
	{
	    Chain_maxcols [k] = p [k] ;
	}
	Chain_maxcols [nchains] = 0 ;
    }

    return (UMFPACK_OK) ;
}
