//------------------------------------------------------------------------------
// UMFPACK/Source/umfpack_get_symbolic: extract contents of Symbolic object
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    User-callable.  Gets the symbolic information held in the Symbolic object.
    See umfpack.h for a more detailed description.
*/

#include "umf_internal.h"
#include "umf_valid_symbolic.h"

int UMFPACK_get_symbolic
(
    Int *p_n_row,
    Int *p_n_col,
    Int *p_n1,			/* number of singletons */
    Int *p_nz,
    Int *p_nfr,
    Int *p_nchains,
    Int P [ ],
    Int Q [ ],
    Int Front_npivcol [ ],
    Int Front_parent [ ],
    Int Front_1strow [ ],
    Int Front_leftmostdesc [ ],
    Int Chain_start [ ],
    Int Chain_maxrows [ ],
    Int Chain_maxcols [ ],
    Int Dmap [ ],               // added for v6.0.0
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
        // changes for v6.0.0: Diagonal_map not included in the row
        // permutation, and Rperm_init is now returned as-is.  In v5.7.9 and
        // earlier, P was constructed from Rperm_init and the Diagonal_map, but
        // this is not useful.
        Int *Rperm_init ;
        Rperm_init = Symbolic->Rperm_init ;
        for (k = 0 ; k < n_row ; k++)
        {
            P [k] = Rperm_init [k] ;
        }
    }

    if (Dmap != NULL)
    {
        if (Symbolic->Diagonal_map == NULL)
        {
            // Diagonal_Map wasn't constructed (implicit identity)
            for (k = 0 ; k < n_col ; k++)
            {
                Dmap [k] = k ;
            }
        }
        else
        {
            memcpy (Dmap, Symbolic->Diagonal_map, n_col * sizeof (Int)) ;
        }
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
