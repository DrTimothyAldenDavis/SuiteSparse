//------------------------------------------------------------------------------
// UMFPACK/Source/umf_symbolic_usage: return final size of Symbolic object
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/* Returns the final size of the Symbolic object, in Units */

#include "umf_internal.h"
#include "umf_symbolic_usage.h"

double UMF_symbolic_usage
(
    Int n_row,
    Int n_col,
    Int nchains,
    Int nfr,
    Int esize,	    /* zero if no dense rows.  Otherwise, equal to the
		     * number of non-singleton, non-empty columns */
    Int prefer_diagonal
)
{
    double units ;

    units =
	DUNITS (SymbolicType, 1)	/* Symbolic structure */
	+ 2 * DUNITS (Int, n_col+1)	/* Cperm_init, Cdeg */
	+ 2 * DUNITS (Int, n_row+1)	/* Rperm_init, Rdeg */
	+ 3 * DUNITS (Int, nchains+1)	/* Chain_ */
	+ 4 * DUNITS (Int, nfr+1) ;	/* Front_ */

    /* if dense rows are present */
    units += DUNITS (Int, esize) ;	/* Esize */

    /* for diagonal pivoting */
    if (prefer_diagonal)
    {
	units += DUNITS (Int, n_col+1) ;    /* Diagonal_map */
    }

    return (units) ;
}
