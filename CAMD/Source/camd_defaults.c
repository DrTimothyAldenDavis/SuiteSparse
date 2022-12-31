//------------------------------------------------------------------------------
// CAMD/Source/camd_defaults: set defaults for CAMD
//------------------------------------------------------------------------------

// CAMD, Copyright (c) 2007-2022, Timothy A. Davis, Yanqing Chen, Patrick R.
// Amestoy, and Iain S. Duff.  All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

/* User-callable.  Sets default control parameters for CAMD.  See camd.h
 * for details.
 */

#include "camd_internal.h"

/* ========================================================================= */
/* === CAMD defaults ======================================================= */
/* ========================================================================= */

void CAMD_defaults
(
    double Control [ ]
)
{
    Int i ;
    if (Control != (double *) NULL)
    {
	for (i = 0 ; i < CAMD_CONTROL ; i++)
	{
	    Control [i] = 0 ;
	}
	Control [CAMD_DENSE] = CAMD_DEFAULT_DENSE ;
	Control [CAMD_AGGRESSIVE] = CAMD_DEFAULT_AGGRESSIVE ;
    }
}
