//------------------------------------------------------------------------------
// UMFPACK/Source/umfpack_report_vector: print a dense vector
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    User-callable.  Prints a real or complex vector.
    See umfpack.h for details.
*/

#include "umf_internal.h"
#include "umf_report_vector.h"

int UMFPACK_report_vector
(
    Int n,
    const double Xx [ ],
#ifdef COMPLEX
    const double Xz [ ],
#endif
    const double Control [UMFPACK_CONTROL]
)
{
    Int prl ;

#ifndef COMPLEX
    double *Xz = (double *) NULL ;
#endif

    prl = GET_CONTROL (UMFPACK_PRL, UMFPACK_DEFAULT_PRL) ;

    if (prl <= 2)
    {
	return (UMFPACK_OK) ;
    }

    return (UMF_report_vector (n, Xx, Xz, prl, TRUE, FALSE)) ;
}
