/* ========================================================================== */
/* === qrtest_C ============================================================= */
/* ========================================================================== */

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/* Test the C wrapper functions. */

#include "SuiteSparseQR_C.h"

#define MAX(a,b) (((a) > (b)) ? (a) : (b))

//------------------------------------------------------------------------------
// qrtest_c64: int64_t version
//------------------------------------------------------------------------------

#define DLONG
#define QRTESTC qrtest_c64
#include "qrtestc_template.c"
#undef DLONG
#undef QRTESTC

//------------------------------------------------------------------------------
// qrtest_c32: int32_t version
//------------------------------------------------------------------------------

#define DINT
#define QRTESTC qrtest_c32
#include "qrtestc_template.c"

//------------------------------------------------------------------------------
// qrtest_C: handles both int32_t and int64_t versions (both double, not float)
//------------------------------------------------------------------------------

void qrtest_C
(
    cholmod_sparse *A,
    double anorm,
    double errs [5],
    double maxresid [2][2],
    cholmod_common *cc
)
{
    if (A->itype == CHOLMOD_INT)
    {
        qrtest_c32 (A, anorm, errs, maxresid, cc) ;
    }
    else
    {
        qrtest_c64 (A, anorm, errs, maxresid, cc) ;
    }
}

