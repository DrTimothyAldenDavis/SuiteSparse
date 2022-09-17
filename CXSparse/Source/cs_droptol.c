// CXSparse/Source/cs_droptol: drop small entries from a sparse matrix
// CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs.h"
static CS_INT cs_tol (CS_INT i, CS_INT j, CS_ENTRY aij, void *tol)
{
    return (CS_ABS (aij) > *((double *) tol)) ;
}
CS_INT cs_droptol (cs *A, double tol)
{
    return (cs_fkeep (A, &cs_tol, &tol)) ;    /* keep all large entries */
}
