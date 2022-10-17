// CSparse/Source/cs_droptol: drop small entries from a sparse matrix
// CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs.h"
static csi cs_tol (csi i, csi j, double aij, void *tol)
{
    return (fabs (aij) > *((double *) tol)) ;
}
csi cs_droptol (cs *A, double tol)
{
    return (cs_fkeep (A, &cs_tol, &tol)) ;    /* keep all large entries */
}
