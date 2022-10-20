// CSparse/Source/cs_dropzeros: drop zeros from a sparse matrix
// CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs.h"
static csi cs_nonzero (csi i, csi j, double aij, void *other)
{
    return (aij != 0) ;
}
csi cs_dropzeros (cs *A)
{
    return (cs_fkeep (A, &cs_nonzero, NULL)) ;  /* keep all nonzero entries */
} 
