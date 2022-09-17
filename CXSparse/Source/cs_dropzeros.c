// CXSparse/Source/cs_dropzeros: drop zeros from a sparse matrix
// CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs.h"
static CS_INT cs_nonzero (CS_INT i, CS_INT j, CS_ENTRY aij, void *other)
{
    return (aij != 0) ;
}
CS_INT cs_dropzeros (cs *A)
{
    return (cs_fkeep (A, &cs_nonzero, NULL)) ;  /* keep all nonzero entries */
} 
