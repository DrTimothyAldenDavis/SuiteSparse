// CXSparse/Source/cs_malloc: wrappers for malloc/calloc/realloc/free
// CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs.h"

/* wrapper for malloc */
void *cs_malloc (CS_INT n, size_t size)
{
    return (SuiteSparse_config_malloc (CS_MAX (n,1) * size)) ;
}

/* wrapper for calloc */
void *cs_calloc (CS_INT n, size_t size)
{
    return (SuiteSparse_config_calloc (CS_MAX (n,1), size)) ;
}

/* wrapper for free */
void *cs_free (void *p)
{
    if (p)
    {
        /* free p if it is not already NULL */
        SuiteSparse_config_free (p) ;
    }
    return (NULL) ;         /* return NULL to simplify the use of cs_free */
}

/* wrapper for realloc */
void *cs_realloc (void *p, CS_INT n, size_t size, CS_INT *ok)
{
    void *pnew ;
    /* realloc the block */
    pnew = SuiteSparse_config_realloc (p, CS_MAX (n,1) * size) ;
    *ok = (pnew != NULL) ;                  /* realloc fails if pnew is NULL */
    return ((*ok) ? pnew : p) ;             /* return original p if failure */
}
