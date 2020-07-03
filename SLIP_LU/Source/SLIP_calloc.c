//------------------------------------------------------------------------------
// SLIP_LU/SLIP_calloc: wrapper for calloc
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

// Allocate and initialize memory space for SLIP_LU.

#include "slip_internal.h"

void *SLIP_calloc
(
    size_t nitems,      // number of items to allocate
    size_t size         // size of each item
)
{
    if (!slip_initialized ( )) return (NULL) ;

    return (SuiteSparse_calloc (nitems, size)) ;
}

