//------------------------------------------------------------------------------
// SLIP_LU/SLIP_malloc: wrapper for malloc
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

// Allocate memory space for SLIP_LU.

#include "slip_internal.h"

void *SLIP_malloc
(
    size_t size        // size of memory space to allocate
)
{
    if (!slip_initialized ( )) return (NULL) ;
    return (SuiteSparse_malloc (1, size)) ;
}

