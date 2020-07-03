//------------------------------------------------------------------------------
// SLIP_LU/SLIP_free: wrapper for free
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

// Free the memory allocated by SLIP_calloc, SLIP_malloc, or SLIP_realloc.

#include "slip_internal.h"

void SLIP_free
(
    void *p         // pointer to memory space to free
)
{
    if (!slip_initialized ( )) return ;
    SuiteSparse_free (p) ;
}

