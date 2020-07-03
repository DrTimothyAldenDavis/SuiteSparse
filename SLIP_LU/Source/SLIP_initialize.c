//------------------------------------------------------------------------------
// SLIP_LU/SLIP_initialize: initialize SLIP_LU
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

// SLIP_initialize initializes the working evironment for SLIP_LU.

#include "slip_internal.h"

//------------------------------------------------------------------------------
// global variable access
//------------------------------------------------------------------------------

// a global variable, but only accessible within this file.
extern bool slip_initialize_has_been_called ;

bool slip_initialize_has_been_called = false ;

bool slip_initialized ( void )
{
    return (slip_initialize_has_been_called) ;
}

void slip_set_initialized (bool s)
{
    slip_initialize_has_been_called = s ;
}

//------------------------------------------------------------------------------
// SLIP_initialize
//------------------------------------------------------------------------------

SLIP_info SLIP_initialize ( void )
{
    if (slip_initialized ( )) return (SLIP_PANIC) ;

    mp_set_memory_functions (slip_gmp_allocate, slip_gmp_reallocate,
        slip_gmp_free) ;

    slip_set_initialized (true) ;
    return (SLIP_OK) ;
}

