//------------------------------------------------------------------------------
// SPEX_Util/SPEX_initialize: initialize SPEX
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// SPEX_initialize initializes the working evironment for SPEX

#include "spex_util_internal.h"

//------------------------------------------------------------------------------
// global variable access
//------------------------------------------------------------------------------

// a global variable, but only accessible within this file.
extern bool spex_initialize_has_been_called ;

bool spex_initialize_has_been_called = false ;

bool spex_initialized ( void )
{
    return (spex_initialize_has_been_called) ;
}

void spex_set_initialized (bool s)
{
    spex_initialize_has_been_called = s ;
}

//------------------------------------------------------------------------------
// SPEX_initialize
//------------------------------------------------------------------------------

SPEX_info SPEX_initialize ( void )
{
    if (spex_initialized ( )) return (SPEX_PANIC) ;

    mp_set_memory_functions (spex_gmp_allocate, spex_gmp_reallocate,
        spex_gmp_free) ;

    spex_set_initialized (true) ;
    return (SPEX_OK) ;
}

