//------------------------------------------------------------------------------
// SPEX_Util/SPEX_initialize_expert: intialize SPEX memory functions for GMP
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// SPEX_initialize_expert initializes the working environment for SPEX with
// custom memory functions that are used for SPEX and GMP.

// The four inputs to this function are pointers to four functions with the
// same signatures as the ANSI C malloc, calloc, realloc, and free functions.
// That is:

//     #include <stdlib.h>
//     void *malloc (size_t size) ;
//     void *calloc (size_t nmemb, size_t size) ;
//     void *realloc (void *ptr, size_t size) ;
//     void free (void *ptr) ;

#include "spex_util_internal.h"

SPEX_info SPEX_initialize_expert
(
    void* (*user_malloc) (size_t),             // user-defined malloc
    void* (*user_calloc) (size_t, size_t),     // user-defined calloc
    void* (*user_realloc) (void *, size_t),    // user-defined realloc
    void  (*user_free) (void *)                // user-defined free
)
{

    if (spex_initialized ( )) return (SPEX_PANIC) ;

    //--------------------------------------------------------------------------
    // define the malloc/calloc/realloc/free functions 
    //--------------------------------------------------------------------------

    SuiteSparse_config_malloc_func_set (user_malloc) ;
    SuiteSparse_config_calloc_func_set (user_calloc) ;
    SuiteSparse_config_realloc_func_set (user_realloc) ;
    SuiteSparse_config_free_func_set (user_free) ;

    //--------------------------------------------------------------------------
    // Set GMP memory functions
    //--------------------------------------------------------------------------

    return (SPEX_initialize ( )) ;
}

