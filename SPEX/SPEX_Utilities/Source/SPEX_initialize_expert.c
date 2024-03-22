//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_initialize_expert: intialize SPEX memory functions for GMP
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// SPEX_initialize_expert initializes the working environment for SPEX with
// custom memory functions that are used for SPEX and GMP.

// The four inputs to this function are pointers to four functions with the
// same signatures as the ANSI C malloc, calloc, realloc, and free functions.
// That is:

//     #include <stdlib.h>
//     void *malloc (size_t size);
//     void *calloc (size_t nmemb, size_t size);
//     void *realloc (void *ptr, size_t size);
//     void free (void *ptr);

#include "spex_util_internal.h"

SPEX_info SPEX_initialize_expert
(
    void *(*MyMalloc) (size_t),             // user-defined malloc
    void *(*MyCalloc) (size_t, size_t),     // user-defined calloc
    void *(*MyRealloc) (void *, size_t),    // user-defined realloc
    void  (*MyFree) (void *)                // user-defined free
)
{

    if (spex_initialized ( )) return (SPEX_PANIC);

    //--------------------------------------------------------------------------
    // define the malloc/calloc/realloc/free functions
    //--------------------------------------------------------------------------

    SuiteSparse_config_malloc_func_set  (MyMalloc);
    SuiteSparse_config_calloc_func_set  (MyCalloc);
    SuiteSparse_config_realloc_func_set (MyRealloc);
    SuiteSparse_config_free_func_set    (MyFree);

    //--------------------------------------------------------------------------
    // Set GMP memory functions
    //--------------------------------------------------------------------------

    return (SPEX_initialize ( ));
}

