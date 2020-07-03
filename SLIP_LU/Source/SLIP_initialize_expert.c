//------------------------------------------------------------------------------
// SLIP_LU/SLIP_initialize_expert: intialize SLIP_LU memory functions for GMP
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

// SLIP_initialize_expert initializes the working environment for SLIP_LU with
// custom memory functions that are used for SLIP_LU and GMP.

// The four inputs to this function are pointers to four functions with the
// same signatures as the ANSI C malloc, calloc, realloc, and free functions.
// That is:

//     #include <stdlib.h>
//     void *malloc (size_t size) ;
//     void *calloc (size_t nmemb, size_t size) ;
//     void *realloc (void *ptr, size_t size) ;
//     void free (void *ptr) ;

#include "slip_internal.h"

SLIP_info SLIP_initialize_expert
(
    void* (*MyMalloc) (size_t),             // user-defined malloc
    void* (*MyCalloc) (size_t, size_t),     // user-defined calloc
    void* (*MyRealloc) (void *, size_t),    // user-defined realloc
    void  (*MyFree) (void *)                // user-defined free
)
{

    if (slip_initialized ( )) return (SLIP_PANIC) ;

    //--------------------------------------------------------------------------
    // define the malloc/calloc/realloc/free functions 
    //--------------------------------------------------------------------------

    SuiteSparse_config.malloc_func  = MyMalloc ;
    SuiteSparse_config.calloc_func  = MyCalloc ;
    SuiteSparse_config.realloc_func = MyRealloc ;
    SuiteSparse_config.free_func    = MyFree ;

    //--------------------------------------------------------------------------
    // Set GMP memory functions
    //--------------------------------------------------------------------------

    return (SLIP_initialize ( )) ;
}

