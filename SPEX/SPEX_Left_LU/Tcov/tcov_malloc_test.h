//------------------------------------------------------------------------------
// SPEX/SPEX/SPEX_Left_LU/Tcov/tcov_malloc_test.h
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#ifndef TCOV_SPEX_MALLOC_TEST_H
#define TCOV_SPEX_MALLOC_TEST_H

#include "spex_left_lu_internal.h"
#include "spex_util_internal.h"
#include "SPEX_gmp.h"

extern int64_t malloc_count ;

#define GOTCHA \
    printf ("%s, line %d, spex_gmp_ntrials = %ld, malloc_count = %ld\n", \
    __FILE__, __LINE__, spex_gmp_ntrials, malloc_count);

#define SPEX_PRINT_INFO(info)                                               \
{                                                                           \
    /*printf ("file %s line %d: ", __FILE__, __LINE__) ; */                 \
    switch(info)                                                            \
    {                                                                       \
        case SPEX_OK:              printf("SPEX_OK\n");            break;   \
        case SPEX_OUT_OF_MEMORY:   printf("OUT OF MEMORY\n");      break;   \
        case SPEX_SINGULAR:        printf("Matrix is SINGULAR\n"); break;   \
        case SPEX_INCORRECT_INPUT: printf("INCORRECT INPUT\n");    break;   \
        case SPEX_INCORRECT:       printf("SPEX_INCORRECT\n");     break;   \
        default:                   printf("unknown!\n");                    \
    }                                                                       \
}

#ifdef SPEX_CHECK
#undef SPEX_CHECK
#endif

#define SPEX_CHECK(method)          \
{                                   \
    info = (method) ;               \
    if (info != SPEX_OK)            \
    {                               \
        SPEX_PRINT_INFO (info)      \
        SPEX_FREE_ALL ;             \
        return (info) ;             \
    }                               \
}

// wrapper for malloc
void *tcov_malloc
(
    size_t size        // Size to alloc
) ;

// wrapper for calloc
void *tcov_calloc
(
    size_t n,          // Size of array
    size_t size        // Size to alloc
) ;

// wrapper for realloc
void *tcov_realloc
(
    void *p,           // Pointer to be realloced
    size_t new_size    // Size to alloc
) ;

// wrapper for free
void tcov_free
(
    void *p            // Pointer to be free
) ;

// used to test spex_gmp_reallocate
int spex_gmp_realloc_test
(
    void **p_new,
    void * p_old,
    size_t old_size,
    size_t new_size
);
#endif

