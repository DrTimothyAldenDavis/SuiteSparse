//------------------------------------------------------------------------------
// SLIP_LU/Tcov/tcov_malloc_test.h
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

#ifndef TCOV_SLIP_MALLOC_TEST_H
#define TCOV_SLIP_MALLOC_TEST_H

#include "slip_internal.h"
#include "SLIP_gmp.h"

extern int64_t malloc_count ;

#define GOTCHA \
    printf ("%s, line %d, slip_gmp_ntrials = %ld, malloc_count = %ld\n", \
    __FILE__, __LINE__, slip_gmp_ntrials, malloc_count);

#define SLIP_PRINT_INFO(info)                                               \
{                                                                           \
    printf ("file %s line %d: ", __FILE__, __LINE__) ;                      \
    switch(info)                                                            \
    {                                                                       \
        case SLIP_OK:              printf("SLIP_OK\n");            break;   \
        case SLIP_OUT_OF_MEMORY:   printf("OUT OF MEMORY\n");      break;   \
        case SLIP_SINGULAR:        printf("Matrix is SINGULAR\n"); break;   \
        case SLIP_INCORRECT_INPUT: printf("INCORRECT INPUT\n");    break;   \
        case SLIP_INCORRECT:       printf("SLIP_INCORRECT\n");     break;   \
        default:                   printf("unknown!\n");                    \
    }                                                                       \
}

#ifdef SLIP_CHECK
#undef SLIP_CHECK
#endif

#define SLIP_CHECK(method)          \
{                                   \
    info = (method) ;               \
    if (info != SLIP_OK)            \
    {                               \
        SLIP_PRINT_INFO (info)      \
        SLIP_FREE_ALL ;             \
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

// used to test slip_gmp_reallocate
int slip_gmp_realloc_test
(
    void **p_new,
    void * p_old,
    size_t old_size,
    size_t new_size
);
#endif

