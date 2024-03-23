// ----------------------------------------------------------------------------
// SPEX/Tcov/tcov_utilities.h: utilities for tcov tests
// ----------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//-----------------------------------------------------------------------------

#ifndef TCOV_UTILITIES_H
#define TCOV_UTILITIES_H

#include "spex_lu_internal.h"
#include <assert.h>
#include <float.h>

extern int64_t malloc_count ;

#define SPEX_PRINT_INFO(info)                                               \
{                                                                           \
    switch(info)                                                            \
    {                                                                       \
        case SPEX_OK:              printf("SPEX_OK\n");            break;   \
        case SPEX_OUT_OF_MEMORY:   printf("OUT OF MEMORY\n");      break;   \
        case SPEX_SINGULAR:        printf("Matrix is SINGULAR\n"); break;   \
        case SPEX_INCORRECT_INPUT: printf("INCORRECT INPUT\n");    break;   \
        case SPEX_NOTSPD:          printf("Matrix is not SPD\n");  break;   \
        case SPEX_PANIC:           printf("SPEX_PANIC\n");         break;   \
        default:                   printf("unknown!\n");                    \
    }                                                                       \
}

// abort the test if a test failure occurs
#define TEST_ABORT(info)                                                    \
{                                                                           \
    SPEX_PRINT_INFO (info) ;                                                \
    printf ("test failure: %s at line %d\n", __FILE__,__LINE__) ;           \
    fprintf (stderr, "test failure: %s at line %d\n", __FILE__, __LINE__) ; \
    fflush (stdout) ;                                                       \
    fflush (stderr) ;                                                       \
    abort ( ) ;                                                             \
}

// assert something that should be true
#define TEST_ASSERT(ok)                             \
{                                                   \
    if (!(ok)) TEST_ABORT (SPEX_PANIC) ;            \
}

// OK: call a method and assert that it succeeds
// The method must return a SPEX_info value.
#define OK(method)                              \
{                                               \
    SPEX_info info3 = (method) ;                \
    if (info3 != SPEX_OK) TEST_ABORT (info3) ;  \
}

// OK2: call a method and assert that it succeeds or runs out of memory.
// If the method runs out of memory, all workspace is freed and control is
// returned to the caller.  The method must return a SPEX_info value.
#define OK2(method)                             \
{                                               \
    SPEX_info info4 = (method) ;                \
    if (info4 == SPEX_OUT_OF_MEMORY)            \
    {                                           \
        SPEX_FREE_ALL;                          \
        return (info4) ;                        \
    }                                           \
    if (info4 != SPEX_OK) TEST_ABORT (info4) ;  \
}

// test wrapper for SPEX_initialize*, SPEX_finalize, and SPEX_*_free methods
#define TEST_OK(method)                             \
if (!pretend_to_fail)                               \
{                                                   \
     OK (method) ;                                  \
}

// test wrapper for all other SPEX_* functions
#define TEST_CHECK(method)                                          \
if (!pretend_to_fail)                                               \
{                                                                   \
    info = (method) ;                                               \
    if (info == SPEX_OUT_OF_MEMORY)                                 \
    {                                                               \
        SPEX_FREE_ALL;                                              \
        pretend_to_fail = true ;                                    \
    }                                                               \
    else if (info != SPEX_OK)                                       \
    {                                                               \
        TEST_ABORT (info) ;                                         \
    }                                                               \
}

// test wrapper for SPEX_* function when expected error would produce
#define TEST_CHECK_FAILURE(method,expected_error)                       \
if (!pretend_to_fail)                                                   \
{                                                                       \
    info = (method) ;                                                   \
    if (info == SPEX_OUT_OF_MEMORY)                                     \
    {                                                                   \
        SPEX_FREE_ALL;                                                  \
        pretend_to_fail = true ;                                        \
    }                                                                   \
    else if (info != expected_error)                                    \
    {                                                                   \
        printf ("SPEX method was expected to fail, but succeeded!\n") ; \
        printf ("this error was expected:\n") ;                         \
        SPEX_PRINT_INFO (expected_error) ;                              \
        printf ("but this error was obtained:\n") ;                     \
        TEST_ABORT (info) ;                                             \
    }                                                                   \
}

// malloc function for test coverage
void *tcov_malloc
(
    size_t size        // Size to alloc
) ;

// calloc function for test coverage
void *tcov_calloc
(
    size_t n,          // Size of array (# of entries)
    size_t size        // Size of each entry to alloc
) ;

// realloc function for test coverage
void *tcov_realloc
(
    void *p,           // Pointer to be realloced
    size_t new_size    // Size to alloc
) ;

// free function for test coverage
void tcov_free
(
    void *p            // Pointer to be freed
) ;

// used to test spex_gmp_reallocate
int spex_gmp_realloc_test
(
    void **p_new,
    void * p_old,
    size_t old_size,
    size_t new_size
) ;

SPEX_info spex_check_solution
(
    // output
    bool *Is_correct,            // true, if the solution is correct
    // input
    const SPEX_matrix A,         // Input matrix of CSC MPZ
    const SPEX_matrix x,         // Solution vectors
    const SPEX_matrix b,         // Right hand side vectors
    const SPEX_options option    // Command options
) ;

#endif

