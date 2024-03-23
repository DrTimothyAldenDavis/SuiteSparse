// ----------------------------------------------------------------------------
// SPEX/Tcov/tcov_for_other.c: test coverage for other methods
// ----------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//-----------------------------------------------------------------------------

#include "tcov_utilities.h"
#include "spex_demos.h"

#undef  SPEX_FREE_ALL
#define SPEX_FREE_ALL ;

//------------------------------------------------------------------------------
// ERR: test wrapper for SPEX_* function when expected error would produce
//------------------------------------------------------------------------------

#define ERR(method,expected_error)                                      \
{                                                                       \
    SPEX_info info5 = (method) ;                                        \
    if (info5 != expected_error)                                        \
    {                                                                   \
        printf ("SPEX method was expected to fail, but succeeded!\n") ; \
        printf ("this error was expected:\n") ;                         \
        SPEX_PRINT_INFO (expected_error) ;                              \
        printf ("but this error was obtained:\n") ;                     \
        TEST_ABORT (info5) ;                                            \
    }                                                                   \
}

//------------------------------------------------------------------------------
// BRUTAL: test a method with debug malloc, until it succeeds
//------------------------------------------------------------------------------

#define NTRIAL_MAX 10000

#define BRUTAL(method)                                                      \
{                                                                           \
    int64_t trial = 0 ;                                                     \
    SPEX_info info2 = SPEX_OK ;                                             \
    for (trial = 0 ; trial <= NTRIAL_MAX ; trial++)                         \
    {                                                                       \
        malloc_count = trial ;                                              \
        info2 = (method) ;                                                  \
        if (info2 != SPEX_OUT_OF_MEMORY) break ;                            \
    }                                                                       \
    if (info2 != SPEX_OK) TEST_ABORT (info2) ;                              \
    malloc_count = INT64_MAX ;                                              \
}

//------------------------------------------------------------------------------
// test program
//------------------------------------------------------------------------------

int main (int argc, char *argv [])
{

    //--------------------------------------------------------------------------
    // start SPEX
    //--------------------------------------------------------------------------

    SPEX_options option = NULL ;

    OK (SPEX_initialize_expert (tcov_malloc, tcov_calloc, tcov_realloc,
        tcov_free)) ;

    // disable malloc testing for the first part of the test
    spex_set_gmp_ntrials (INT64_MAX) ;
    malloc_count = INT64_MAX ;

    OK (SPEX_create_default_options (&option)) ;

    //--------------------------------------------------------------------------
    // basic tests of mpfr methods
    //--------------------------------------------------------------------------

    mpfr_t x ;
    printf ("MPFR_PREC_MAX: %g\n", (double) MPFR_PREC_MAX) ;
    ERR (SPEX_mpfr_init2 (x, MPFR_PREC_MAX), SPEX_PANIC) ;
    BRUTAL (SPEX_mpfr_init2 (x, 4)) ;
    ERR (SPEX_mpfr_set_prec (x, MPFR_PREC_MAX), SPEX_PANIC) ;
    for (uint64_t k = 4 ; k < 32*1024 ; k = k*2)
    {
        BRUTAL (SPEX_mpfr_set_prec (x, k)) ;
    }
    OK (SPEX_mpfr_clear (x)) ;

    //--------------------------------------------------------------------------
    // finalize the tests
    //--------------------------------------------------------------------------

    SPEX_FREE_ALL ;
    OK (SPEX_finalize ( )) ;
    SPEX_FREE (option) ;

    printf ("%s: all tests passed\n\n", __FILE__) ;
    fprintf (stderr, "%s: all tests passed\n\n", __FILE__) ;
    return (0) ;
}

