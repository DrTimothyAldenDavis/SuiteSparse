// =============================================================================
// === qrtest ==================================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// This is an exhaustive test for SuiteSparseQR.  With the right input matrices,
// it tests each and every line of code in the package.  A malloc wrapper is
// used that can pretend to run out of memory, to test the out-of-memory
// conditions in the package.
//
// To compile and run this test, type "make".  To compile and run with valgrind,
// type "make vgo".
//
// For best results, this test requires a vanilla BLAS and LAPACK library (see
// the FLIB definition in the Makefile).  The vanilla BLAS should be the
// standard reference BLAS, and both it and LAPACK should be compiled with -g
// for best results.  With some highly-optimized BLAS packages, valgrind
// complains about not understanding some of the assembly-level instructions
// used.

#include "spqr.hpp"
#include "SuiteSparseQR_C.h"

// Use a global variable, to compute Inf.  This could be done with
// #define INF (1./0.), but the overzealous g++ compiler complains
// about divide-by-zero.
double xx = 1 ;
double yy = 0 ;         
#define INF (xx / yy)
#define CHECK_NAN(x) (((x) < 0) || ((x) != (x)) ? INF : (x))

#define NTOL 4

// =============================================================================
// === qrtest_C ================================================================
// =============================================================================

extern "C" {
void qrtest_C       // handles both int32_t and int64_t versions
(
    cholmod_sparse *A,
    double anorm,
    double errs [5],
    double maxresid [2][2],
    cholmod_common *cc
) ;
}

// =============================================================================
// === memory testing ==========================================================
// =============================================================================

int64_t my_tries = -2 ;     // number of mallocs to allow (-2 means allow all)
int64_t my_punt = FALSE ;   // if true, then my_malloc will fail just once
int64_t save_my_tries = -2 ;
int64_t save_my_punt = FALSE ;

void set_tries (int64_t tries)
{
    my_tries = tries ;
}

void set_punt (int64_t punt)
{
    my_punt = punt ;
}

void *my_malloc (size_t size)
{
    if (my_tries >= 0 || (my_punt && my_tries >= -1)) my_tries-- ;
    if (my_tries == -1)
    {
        // printf ("malloc pretends to fail\n") ;
        return (NULL) ;          // pretend to fail
    }
    return (malloc (size)) ;
}

void *my_calloc (size_t n, size_t size)
{
    if (my_tries >= 0 || (my_punt && my_tries >= -1)) my_tries-- ;
    if (my_tries == -1)
    {
        // printf ("calloc pretends to fail\n") ;
        return (NULL) ;          // pretend to fail
    }
    return (calloc (n, size)) ;
}

void *my_realloc (void *p, size_t size)
{
    if (my_tries >= 0 || (my_punt && my_tries >= -1)) my_tries-- ;
    if (my_tries == -1)
    {
        // printf ("realloc pretends to fail\n") ;
        return (NULL) ;          // pretend to fail
    }
    return (realloc (p, size)) ;
}

void my_free (void *p)
{
    if (p) free (p) ;
}

void my_handler (int status, const char *file, int line, const char *msg)
{
    printf ("ERROR: %d in %s line %d : %s\n", status, file, line, msg) ;
}

void normal_memory_handler (cholmod_common *cc, bool free_work)
{
    SuiteSparse_config_printf_func_set (printf) ;
    SuiteSparse_config_malloc_func_set (malloc) ;
    SuiteSparse_config_calloc_func_set (calloc) ;
    SuiteSparse_config_realloc_func_set (realloc) ;
    SuiteSparse_config_free_func_set (free) ;

    cc->error_handler = my_handler ;

    if (free_work)
    {
        if (cc->itype == CHOLMOD_LONG)
        {
            spqr_free_work <int64_t> (cc) ;
        }
        else
        {
            spqr_free_work <int32_t> (cc) ;
        }
    }

    my_tries = -2 ;
    my_punt = FALSE ;
}

void test_memory_handler (cholmod_common *cc, bool free_work)
{
    SuiteSparse_config_printf_func_set (NULL) ;
    SuiteSparse_config_malloc_func_set (my_malloc) ;
    SuiteSparse_config_calloc_func_set (my_calloc) ;
    SuiteSparse_config_realloc_func_set (my_realloc) ;
    SuiteSparse_config_free_func_set (my_free) ;

    cc->error_handler = NULL ;

    if (free_work)
    {
        if (cc->itype == CHOLMOD_LONG)
        {
            spqr_free_work <int64_t> (cc) ;
        }
        else
        {
            spqr_free_work <int32_t> (cc) ;
        }
    }

    my_tries = -2 ;
    my_punt = FALSE ;
}

// =============================================================================
// === qrest_template ==========================================================
// =============================================================================

#include "qrtest_template.hpp"

// =============================================================================
// === qrtest main =============================================================
// =============================================================================

#ifdef INT32
#define Int int32_t
#else
#define Int int64_t
#endif

#define LEN 200

int main (int argc, char **argv)
{
    cholmod_common Common, *cc ;
    char matrix_name [LEN+1] ;
    int kind, nfail = 0 ;

    // -------------------------------------------------------------------------
    // start CHOLMOD
    // -------------------------------------------------------------------------

    cc = &Common ;
    spqr_start <Int> (cc) ;
    normal_memory_handler (cc, true) ;

    if (argc == 1)
    {

        // ---------------------------------------------------------------------
        // Usage:  qrtest < input.mtx
        // ---------------------------------------------------------------------

        nfail += do_matrix <Int> (1, stdin, cc) ;
    }
    else
    {

        // ---------------------------------------------------------------------
        // Usage:  qrtest matrixlist
        // ---------------------------------------------------------------------

        // Each line of the matrixlist file contains an integer indicating if
        // the residuals should all be low (0=lo, 1=can be high), and a file
        // name containing the matrix in Matrix Market format.

        FILE *file = fopen (argv [1], "r") ;
        if (file == NULL)
        {
            fprintf (stderr, "Unable to open %s\n", argv [1]) ;
            exit (1) ;
        }

        while (1)
        {
            if (fscanf (file, "%d %100s\n", &kind, matrix_name) != 2)
            {
                break ;
            }
            fprintf (stderr, "%-30s ", matrix_name) ;
            FILE *matrix = fopen (matrix_name, "r") ;
            if (matrix == NULL)
            {
                fprintf (stderr, "Unable to open %s\n", matrix_name) ;
                nfail++ ;
            }
            nfail += do_matrix <Int> (kind, matrix, cc) ;

            fclose (matrix) ;
        }
        fclose (file) ;
    }

    // -------------------------------------------------------------------------
    // report the results
    // -------------------------------------------------------------------------

    spqr_finish <Int> (cc) ;

    if (cc->malloc_count != 0)
    {
        nfail++ ;
        fprintf (stderr, "memory leak: %ld objects\n", (int64_t) cc->malloc_count);
    }
    if (cc->memory_inuse != 0)
    {
        nfail++ ;
        fprintf (stderr, "memory leak: %ld bytes\n", (int64_t) cc->memory_inuse) ;
    }

    if (nfail == 0)
    {
        fprintf (stderr, "\nAll tests passed\n") ;
    }
    else
    {
        fprintf (stderr, "\nTest FAILURES: %d\n", nfail) ;
    }

    return (0) ;
}
