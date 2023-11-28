//------------------------------------------------------------------------------
// LAGraph/src/test/test_Xinit.c:  test LAGr_Init and LAGraph_Global
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

#include "LAGraph_test.h"

//------------------------------------------------------------------------------
// global variables
//------------------------------------------------------------------------------

char msg [LAGRAPH_MSG_LEN] ;

//------------------------------------------------------------------------------
// test_Xinit:  test LAGr_Init
//------------------------------------------------------------------------------

void test_Xinit (void)
{

    printf ("\nTesting LAGr_Init:\n") ;

    TEST_CHECK (LAGr_Init (GrB_NONBLOCKING, NULL, NULL, NULL, NULL, msg)
        == GrB_NULL_POINTER) ;
    printf ("msg: %s\n", msg) ;

    TEST_CHECK (LAGr_Init (GrB_NONBLOCKING, malloc, NULL, NULL, NULL, msg)
        == GrB_NULL_POINTER) ;
    printf ("msg: %s\n", msg) ;

    TEST_CHECK (LAGr_Init (GrB_NONBLOCKING, NULL, NULL, NULL, free, msg)
        == GrB_NULL_POINTER) ;
    printf ("msg: %s\n", msg) ;

    OK (LAGr_Init (GrB_NONBLOCKING, malloc, calloc, realloc, free, msg)) ;
    printf ("msg: [%s]\n", msg) ;

    // LAGr_Init cannot be called twice
    int status = LAGr_Init (GrB_NONBLOCKING,
        malloc, calloc, realloc, free, msg) ;
    TEST_CHECK (status != GrB_SUCCESS) ;
    printf ("msg: %s\n", msg) ;

    OK (LAGraph_Finalize (msg)) ;
}

//------------------------------------------------------------------------------
// test_Xinit_brutal:  test LAGr_Init with brutal memory debug
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
bool LG_init_has_been_called_from_Xinit ;
void test_Xinit_brutal (void)
{
    // no brutal memory failures, but test LG_brutal_malloc/calloc/realloc/free
    LG_brutal = -1 ;
    LG_nmalloc = 0 ;
    OK (LAGr_Init (GrB_NONBLOCKING,
        LG_brutal_malloc, LG_brutal_calloc, LG_brutal_realloc, LG_brutal_free,
        msg)) ;

    int32_t *p = LG_brutal_malloc (42 * sizeof (int32_t)) ;
    TEST_CHECK (p != NULL) ;
    LG_brutal_free (p) ;
    p = LG_brutal_calloc (42, sizeof (int32_t)) ;
    for (int k = 0 ; k < 42 ; k++)
    {
        TEST_CHECK (p [k] == 0) ;
    }
    p = LG_brutal_realloc (p, 99 * sizeof (int32_t)) ;
    for (int k = 0 ; k < 42 ; k++)
    {
        TEST_CHECK (p [k] == 0) ;
    }
    LG_brutal_free (p) ;
    p = LG_brutal_realloc (NULL, 4 * sizeof (int32_t)) ;
    for (int k = 0 ; k < 4 ; k++)
    {
        p [k] = k ;
    }
    LG_brutal_free (p) ;

    OK (LAGraph_Finalize (msg)) ;
    TEST_CHECK (LG_nmalloc == 0) ;

    // brutal tests: keep giving the method more malloc's until it succeeds

    for (int nbrutal = 0 ; nbrutal < 1000 ; nbrutal++)
    {
        LG_brutal = nbrutal ;
        GB_Global_GrB_init_called_set (false) ;
        GrB_Info info = GxB_init (GrB_NONBLOCKING, LG_brutal_malloc,
            LG_brutal_calloc, LG_brutal_realloc, LG_brutal_free) ;
        void *p = NULL, *pnew = NULL ;
        bool ok = false ;
        if (info == GrB_SUCCESS)
        {
            p = LG_brutal_realloc (NULL, 42) ;
            pnew = NULL ;
            ok = (p != NULL) ;
            if (ok)
            {
                pnew = LG_brutal_realloc (p, 107) ;
                ok = (pnew != NULL) ;
                LG_brutal_free (ok ? pnew : p) ;
            }
        }
        if (ok)
        {
            OK (GrB_finalize ( )) ;
            printf ("\nGxB_init, finally: %d %g\n", nbrutal,
                (double) LG_nmalloc) ;
            TEST_CHECK (LG_nmalloc == 0) ;
            break ;
        }
    }

    for (int nbrutal = 0 ; nbrutal < 1000 ; nbrutal++)
    {
        LG_brutal = nbrutal ;
        GB_Global_GrB_init_called_set (false) ;
        LG_init_has_been_called_from_Xinit = false ;
        int result = LAGr_Init (GrB_NONBLOCKING,
            LG_brutal_malloc, LG_brutal_calloc,
            LG_brutal_realloc, LG_brutal_free, msg) ;
        if (result == 0)
        {
            OK (LAGraph_Finalize (msg)) ;
            printf ("LAGr_Init: finally: %d %g\n", nbrutal,
                (double) LG_nmalloc) ;
            TEST_CHECK (LG_nmalloc == 0) ;
            break ;
        }
        OK (LAGraph_Finalize (msg)) ;
    }
}
#endif

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "Xinit", test_Xinit },
    #if LAGRAPH_SUITESPARSE
    { "Xinit_brutal", test_Xinit_brutal },
    #endif
    { NULL, NULL }
} ;
