////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_cov.hpp ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PARU_COV_HPP
#define PARU_COV_HPP

#include "paru_internal.hpp"

// not user-callable: for testing only
ParU_Info paru_backward
(
    double *x1,
    double &resid,
    double &anorm,
    double &xnorm,
    cholmod_sparse *A,
    const ParU_Symbolic Sym,
    ParU_Numeric Num,
    ParU_Control Control
) ;

#define TEST_PASSES                 \
{                                   \
    printf ("all tests pass\n\n") ; \
    TEST_FREE_ALL                   \
    return (PARU_SUCCESS) ;         \
}

#define TEST_ASSERT(ok)                                                     \
{                                                                           \
    if (!(ok))                                                              \
    {                                                                       \
        printf ("TEST FAILURE: %s line: %d\n",                              \
            __FILE__, __LINE__) ;                                           \
        fprintf (stderr, "TEST FAILURE: %s line: %d\n",                     \
            __FILE__, __LINE__) ;                                           \
        fflush (stdout) ;                                                   \
        fflush (stderr) ;                                                   \
        abort ( ) ;                                                         \
    }                                                                       \
}

#define TEST_ASSERT_INFO(ok,info)                                           \
{                                                                           \
    if (!(ok))                                                              \
    {                                                                       \
        printf ("TEST FAILURE: info %d %s line: %d\n",                      \
            (int) info, __FILE__, __LINE__) ;                               \
        fprintf (stderr, "TEST FAILURE: info %d %s line: %d\n",             \
            (int) info, __FILE__, __LINE__) ;                               \
        fflush (stdout) ;                                                   \
        fflush (stderr) ;                                                   \
        abort ( ) ;                                                         \
    }                                                                       \
}

#ifdef PARU_ALLOC_TESTING

    #define BRUTAL_ALLOC_TEST(info, method)                 \
        {                                                   \
            paru_set_malloc_tracking(true);                 \
            for (int64_t nmalloc = 0;; nmalloc++)           \
            {                                               \
                if (!paru_get_nmalloc() )                   \
                    paru_set_nmalloc(nmalloc);              \
                info = method;                              \
                if (info != PARU_OUT_OF_MEMORY)             \
                {                                           \
                    /* printf("nmalloc=%ld\n",nmalloc); */  \
                    break;                                  \
                }                                           \
                if (nmalloc > 1000000)                      \
                {                                           \
                    printf ("Infinite loop!\n") ;           \
                    fprintf (stderr, "Infinite loop!\n") ;  \
                    TEST_ASSERT (false) ;                   \
                }                                           \
            }                                               \
            paru_set_malloc_tracking(false);                \
        }

#else

    #define BRUTAL_ALLOC_TEST(info, method) \
        {                                   \
            info = method;                  \
        }
    #endif

#endif

