//------------------------------------------------------------------------------
// SPEX_Util/SPEX_gmp.h: definitions for SPEX_gmp.c
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// These macros are used by SPEX_gmp.c to create wrapper functions around all
// GMP functions used by SPEX, to safely handle out-of-memory conditions.
// They are placed in this separate #include file so that a future developer
// can use them to construct their own wrappers around GMP functions.  See
// SPEX_gmp.c for more details.

#ifndef SPEX_GMP_H
#define SPEX_GMP_H

#define SPEX_GMP_WRAPPER_START                                          \
{                                                                       \
    spex_gmp_nmalloc = 0 ;                                              \
    /* setjmp returns 0 if called from here, or > 0 if from longjmp */  \
    int spex_gmp_status = setjmp (spex_gmp_environment) ;               \
    if (spex_gmp_status != 0)                                           \
    {                                                                   \
        /* failure from longjmp */                                      \
        spex_gmp_failure (spex_gmp_status) ;                            \
        return (SPEX_OUT_OF_MEMORY) ;                                   \
    }                                                                   \
}

#define SPEX_GMPZ_WRAPPER_START(x)                                      \
{                                                                       \
    spex_gmpz_archive = (mpz_t *) x;                                    \
    spex_gmpq_archive = NULL;                                           \
    spex_gmpfr_archive = NULL;                                          \
    SPEX_GMP_WRAPPER_START;                                             \
}

#define SPEX_GMPQ_WRAPPER_START(x)                                      \
{                                                                       \
    spex_gmpz_archive = NULL;                                           \
    spex_gmpq_archive =(mpq_t *) x;                                     \
    spex_gmpfr_archive = NULL;                                          \
    SPEX_GMP_WRAPPER_START;                                             \
}

#define SPEX_GMPFR_WRAPPER_START(x)                                     \
{                                                                       \
    spex_gmpz_archive = NULL;                                           \
    spex_gmpq_archive = NULL;                                           \
    spex_gmpfr_archive = (mpfr_t *) x;                                  \
    SPEX_GMP_WRAPPER_START;                                             \
}

#define SPEX_GMP_WRAPPER_FINISH                                         \
{                                                                       \
    /* clear (but do not free) the list.  The caller must ensure */     \
    /* the result is eventually freed. */                               \
    spex_gmpz_archive = NULL ;                                          \
    spex_gmpq_archive = NULL ;                                          \
    spex_gmpfr_archive = NULL ;                                         \
    spex_gmp_nmalloc = 0 ;                                              \
}

// free a block of memory, and also remove it from the archive if it's there
#define SPEX_GMP_SAFE_FREE(p)                                           \
{                                                                       \
    if (spex_gmpz_archive != NULL)                                      \
    {                                                                   \
        if (p == SPEX_MPZ_PTR(*spex_gmpz_archive))                      \
        {                                                               \
            SPEX_MPZ_PTR(*spex_gmpz_archive) = NULL ;                   \
        }                                                               \
    }                                                                   \
    else if (spex_gmpq_archive != NULL)                                 \
    {                                                                   \
        if (p == SPEX_MPZ_PTR(SPEX_MPQ_NUM(*spex_gmpq_archive)))        \
        {                                                               \
            SPEX_MPZ_PTR(SPEX_MPQ_NUM(*spex_gmpq_archive)) = NULL ;     \
        }                                                               \
        if (p == SPEX_MPZ_PTR(SPEX_MPQ_DEN(*spex_gmpq_archive)))        \
        {                                                               \
            SPEX_MPZ_PTR(SPEX_MPQ_DEN(*spex_gmpq_archive)) = NULL ;     \
        }                                                               \
    }                                                                   \
    else if (spex_gmpfr_archive != NULL)                                \
    {                                                                   \
        if (p == SPEX_MPFR_REAL_PTR(*spex_gmpfr_archive))               \
        {                                                               \
            SPEX_MPFR_MANT(*spex_gmpfr_archive) = NULL ;                \
        }                                                               \
    }                                                                   \
    SPEX_FREE (p) ;                                                     \
}

#endif

