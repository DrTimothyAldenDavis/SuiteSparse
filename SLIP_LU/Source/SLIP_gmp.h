//------------------------------------------------------------------------------
// SLIP_LU/SLIP_gmp.h: definitions for SLIP_gmp.c
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

// These macros are used by SLIP_gmp.c to create wrapper functions around all
// GMP functions used by SLIP_LU, to safely handle out-of-memory conditions.
// They are placed in this separate #include file so that a future developer
// can use them to construct their own wrappers around GMP functions.  See
// SLIP_gmp.c for more details.

#ifndef SLIP_GMP_H
#define SLIP_GMP_H

#define SLIP_GMP_WRAPPER_START                                          \
{                                                                       \
    slip_gmp_nmalloc = 0 ;                                              \
    /* setjmp returns 0 if called from here, or > 0 if from longjmp */  \
    int slip_gmp_status = setjmp (slip_gmp_environment) ;               \
    if (slip_gmp_status != 0)                                           \
    {                                                                   \
        /* failure from longjmp */                                      \
        slip_gmp_failure (slip_gmp_status) ;                            \
        return (SLIP_OUT_OF_MEMORY) ;                                   \
    }                                                                   \
}

#define SLIP_GMPZ_WRAPPER_START(x)                                      \
{                                                                       \
    slip_gmpz_archive = (mpz_t *) x;                                    \
    slip_gmpq_archive = NULL;                                           \
    slip_gmpfr_archive = NULL;                                          \
    SLIP_GMP_WRAPPER_START;                                             \
}

#define SLIP_GMPQ_WRAPPER_START(x)                                      \
{                                                                       \
    slip_gmpz_archive = NULL;                                           \
    slip_gmpq_archive =(mpq_t *) x;                                     \
    slip_gmpfr_archive = NULL;                                          \
    SLIP_GMP_WRAPPER_START;                                             \
}

#define SLIP_GMPFR_WRAPPER_START(x)                                     \
{                                                                       \
    slip_gmpz_archive = NULL;                                           \
    slip_gmpq_archive = NULL;                                           \
    slip_gmpfr_archive = (mpfr_t *) x;                                  \
    SLIP_GMP_WRAPPER_START;                                             \
}

#define SLIP_GMP_WRAPPER_FINISH                                         \
{                                                                       \
    /* clear (but do not free) the list.  The caller must ensure */     \
    /* the result is eventually freed. */                               \
    slip_gmpz_archive = NULL ;                                          \
    slip_gmpq_archive = NULL ;                                          \
    slip_gmpfr_archive = NULL ;                                         \
    slip_gmp_nmalloc = 0 ;                                              \
}

// free a block of memory, and also remove it from the archive if it's there
#define SLIP_GMP_SAFE_FREE(p)                                           \
{                                                                       \
    if (slip_gmpz_archive != NULL)                                      \
    {                                                                   \
        if (p == SLIP_MPZ_PTR(*slip_gmpz_archive))                      \
        {                                                               \
            SLIP_MPZ_PTR(*slip_gmpz_archive) = NULL ;                   \
        }                                                               \
    }                                                                   \
    else if (slip_gmpq_archive != NULL)                                 \
    {                                                                   \
        if (p == SLIP_MPZ_PTR(SLIP_MPQ_NUM(*slip_gmpq_archive)))        \
        {                                                               \
            SLIP_MPZ_PTR(SLIP_MPQ_NUM(*slip_gmpq_archive)) = NULL ;     \
        }                                                               \
        if (p == SLIP_MPZ_PTR(SLIP_MPQ_DEN(*slip_gmpq_archive)))        \
        {                                                               \
            SLIP_MPZ_PTR(SLIP_MPQ_DEN(*slip_gmpq_archive)) = NULL ;     \
        }                                                               \
    }                                                                   \
    else if (slip_gmpfr_archive != NULL)                                \
    {                                                                   \
        if (p == SLIP_MPFR_REAL_PTR(*slip_gmpfr_archive))               \
        {                                                               \
            SLIP_MPFR_MANT(*slip_gmpfr_archive) = NULL ;                \
        }                                                               \
    }                                                                   \
    SLIP_FREE (p) ;                                                     \
}

#endif

