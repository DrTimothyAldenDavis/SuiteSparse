//------------------------------------------------------------------------------
// SPEX_Utilities/spex_gmp.h: definitions for SPEX_gmp.c
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// These macros are used by SPEX_gmp.c to create wrapper functions around all
// GMP functions used by SPEX, to safely handle out-of-memory conditions.
// They are placed in this separate #include file so that a future developer
// can use them to construct their own wrappers around GMP functions.  See
// SPEX_gmp.c for more details.

#ifndef SPEX_GMP_H
#define SPEX_GMP_H

#include "SPEX.h"

//------------------------------------------------------------------------------
// spex_gmp environment
//------------------------------------------------------------------------------

#include <setjmp.h>

typedef struct
{
    jmp_buf environment ;   // for setjmp and longjmp
    int64_t nmalloc ;       // # of malloc'd objects in spex_gmp->list
    int64_t nlist ;         // size of the spex_gmp->list
    void **list ;           // list of malloc'd objects
    mpz_ptr  mpz_archive  ; // current mpz object
    mpz_ptr  mpz_archive2 ; // current second mpz object
    mpq_ptr  mpq_archive  ; // current mpq object
    mpfr_ptr mpfr_archive ; // current mpfr object
    int primary ;           // 1 if created by SPEX_initialize; 0 for
                            // SPEX_thread_initialize
} spex_gmp_t ;

#ifndef SPEX_GMP_LIST_INIT
// Initial size of the spex_gmp->list.  A size of 32 ensures that the list
// never needs to be increased in size (at least in practice; it is possible
// that GMP or MPFR could exceed this size).  The test coverage suite in
// SPEX/Tcov reduces this initial size to exercise the code, in
// SPEX/Tcov/Makefile.
#define SPEX_GMP_LIST_INIT 32
#endif

#ifdef SPEX_GMP_TEST_COVERAGE
// For testing only
void spex_set_gmp_ntrials (int64_t ntrials) ;
int64_t spex_get_gmp_ntrials (void) ;
#endif

//------------------------------------------------------------------------------
// SPEX GMP functions
//------------------------------------------------------------------------------

// uncomment this to print memory debugging info
// #define SPEX_GMP_MEMORY_DEBUG

int spex_gmp_initialize (int primary) ;

void spex_gmp_finalize (int primary) ;

spex_gmp_t *spex_gmp_get (void) ;

void *spex_gmp_allocate (size_t size) ;

void spex_gmp_free (void *p, size_t size) ;

void *spex_gmp_reallocate (void *p_old, size_t old_size, size_t new_size );

#ifdef SPEX_GMP_MEMORY_DEBUG
void spex_gmp_dump ( void ) ;
#endif

SPEX_info spex_gmp_failure (int status) ;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------------------GMP/MPFR wrapper macros------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GMP/MPFR wrapper macros that already incorporate the SPEX_CHECK macro, which
// is use to prevent segmentation faults in case gmp runs out of memory inside
// a gmp call.
//------------------------------------------------------------------------------

#define SPEX_MPZ_INIT(x)                 SPEX_CHECK( SPEX_mpz_init         (x)            )
#define SPEX_MPZ_INIT2(x,size)           SPEX_CHECK( SPEX_mpz_init2        (x,size)       )
#define SPEX_MPZ_SET(x,y)                SPEX_CHECK( SPEX_mpz_set          (x,y)          )
#define SPEX_MPZ_SET_UI(x,y)             SPEX_CHECK( SPEX_mpz_set_ui       (x,y)          )
#define SPEX_MPZ_SET_SI(x,y)             SPEX_CHECK( SPEX_mpz_set_si       (x,y)          )
#define SPEX_MPZ_SET_D(x,y)              SPEX_CHECK( SPEX_mpz_set_d        (x,y)          )
#define SPEX_MPZ_GET_D(x,y)              SPEX_CHECK( SPEX_mpz_get_d        (x,y)          )
#define SPEX_MPZ_GET_SI(x,y)             SPEX_CHECK( SPEX_mpz_get_si       (x,y)          )
#define SPEX_MPZ_MUL(a,b,c)              SPEX_CHECK( SPEX_mpz_mul          (a,b,c)        )
#define SPEX_MPZ_ADDMUL(x,y,z)           SPEX_CHECK( SPEX_mpz_addmul       (x,y,z)        )
#define SPEX_MPZ_SUB(x,y,z)              SPEX_CHECK( SPEX_mpz_sub          (x,y,z)        )
#define SPEX_MPZ_SUBMUL(x,y,z)           SPEX_CHECK( SPEX_mpz_submul       (x,y,z)        )
#define SPEX_MPZ_CDIV_QR(q,r,n,d)        SPEX_CHECK( SPEX_mpz_cdiv_qr      (q,r,n,d)      )
#define SPEX_MPZ_DIVEXACT(x,y,z)         SPEX_CHECK( SPEX_mpz_divexact     (x,y,z)        )
#define SPEX_MPZ_GCD(x,y,z)              SPEX_CHECK( SPEX_mpz_gcd          (x,y,z)        )
#define SPEX_MPZ_LCM(lcm,x,y)            SPEX_CHECK( SPEX_mpz_lcm          (lcm,x,y)      )
#define SPEX_MPZ_NEG(x,y)                SPEX_CHECK( SPEX_mpz_neg          (x,y)          )
#define SPEX_MPZ_ABS(x,y)                SPEX_CHECK( SPEX_mpz_abs          (x,y)          )
#define SPEX_MPZ_CMP(r,x,y)              SPEX_CHECK( SPEX_mpz_cmp          (r,x,y)        )
#define SPEX_MPZ_CMPABS(r,x,y)           SPEX_CHECK( SPEX_mpz_cmpabs       (r,x,y)        )
#define SPEX_MPZ_CMP_UI(r,x,y)           SPEX_CHECK( SPEX_mpz_cmp_ui       (r,x,y)        )
#define SPEX_MPZ_SGN(sgn,x)              SPEX_CHECK( SPEX_mpz_sgn          (sgn,x)        )
#define SPEX_MPZ_SIZEINBASE(size,x,base) SPEX_CHECK( SPEX_mpz_sizeinbase   (size,x,base)  )
#define SPEX_MPQ_INIT(x)                 SPEX_CHECK( SPEX_mpq_init         (x)            )
#define SPEX_MPQ_SET(x,y)                SPEX_CHECK( SPEX_mpq_set          (x,y)          )
#define SPEX_MPQ_SET_Z(x,y)              SPEX_CHECK( SPEX_mpq_set_z        (x,y)          )
#define SPEX_MPQ_SET_D(x,y)              SPEX_CHECK( SPEX_mpq_set_d        (x,y)          )
#define SPEX_MPQ_SET_UI(x,y,z)           SPEX_CHECK( SPEX_mpq_set_ui       (x,y,z)        )
#define SPEX_MPQ_SET_SI(x,y,z)           SPEX_CHECK( SPEX_mpq_set_si       (x,y,z)        )
#define SPEX_MPQ_SET_NUM(x,y)            SPEX_CHECK( SPEX_mpq_set_num      (x,y)          )
#define SPEX_MPQ_SET_DEN(x,y)            SPEX_CHECK( SPEX_mpq_set_den      (x,y)          )
#define SPEX_MPQ_GET_DEN(x,y)            SPEX_CHECK( SPEX_mpq_get_den      (x,y)          )
#define SPEX_MPQ_GET_D(x,y)              SPEX_CHECK( SPEX_mpq_get_d        (x,y)          )
#define SPEX_MPQ_SWAP(x,y)               SPEX_CHECK( SPEX_mpq_swap         (x,y)          )
#define SPEX_MPQ_NEG(x,y)                SPEX_CHECK( SPEX_mpq_neg          (x,y)          )
#define SPEX_MPQ_ABS(x,y)                SPEX_CHECK( SPEX_mpq_abs          (x,y)          )
#define SPEX_MPQ_ADD(x,y,z)              SPEX_CHECK( SPEX_mpq_add          (x,y,z)        )
#define SPEX_MPQ_MUL(x,y,z)              SPEX_CHECK( SPEX_mpq_mul          (x,y,z)        )
#define SPEX_MPQ_DIV(x,y,z)              SPEX_CHECK( SPEX_mpq_div          (x,y,z)        )
#define SPEX_MPQ_CMP(r,x,y)              SPEX_CHECK( SPEX_mpq_cmp          (r,x,y)        )
#define SPEX_MPQ_CMP_UI(r,x,num,den)     SPEX_CHECK( SPEX_mpq_cmp_ui       (r,x,num,den)  )
#define SPEX_MPQ_CMP_Z(r,x,y)            SPEX_CHECK( SPEX_mpq_cmp_z        (r,x,y)        )
#define SPEX_MPQ_EQUAL(r,x,y)            SPEX_CHECK( SPEX_mpq_equal        (r,x,y)        )
#define SPEX_MPQ_SGN(sgn,x)              SPEX_CHECK( SPEX_mpq_sgn          (sgn,x)        )
#define SPEX_MPFR_INIT2(x,size)          SPEX_CHECK( SPEX_mpfr_init2       (x,size)       )
#define SPEX_MPFR_SET(x,y,rnd)           SPEX_CHECK( SPEX_mpfr_set         (x,y,rnd)      )
#define SPEX_MPFR_SET_D(x,y,rnd)         SPEX_CHECK( SPEX_mpfr_set_d       (x,y,rnd)      )
#define SPEX_MPFR_SET_SI(x,y,rnd)        SPEX_CHECK( SPEX_mpfr_set_si      (x,y,rnd)      )
#define SPEX_MPFR_SET_Q(x,y,rnd)         SPEX_CHECK( SPEX_mpfr_set_q       (x,y,rnd)      )
#define SPEX_MPFR_SET_Z(x,y,rnd)         SPEX_CHECK( SPEX_mpfr_set_z       (x,y,rnd)      )
#define SPEX_MPFR_GET_Z(x,y,rnd)         SPEX_CHECK( SPEX_mpfr_get_z       (x,y,rnd)      )
#define SPEX_MPFR_GET_Q(x,y,rnd)         SPEX_CHECK( SPEX_mpfr_get_q       (x,y,rnd)      )
#define SPEX_MPFR_GET_D(x,y,rnd)         SPEX_CHECK( SPEX_mpfr_get_d       (x,y,rnd)      )
#define SPEX_MPFR_GET_SI(x,y,rnd)        SPEX_CHECK( SPEX_mpfr_get_si      (x,y,rnd)      )
#define SPEX_MPFR_MUL(x,y,z,rnd)         SPEX_CHECK( SPEX_mpfr_mul         (x,y,z,rnd)    )
#define SPEX_MPFR_MUL_D(x,y,z,rnd)       SPEX_CHECK( SPEX_mpfr_mul_d       (x,y,z,rnd)    )
#define SPEX_MPFR_DIV_D(x,y,z,rnd)       SPEX_CHECK( SPEX_mpfr_div_d       (x,y,z,rnd)    )
#define SPEX_MPFR_UI_POW_UI(x,y,z,rnd)   SPEX_CHECK( SPEX_mpfr_ui_pow_ui   (x,y,z,rnd)    )
#define SPEX_MPFR_LOG2(x,y,rnd)          SPEX_CHECK( SPEX_mpfr_log2        (x,y,rnd)      )
#define SPEX_MPFR_SGN(sgn,x)             SPEX_CHECK( SPEX_mpfr_sgn         (sgn,x)        )
#define SPEX_MPFR_FREE_CACHE()           SPEX_CHECK( SPEX_mpfr_free_cache  ()             )

#endif
