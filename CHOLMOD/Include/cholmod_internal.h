//------------------------------------------------------------------------------
// CHOLMOD/Include/cholmod_internal.h
//------------------------------------------------------------------------------

// CHOLMOD/Include/cholmod_internal.h. Copyright (C) 2005-2023,
// Timothy A. Davis.  All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// CHOLMOD internal include file.
//
// This file contains internal definitions for CHOLMOD, not meant to be
// included in user code.  They define macros that are not prefixed with
// CHOLMOD_.  This file can safely #include'd in user code if you want to make
// use of the macros defined here, and don't mind the possible name conflicts
// with your code, however.
//
// Required by all CHOLMOD routines.  Not required by any user routine that
// uses CHOLMOMD.  Unless debugging is enabled, this file does not require any
// CHOLMOD module (not even the Utility module).
//
// If debugging is enabled, all CHOLMOD modules require the Check module.
// Enabling debugging requires that this file be editted.  Debugging cannot be
// enabled with a compiler flag.  This is because CHOLMOD is exceedingly slow
// when debugging is enabled.  Debugging is meant for development of CHOLMOD
// itself, not by users of CHOLMOD.

#ifndef CHOLMOD_INTERNAL_H
#define CHOLMOD_INTERNAL_H

#define SUITESPARSE_BLAS_DEFINITIONS
#include "cholmod.h"

//------------------------------------------------------------------------------
// debugging and basic includes
//------------------------------------------------------------------------------

// turn off debugging
#ifndef NDEBUG
#define NDEBUG
#endif

// Uncomment this line to enable debugging.  CHOLMOD will be very slow.
// #undef NDEBUG

// Uncomment this line to get a summary of the time spent in the BLAS,
// for development diagnostics only:
// #define BLAS_TIMER

// Uncomment this line to get a long dump as a text file (blas_dump.txt), that
// records each call to the BLAS, for development diagnostics only:
// #define BLAS_DUMP

// if BLAS_DUMP is enabled, the BLAS_TIMER must also be enabled.
#if defined ( BLAS_DUMP ) && ! defined ( BLAS_TIMER )
#define BLAS_TIMER
#endif

//------------------------------------------------------------------------------
// basic definitions
//------------------------------------------------------------------------------

// Some non-conforming compilers insist on defining TRUE and FALSE.
#undef TRUE
#undef FALSE
#define TRUE 1
#define FALSE 0

// NULL should already be defined, but ensure it is here.
#ifndef NULL
#define NULL ((void *) 0)
#endif

// FLIP is a "negation about -1", and is used to mark an integer i that is
// normally non-negative.  FLIP (EMPTY) is EMPTY.  FLIP of a number > EMPTY
// is negative, and FLIP of a number < EMTPY is positive.  FLIP (FLIP (i)) = i
// for all integers i.  UNFLIP (i) is >= EMPTY.
#define EMPTY (-1)
#define FLIP(i) (-(i)-2)
#define UNFLIP(i) (((i) < EMPTY) ? FLIP (i) : (i))

// MAX and MIN are not safe to use for NaN's
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MAX3(a,b,c) (((a) > (b)) ? (MAX (a,c)) : (MAX (b,c)))
#define MAX4(a,b,c,d) (((a) > (b)) ? (MAX3 (a,c,d)) : (MAX3 (b,c,d)))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define IMPLIES(p,q) (!(p) || (q))

// RANGE (k, lo, hi): ensures k is in range lo:hi
#define RANGE(k,lo,hi)          \
    (((k) < (lo)) ? (lo) :      \
    (((k) > (hi)) ? (hi) : (k)))

// find the sign: -1 if x < 0, 1 if x > 0, zero otherwise.
// Not safe for NaN's
#define SIGN(x) (((x) < 0) ? (-1) : (((x) > 0) ? 1 : 0))

// round up an integer x to a multiple of s
#define ROUNDUP(x,s) ((s) * (((x) + ((s) - 1)) / (s)))

#define ERROR(status,msg) \
    CHOLMOD(error) (status, __FILE__, __LINE__, msg, Common)

// Check a pointer and return if null.  Set status to invalid, unless the
// status is already "out of memory"
#define RETURN_IF_NULL(A,result)                            \
{                                                           \
    if ((A) == NULL)                                        \
    {                                                       \
        if (Common->status != CHOLMOD_OUT_OF_MEMORY)        \
        {                                                   \
            ERROR (CHOLMOD_INVALID, "argument missing") ;   \
        }                                                   \
        return (result) ;                                   \
    }                                                       \
}

// Return if Common is NULL or invalid
#define RETURN_IF_NULL_COMMON(result)                       \
{                                                           \
    if (Common == NULL)                                     \
    {                                                       \
        return (result) ;                                   \
    }                                                       \
    if (Common->itype != ITYPE)                             \
    {                                                       \
        Common->status = CHOLMOD_INVALID ;                  \
        return (result) ;                                   \
    }                                                       \
}

// 1e308 is a huge number that doesn't take many characters to print in a
// file, in CHOLMOD/Check/cholmod_read and _write.  Numbers larger than this
// are interpretted as Inf, since sscanf doesn't read in Inf's properly.
// This assumes IEEE double precision arithmetic.  DBL_MAX would be a little
// better, except that it takes too many digits to print in a file.
#define HUGE_DOUBLE 1e308

//==============================================================================
// int32/int64 and double/single definitions
//==============================================================================

#include "cholmod_types.h"

#ifndef CHOLMOD_INT64
// GPU acceleration only available for the CHOLMOD_INT64 case (int64)
#undef CHOLMOD_HAS_CUDA
#endif

//------------------------------------------------------------------------------
// internal routines
//------------------------------------------------------------------------------

bool cholmod_mult_uint64_t      // c = a*b, return true if ok
(
    uint64_t *c,
    const uint64_t a,
    const uint64_t b
) ;

size_t cholmod_add_size_t    (size_t a, size_t b, int *ok) ;
size_t cholmod_mult_size_t   (size_t a, size_t b, int *ok) ;
size_t cholmod_l_add_size_t  (size_t a, size_t b, int *ok) ;
size_t cholmod_l_mult_size_t (size_t a, size_t b, int *ok) ;

int64_t cholmod_cumsum  // return sum (Cnz), or -1 if int32_t overflow
(
    int32_t *Cp,    // size n+1, output array, the cumsum of Cnz
    int32_t *Cnz,   // size n, input array
    size_t n        // size of Cp and Cnz
) ;

int64_t cholmod_l_cumsum  // return sum (Cnz), or -1 if int64_t overflow
(
    int64_t *Cp,    // size n+1, output array, the cumsum of Cnz
    int64_t *Cnz,   // size n, input array
    size_t n        // size of Cp and Cnz
) ;

void cholmod_set_empty
(
    int32_t *S,     // int32 array of size n
    size_t n
) ;

void cholmod_l_set_empty
(
    int64_t *S,     // int64 array of size n
    size_t n
) ;

void cholmod_to_simplicial_sym
(
    cholmod_factor *L,          // sparse factorization to modify
    int to_ll,                  // change L to hold a LL' or LDL' factorization
    cholmod_common *Common
) ;

void cholmod_l_to_simplicial_sym
(
    cholmod_factor *L,          // sparse factorization to modify
    int to_ll,                  // change L to hold a LL' or LDL' factorization
    cholmod_common *Common
) ;

//------------------------------------------------------------------------------
// operations for pattern/real/complex/zomplex
//------------------------------------------------------------------------------

// Define operations on pattern, real, complex, and zomplex objects.
//
// The xtype of an object defines it numerical type.  A qttern object has no
// numerical values (A->x and A->z are NULL).  A real object has no imaginary
// qrt (A->x is used, A->z is NULL).  A complex object has an imaginary qrt
// that is stored interleaved with its real qrt (A->x is of size 2*nz, A->z
// is NULL).  A zomplex object has both real and imaginary qrts, which are
// stored seqrately, as in MATLAB (A->x and A->z are both used).
//
// XTYPE is CHOLMOD_PATTERN, _REAL, _COMPLEX or _ZOMPLEX, and is the xtype of
// the template routine under construction.  XTYPE2 is equal to XTYPE, except
// if XTYPE is CHOLMOD_PATTERN, in which case XTYPE is CHOLMOD_REAL.
// XTYPE and XTYPE2 are defined in cholmod_template.h.

//------------------------------------------------------------------------------
// pattern: single or double
//------------------------------------------------------------------------------

#define P_TEMPLATE(name)                        p_ ## name
#define PS_TEMPLATE(name)                       ps_ ## name

#define P_ASSIGN2(x,z,p,ax,az,q)                x [p] = 1
#define P_PRINT(k,x,z,p)                        PRK(k, ("1"))

//------------------------------------------------------------------------------
// real: single or double
//------------------------------------------------------------------------------

#define RD_TEMPLATE(name)                       rd_ ## name
#define RS_TEMPLATE(name)                       rs_ ## name

#define R_ABS(x,z,p)                            fabs ((double) (x [p]))
#define R_ASSEMBLE(x,z,p,ax,az,q)               x [p] += ax [q]
#define R_ASSIGN(x,z,p,ax,az,q)                 x [p]  = ax [q]
#define R_ASSIGN_CONJ(x,z,p,ax,az,q)            x [p]  = ax [q]
#define R_ASSIGN_REAL(x,p,ax,q)                 x [p]  = ax [q]
#define R_XTYPE_OK(type)                        ((type) == CHOLMOD_REAL)
#define R_IS_NONZERO(ax,az,q)                   (ax [q] != 0)
#define R_IS_ZERO(ax,az,q)                      (ax [q] == 0)
#define R_IS_ONE(ax,az,q)                       (ax [q] == 1)
#define R_MULT(x,z,p, ax,az,q, bx,bz,r)         x [p]  = ax [q] * bx [r]
#define R_MULTADD(x,z,p, ax,az,q, bx,bz,r)      x [p] += ax [q] * bx [r]
#define R_MULTSUB(x,z,p, ax,az,q, bx,bz,r)      x [p] -= ax [q] * bx [r]
#define R_MULTADDCONJ(x,z,p, ax,az,q, bx,bz,r)  x [p] += ax [q] * bx [r]
#define R_MULTSUBCONJ(x,z,p, ax,az,q, bx,bz,r)  x [p] -= ax [q] * bx [r]
#define R_ADD(x,z,p, ax,az,q, bx,bz,r)          x [p]  = ax [q] + bx [r]
#define R_ADD_REAL(x,p, ax,q, bx,r)             x [p]  = ax [q] + bx [r]
#define R_CLEAR(x,z,p)                          x [p]  = 0
#define R_CLEAR_IMAG(x,z,p)
#define R_DIV(x,z,p,ax,az,q)                    x [p] /= ax [q]
#define R_LLDOT(x,p, ax,az,q)                   x [p] -= ax [q] * ax [q]
#define R_PRINT(k,x,z,p)                        PRK(k, ("%24.16e", x [p]))

#define R_DIV_REAL(x,z,p, ax,az,q, bx,r)        x [p] = ax [q] / bx [r]
#define R_MULT_REAL(x,z,p, ax,az,q, bx,r)       x [p] = ax [q] * bx [r]

#define R_LDLDOT(x,p, ax,az,q, bx,r)            x [p] -=(ax[q] * ax[q])/ bx[r]

//------------------------------------------------------------------------------
// complex: single or double
//------------------------------------------------------------------------------

#define CD_TEMPLATE(name)                       cd_ ## name
#define CD_T_TEMPLATE(name)                     cd_t_ ## name

#define CS_TEMPLATE(name)                       cs_ ## name
#define CS_T_TEMPLATE(name)                     cs_t_ ## name

#define C_ABS(x,z,p) SuiteSparse_config_hypot ((double) (x [2*(p)]), \
    (double) (x [2*(p)+1]))

#define C_ASSEMBLE(x,z,p,ax,az,q) \
    x [2*(p)  ] += ax [2*(q)  ] ; \
    x [2*(p)+1] += ax [2*(q)+1]

#define C_ASSIGN(x,z,p,ax,az,q) \
    x [2*(p)  ] = ax [2*(q)  ] ; \
    x [2*(p)+1] = ax [2*(q)+1]

#define C_ASSIGN_REAL(x,p,ax,q)                 x [2*(p)]  = ax [2*(q)]

#define C_ASSIGN_CONJ(x,z,p,ax,az,q) \
    x [2*(p)  ] =  ax [2*(q)  ] ; \
    x [2*(p)+1] = -ax [2*(q)+1]

#define C_XTYPE_OK(type)                ((type) == CHOLMOD_COMPLEX)

#define C_IS_NONZERO(ax,az,q) ((ax [2*(q)] != 0) || (ax [2*(q)+1] != 0))

#define C_IS_ZERO(ax,az,q) ((ax [2*(q)] == 0) && (ax [2*(q)+1] == 0))

#define C_IS_ONE(ax,az,q) \
    ((ax [2*(q)] == 1) && (ax [2*(q)+1]) == 0)

#define C_IMAG_IS_NONZERO(ax,az,q)  (ax [2*(q)+1] != 0)

#define C_MULT(x,z,p, ax,az,q, bx,bz,r) \
    x [2*(p)  ] = ax [2*(q)  ] * bx [2*(r)] - ax [2*(q)+1] * bx [2*(r)+1] ; \
    x [2*(p)+1] = ax [2*(q)+1] * bx [2*(r)] + ax [2*(q)  ] * bx [2*(r)+1]

#define C_MULTADD(x,z,p, ax,az,q, bx,bz,r) \
    x [2*(p)  ] += ax [2*(q)  ] * bx [2*(r)] - ax [2*(q)+1] * bx [2*(r)+1] ; \
    x [2*(p)+1] += ax [2*(q)+1] * bx [2*(r)] + ax [2*(q)  ] * bx [2*(r)+1]

#define C_MULTSUB(x,z,p, ax,az,q, bx,bz,r) \
    x [2*(p)  ] -= ax [2*(q)  ] * bx [2*(r)] - ax [2*(q)+1] * bx [2*(r)+1] ; \
    x [2*(p)+1] -= ax [2*(q)+1] * bx [2*(r)] + ax [2*(q)  ] * bx [2*(r)+1]

// s += conj(a)*b
#define C_MULTADDCONJ(x,z,p, ax,az,q, bx,bz,r) \
    x [2*(p)  ] +=   ax [2*(q)  ]  * bx [2*(r)] + ax [2*(q)+1] * bx [2*(r)+1] ;\
    x [2*(p)+1] += (-ax [2*(q)+1]) * bx [2*(r)] + ax [2*(q)  ] * bx [2*(r)+1]

// s -= conj(a)*b
#define C_MULTSUBCONJ(x,z,p, ax,az,q, bx,bz,r) \
    x [2*(p)  ] -=   ax [2*(q)  ]  * bx [2*(r)] + ax [2*(q)+1] * bx [2*(r)+1] ;\
    x [2*(p)+1] -= (-ax [2*(q)+1]) * bx [2*(r)] + ax [2*(q)  ] * bx [2*(r)+1]

#define C_ADD(x,z,p, ax,az,q, bx,bz,r) \
    x [2*(p)  ] = ax [2*(q)  ] + bx [2*(r)  ] ; \
    x [2*(p)+1] = ax [2*(q)+1] + bx [2*(r)+1]

#define C_ADD_REAL(x,p, ax,q, bx,r) \
    x [2*(p)] = ax [2*(q)] + bx [2*(r)]

#define C_CLEAR(x,z,p) \
    x [2*(p)  ] = 0 ; \
    x [2*(p)+1] = 0

#define C_CLEAR_IMAG(x,z,p) \
    x [2*(p)+1] = 0

// s = s / a (complex double case)
#define C_DIV(x,z,p,ax,az,q) \
    SuiteSparse_config_divcomplex ( \
              x [2*(p)],  x [2*(p)+1], \
             ax [2*(q)], ax [2*(q)+1], \
             &x [2*(p)], &x [2*(p)+1])

// s = s / a (complex single case)
#define C_S_DIV(x,z,p,ax,az,q)                                  \
{                                                               \
    double cr, ci ;                                             \
    SuiteSparse_config_divcomplex (                             \
        (double)  x [2*(p)], (double)  x [2*(p)+1],             \
        (double) ax [2*(q)], (double) ax [2*(q)+1],             \
        &cr, &ci) ;                                             \
    x [2*(p)  ] = (float) cr ;                                  \
    x [2*(p)+1] = (float) ci ;                                  \
}

// s -= conj(a)*a ; note that the result of conj(a)*a is real
#define C_LLDOT(x,p, ax,az,q) \
    x [2*(p)] -= ax [2*(q)] * ax [2*(q)] + ax [2*(q)+1] * ax [2*(q)+1]

#define C_PRINT(k,x,z,p) PRK(k, ("(%24.16e,%24.16e)", x [2*(p)], x [2*(p)+1]))

#define C_DIV_REAL(x,z,p, ax,az,q, bx,r) \
    x [2*(p)  ] = ax [2*(q)  ] / bx [2*(r)] ; \
    x [2*(p)+1] = ax [2*(q)+1] / bx [2*(r)]

#define C_MULT_REAL(x,z,p, ax,az,q, bx,r) \
    x [2*(p)  ] = ax [2*(q)  ] * bx [2*(r)] ; \
    x [2*(p)+1] = ax [2*(q)+1] * bx [2*(r)]

// s -= conj(a)*a/t
#define C_LDLDOT(x,p, ax,az,q, bx,r) \
    x [2*(p)] -= (ax [2*(q)] * ax [2*(q)] + ax [2*(q)+1] * ax [2*(q)+1]) / bx[r]

//------------------------------------------------------------------------------
// zomplex: single or double
//------------------------------------------------------------------------------

#define ZD_TEMPLATE(name)                       zd_ ## name
#define ZD_T_TEMPLATE(name)                     zd_t_ ## name

#define ZS_TEMPLATE(name)                       zs_ ## name
#define ZS_T_TEMPLATE(name)                     zs_t_ ## name

#define Z_ABS(x,z,p) SuiteSparse_config_hypot ((double) (x [p]), \
    (double) (z [p]))

#define Z_ASSEMBLE(x,z,p,ax,az,q) \
    x [p] += ax [q] ; \
    z [p] += az [q]

#define Z_ASSIGN(x,z,p,ax,az,q) \
    x [p] = ax [q] ; \
    z [p] = az [q]

#define Z_ASSIGN_REAL(x,p,ax,q)                 x [p]  = ax [q]

#define Z_ASSIGN_CONJ(x,z,p,ax,az,q) \
    x [p] =  ax [q] ; \
    z [p] = -az [q]

#define Z_XTYPE_OK(type)                ((type) == CHOLMOD_ZOMPLEX)

#define Z_IS_NONZERO(ax,az,q) ((ax [q] != 0) || (az [q] != 0))

#define Z_IS_ZERO(ax,az,q) ((ax [q] == 0) && (az [q] == 0))

#define Z_IS_ONE(ax,az,q)  ((ax [q] == 1) && (az [q] == 0))

#define Z_IMAG_IS_NONZERO(ax,az,q)  (az [q] != 0)

#define Z_MULT(x,z,p, ax,az,q, bx,bz,r) \
    x [p] = ax [q] * bx [r] - az [q] * bz [r] ; \
    z [p] = az [q] * bx [r] + ax [q] * bz [r]

#define Z_MULTADD(x,z,p, ax,az,q, bx,bz,r) \
    x [p] += ax [q] * bx [r] - az [q] * bz [r] ; \
    z [p] += az [q] * bx [r] + ax [q] * bz [r]

#define Z_MULTSUB(x,z,p, ax,az,q, bx,bz,r) \
    x [p] -= ax [q] * bx [r] - az [q] * bz [r] ; \
    z [p] -= az [q] * bx [r] + ax [q] * bz [r]

#define Z_MULTADDCONJ(x,z,p, ax,az,q, bx,bz,r) \
    x [p] +=   ax [q]  * bx [r] + az [q] * bz [r] ; \
    z [p] += (-az [q]) * bx [r] + ax [q] * bz [r]

#define Z_MULTSUBCONJ(x,z,p, ax,az,q, bx,bz,r) \
    x [p] -=   ax [q]  * bx [r] + az [q] * bz [r] ; \
    z [p] -= (-az [q]) * bx [r] + ax [q] * bz [r]

#define Z_ADD(x,z,p, ax,az,q, bx,bz,r) \
    x [p] = ax [q] + bx [r] ; \
    z [p] = az [q] + bz [r]

#define Z_ADD_REAL(x,p, ax,q, bx,r) \
    x [p] = ax [q] + bx [r]

#define Z_CLEAR(x,z,p) \
    x [p] = 0 ; \
    z [p] = 0

#define Z_CLEAR_IMAG(x,z,p) \
    z [p] = 0

// s = s / a (zomplex double case)
#define Z_DIV(x,z,p,ax,az,q) \
    SuiteSparse_config_divcomplex \
        (x [p], z [p], ax [q], az [q], &x [p], &z [p])

// s = s / a (zomplex single case)
#define Z_S_DIV(x,z,p,ax,az,q)                                  \
{                                                               \
    double cr, ci ;                                             \
    SuiteSparse_config_divcomplex (                             \
        (double)  x [p], (double)  z [p],                       \
        (double) ax [q], (double) az [q],                       \
        &cr, &ci) ;                                             \
    x [p] = (float) cr ;                                        \
    z [p] = (float) ci ;                                        \
}

// s -= conj(a)*a ; note that the result of conj(a)*a is real
#define Z_LLDOT(x,p, ax,az,q) \
    x [p] -= ax [q] * ax [q] + az [q] * az [q]

#define Z_PRINT(k,x,z,p)        PRK(k, ("(%24.16e,%24.16e)", x [p], z [p]))

#define Z_DIV_REAL(x,z,p, ax,az,q, bx,r) \
    x [p] = ax [q] / bx [r] ; \
    z [p] = az [q] / bx [r]

#define Z_MULT_REAL(x,z,p, ax,az,q, bx,r) \
    x [p] = ax [q] * bx [r] ; \
    z [p] = az [q] * bx [r]

// s -= conj(a)*a/t
#define Z_LDLDOT(x,p, ax,az,q, bx,r) \
    x [p] -= (ax [q] * ax [q] + az [q] * az [q]) / bx[r]

//------------------------------------------------------------------------------
// all classes
//------------------------------------------------------------------------------

// Check if A->xtype and the two arrays A->x and A->z are valid.  Set status to
// invalid, unless status is already "out of memory".  A can be a sparse matrix,
// dense matrix, factor, or triplet.

#define RETURN_IF_XTYPE_IS_INVALID(xtype,xtype1,xtype2,result)      \
    if (xtype < (xtype1) || xtype > (xtype2))                       \
    {                                                               \
        if (Common->status != CHOLMOD_OUT_OF_MEMORY)                \
        {                                                           \
            ERROR (CHOLMOD_INVALID, "invalid xtype") ;              \
        }                                                           \
        return (result) ;                                           \
    }                                                               \

#define RETURN_IF_XTYPE_INVALID(A,xtype1,xtype2,result)                       \
{                                                                             \
    if ((A)->xtype < (xtype1) || (A)->xtype > (xtype2) ||                     \
        ((A)->xtype != CHOLMOD_PATTERN && ((A)->x) == NULL) ||                \
        ((A)->xtype == CHOLMOD_ZOMPLEX && ((A)->z) == NULL) ||                \
        !(((A)->dtype == CHOLMOD_DOUBLE) || ((A)->dtype == CHOLMOD_SINGLE)))  \
    {                                                                         \
        if (Common->status != CHOLMOD_OUT_OF_MEMORY)                          \
        {                                                                     \
            ERROR (CHOLMOD_INVALID, "invalid xtype or dtype") ;               \
        }                                                                     \
        return (result) ;                                                     \
    }                                                                         \
}

#define RETURN_IF_DENSE_MATRIX_INVALID(X,result)                            \
    RETURN_IF_NULL (X, result) ;                                            \
    RETURN_IF_XTYPE_INVALID (X, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, result) ;    \
    if ((X)->d < (X)->nrow)                                                 \
    {                                                                       \
        if (Common->status != CHOLMOD_OUT_OF_MEMORY)                        \
        {                                                                   \
            ERROR (CHOLMOD_INVALID, "dense matrix invalid") ;               \
        }                                                                   \
        return (result) ;                                                   \
    }

#define RETURN_IF_SPARSE_MATRIX_INVALID(A,result)                           \
    RETURN_IF_NULL (A, result) ;                                            \
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, result) ; \
    if ((A)->p == NULL || (!(A)->packed && ((A)->nz == NULL)) ||            \
        (A->stype != 0 && A->nrow != A->ncol))                              \
    {                                                                       \
        if (Common->status != CHOLMOD_OUT_OF_MEMORY)                        \
        {                                                                   \
            ERROR (CHOLMOD_INVALID, "sparse matrix invalid") ;              \
        }                                                                   \
        return (result) ;                                                   \
    }

#define RETURN_IF_TRIPLET_MATRIX_INVALID(T,result)                          \
    RETURN_IF_NULL (T, result) ;                                            \
    RETURN_IF_XTYPE_INVALID (T, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, result) ; \
    if ((T)->nnz > 0 && ((T)->i == NULL || (T)->j == NULL ||                \
        ((T)->xtype != CHOLMOD_PATTERN && (T)->x == NULL) ||                \
        ((T)->xtype == CHOLMOD_ZOMPLEX && (T)->z == NULL)))                 \
    {                                                                       \
        if (Common->status != CHOLMOD_OUT_OF_MEMORY)                        \
        {                                                                   \
            ERROR (CHOLMOD_INVALID, "triplet matrix invalid") ;             \
        }                                                                   \
        return (result) ;                                                   \
    }

#define RETURN_IF_FACTOR_INVALID(L,result)                                  \
    RETURN_IF_NULL (L, result) ;                                            \
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, result) ;

//==============================================================================
//=== openmp support ===========================================================
//==============================================================================

static inline int cholmod_nthreads  // returns # of OpenMP threads to use
(
    double work,                    // total work to do
    cholmod_common *Common
)
{
    #ifdef _OPENMP
    double chunk = Common->chunk ;  // give each thread at least this much work
    int nthreads_max = Common->nthreads_max ;   // max # of threads to use
    if (nthreads_max <= 0)
    {
        nthreads_max = SUITESPARSE_OPENMP_MAX_THREADS ;
    }
    work  = MAX (work, 1) ;
    chunk = MAX (chunk, 1) ;
    int64_t nthreads = (int64_t) floor (work / chunk) ;
    nthreads = MIN (nthreads, nthreads_max) ;
    nthreads = MAX (nthreads, 1) ;
    return ((int) nthreads) ;
    #else
    return (1) ;
    #endif
}

//==============================================================================
//==== debugging definitions ===================================================
//==============================================================================

#if 0
#if 0
#define GOTCHA ;
#else
#define GOTCHA                                          \
{                                                       \
    printf ("Gotcha! %d:%s\n", __LINE__, __FILE__) ;    \
    fflush (stdout) ;                                   \
    abort ( ) ;                                         \
}
#endif
#endif

#ifndef NDEBUG

#include <assert.h>

// The cholmod_dump routines are in the Check module.  No CHOLMOD routine
// calls the cholmod_check_* or cholmod_print_* routines in the Check module,
// since they use Common workspace that may already be in use.  Instead, they
// use the cholmod_dump_* routines defined there, which allocate their own
// workspace if they need it.

#ifndef EXTERN
#define EXTERN extern
#endif

// int32_t
EXTERN int cholmod_dump ;
EXTERN int cholmod_dump_malloc ;
int64_t cholmod_dump_sparse (cholmod_sparse  *, const char *,
    cholmod_common *) ;
int  cholmod_dump_factor (cholmod_factor  *, const char *, cholmod_common *) ;
int  cholmod_dump_triplet (cholmod_triplet *, const char *, cholmod_common *) ;
int  cholmod_dump_dense (cholmod_dense   *, const char *, cholmod_common *) ;
int  cholmod_dump_subset (int *, size_t, size_t, const char *,
    cholmod_common *) ;
int  cholmod_dump_perm (int *, size_t, size_t, const char *, cholmod_common *) ;
int  cholmod_dump_parent (int *, size_t, const char *, cholmod_common *) ;
void cholmod_dump_init (const char *, cholmod_common *) ;
int  cholmod_dump_mem (const char *, int64_t, cholmod_common *) ;
void cholmod_dump_real (const char *, void *, int, int64_t,
    int64_t, int, int, cholmod_common *) ;
void cholmod_dump_super (int64_t, int *, int *, int *, int *, void *, int,
    int, cholmod_common *) ;
int  cholmod_dump_partition (int64_t, int *, int *, int *, int *,
    int64_t, cholmod_common *) ;
int  cholmod_dump_work(int, int, int64_t, int, cholmod_common *) ;

// int64_t
EXTERN int cholmod_l_dump ;
EXTERN int cholmod_l_dump_malloc ;
int64_t cholmod_l_dump_sparse (cholmod_sparse  *, const char *,
    cholmod_common *) ;
int  cholmod_l_dump_factor (cholmod_factor  *, const char *, cholmod_common *) ;
int  cholmod_l_dump_triplet (cholmod_triplet *, const char *, cholmod_common *);
int  cholmod_l_dump_dense (cholmod_dense   *, const char *, cholmod_common *) ;
int  cholmod_l_dump_subset (int64_t *, size_t, size_t, const char *,
    cholmod_common *) ;
int  cholmod_l_dump_perm (int64_t *, size_t, size_t, const char *,
    cholmod_common *) ;
int  cholmod_l_dump_parent (int64_t *, size_t, const char *,
    cholmod_common *) ;
void cholmod_l_dump_init (const char *, cholmod_common *) ;
int  cholmod_l_dump_mem (const char *, int64_t, cholmod_common *) ;
void cholmod_l_dump_real (const char *, void *, int, int64_t,
    int64_t, int, int, cholmod_common *) ;
void cholmod_l_dump_super (int64_t, int64_t *,
    int64_t *, int64_t *, int64_t *,
    void *, int, int, cholmod_common *) ;
int  cholmod_l_dump_partition (int64_t, int64_t *,
    int64_t *, int64_t *,
    int64_t *, int64_t, cholmod_common *) ;
int  cholmod_l_dump_work(int, int, int64_t, int, cholmod_common *) ;

#define DEBUG_INIT(s,Common)  { CHOLMOD(dump_init)(s, Common) ; }
#ifdef MATLAB_MEX_FILE
#define ASSERT(expression) (mxAssert ((expression), ""))
#else
#define ASSERT(expression) (assert (expression))
#endif

#define PRK(k,params)                                           \
{                                                               \
    if (CHOLMOD(dump) >= (k))                                   \
    {                                                           \
        int (*printf_func) (const char *, ...) ;                \
        printf_func = SuiteSparse_config_printf_func_get ( ) ;  \
        if (printf_func != NULL)                                \
        {                                                       \
            (void) (printf_func) params ;                       \
        }                                                       \
    }                                                           \
}

#define PRINT0(params) PRK (0, params)
#define PRINT1(params) PRK (1, params)
#define PRINT2(params) PRK (2, params)
#define PRINT3(params) PRK (3, params)

void CM_memtable_dump (void) ;
int CM_memtable_n (void) ;
void CM_memtable_clear (void) ;
void CM_memtable_add (void *p, size_t size) ;
size_t CM_memtable_size (void *p) ;
bool CM_memtable_find (void *p) ;
void CM_memtable_remove (void *p) ;

#define PRINTM(params)              \
{                                   \
    if (CHOLMOD(dump_malloc) > 0)   \
    {                               \
        printf params ;             \
    }                               \
}

#define DEBUG(statement) statement

static bool check_flag (cholmod_common *Common)
{
    int64_t mark = Common->mark ;
    size_t n = Common->nrow ;
    if (Common->itype == CHOLMOD_LONG)
    {
        int64_t *Flag = Common->Flag ;
        for (int64_t i = 0 ; i < n ; i++)
        {
            if (Flag [i] >= mark) return (false) ;
        }
    }
    else
    {
        ASSERT (mark <= INT32_MAX) ;
        int32_t *Flag = Common->Flag ;
        for (int32_t i = 0 ; i < n ; i++)
        {
            if (Flag [i] >= mark) return (false) ;
        }
    }
    return (true) ;
}

#else

// Debugging disabled (the normal case)
#define PRK(k,params)
#define DEBUG_INIT(s,Common)
#define PRINT0(params)
#define PRINT1(params)
#define PRINT2(params)
#define PRINT3(params)
#define PRINTM(params)
#define ASSERT(expression)
#define DEBUG(statement)
#endif

#endif
