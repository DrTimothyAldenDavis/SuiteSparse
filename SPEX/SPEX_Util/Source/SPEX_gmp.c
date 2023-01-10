//------------------------------------------------------------------------------
// SPEX_Util/SPEX_gmp.c: interface to the gmp library
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// Purpose: This file (SPEX_gmp.c) provides a wrapper for all functions in the
// GMP library used by SPEX.  The wrappers enable memory failures to be
// caught and handled properly.  GMP, by default, aborts the user's application
// if any internal malloc fails.  This is not acceptable in a robust end-user
// application.  Fortunately, GMP allows the user package (SPEX in this
// case) to pass in function pointers for malloc, calloc, realloc, and free.
// These functions are defined below.  If they fail, they do not return to GMP.
// Instead, they use the ANSI C longjmp feature to trap the error, and return
// the error condition to the caller.

// Note that not all functions in the GMP library are wrapped by these
// functions, but just the ones used by SPEX.  However, most of the wrapper
// functions follow the same structure, and this method would easily extend to
// all GMP functions.  To that end, the wrapper mechanism (here, and in
// SPEX_gmp.h) is described below.

// For a given GMP function 'gmpfunc' with no return value, the SPEX wrapper is
// SPEX_gmpfunc, with the same arguments:

/*
    SPEX_info SPEX_gmpfunc (args)
    {
        SPEX_GMP_WRAPPER_START ;
        gmpfunc (args) ;
        SPEX_GMP_WRAPPER_FINISH ;
        return SPEX_OK ;
    }
*/

// The SPEX_GMP*_WRAPPER_START and SPEX_GMP_WRAPPER_FINISH macros work together
// to establish a try/catch mechanism, via setjmp and longjmp.  If a memory
// error occurs, a NULL is not returned to GMP (which would terminate the user
// application).  Instead, the malloc wrapper traps the error via the longjmp,
// and an out-of-memory condition is returned to the caller of SPEX_gmpfunc.

// If the gmpfunc has a return value, as in r = mpz_cmp (x,y), the return value
// is passed as the first argument to the SPEX_gmpfunc:

/*
    SPEX_info SPEX_gmfunc (result, args)
    {
        SPEX_GMP_WRAPPER_START ;
        (*result) = gmpfunc (args) ;
        SPEX_GMP_WRAPPER_FINISH ;
        return SPEX_OK ;
    }
*/

// The SPEX_GMP*_WRAPPER_START macros also take an single 'archive' parameter,
// for the current mpz, mpq, or mpfr object being operated on.  A pointer
// parameter to this parameter is kept so that it can be safely freed in case
// a memory error occurs (avoiding a double-free), in SPEX_GMP_SAFE_FREE.

#include "spex_util_internal.h"
#include "SPEX_gmp.h"

// ignore warnings about unused parameters in this file
#pragma GCC diagnostic ignored "-Wunused-parameter"

//------------------------------------------------------------------------------
// global variables
//------------------------------------------------------------------------------

jmp_buf spex_gmp_environment ;  // for setjmp and longjmp
int64_t spex_gmp_nmalloc = 0 ;  // number of malloc'd objects in SPEX_gmp_list
int64_t spex_gmp_nlist = 0 ;    // size of the SPEX_gmp_list
void **spex_gmp_list = NULL ;   // list of malloc'd objects

int64_t spex_gmp_ntrials = -1 ; // number of malloc's allowed (for
                                // testing only): -1 means unlimited.

mpz_t  *spex_gmpz_archive  = NULL ;    // current mpz object
mpq_t  *spex_gmpq_archive  = NULL ;    // current mpq object
mpfr_t *spex_gmpfr_archive = NULL ;    // current mpfr object

//------------------------------------------------------------------------------
// spex_gmp_init: initialize gmp
//------------------------------------------------------------------------------

/* Purpose: Create the list of malloc'd objects. This should be called before
 * calling any GMP function. It is also called by SPEX_gmp_allocate when
 * SPEX_gmp_list is NULL
 */

bool spex_gmp_init ( )
{
    spex_gmp_nmalloc = 0 ;
    spex_gmp_nlist = SPEX_GMP_LIST_INIT ;
    spex_gmp_list = (void **) SPEX_malloc (spex_gmp_nlist * sizeof (void *)) ;
    return (spex_gmp_list != NULL) ;
}

//------------------------------------------------------------------------------
// SPEX_gmp_finalize: finalize gmp
//------------------------------------------------------------------------------

/* Purpose: Free the list. Must be called when all use of GMP is done */
void spex_gmp_finalize ( )
{
    spex_gmpz_archive = NULL ;
    spex_gmpq_archive = NULL ;
    spex_gmpfr_archive = NULL ;
    spex_gmp_nmalloc = 0 ;
    spex_gmp_nlist = 0 ;
    SPEX_FREE (spex_gmp_list) ;
}

//------------------------------------------------------------------------------
// SPEX_gmp_allocate: malloc space for gmp
//------------------------------------------------------------------------------

/* Purpose: malloc space for gmp. A NULL pointer is never returned to the GMP
 * library. If the allocation fails, all memory allocated since the start of
 * the SPEX_gmp_wrapper is freed and an error is thrown to the GMP wrapper via
 * longjmp
 */

void *spex_gmp_allocate
(
    size_t size // Amount of memory to be allocated
)
{

    #ifdef SPEX_GMP_MEMORY_DEBUG
    SPEX_PRINTF ("spex_gmp_malloc (%g): ", (double) size) ;
    #endif

    //--------------------------------------------------------------------------
    // for testing only:
    //--------------------------------------------------------------------------

    if (spex_gmp_ntrials == 0)
    {
        // pretend to fail
        #ifdef SPEX_GMP_MEMORY_DEBUG
        SPEX_PRINTF ("spex_gmp_allocate pretends to fail\n") ;
        #endif
        longjmp (spex_gmp_environment, 1) ;
    }
    else if (spex_gmp_ntrials > 0)
    {
        // one more malloc has been used up
        spex_gmp_ntrials-- ;
    }

    //--------------------------------------------------------------------------
    // ensure the SPEX_gmp_list is large enough
    //--------------------------------------------------------------------------

    if (spex_gmp_list == NULL)
    {
        // create the initial SPEX_gmp_list
        if (!spex_gmp_init ( ))
        {
            // failure to create the SPEX_gmp_list
            longjmp (spex_gmp_environment, 2) ;
        }
    }
    else if (spex_gmp_nmalloc == spex_gmp_nlist)
    {
        // double the size of the SPEX_gmp_list
        bool ok ;
        int64_t newsize = 2 * spex_gmp_nlist ;
        spex_gmp_list = (void **)
            SPEX_realloc (newsize, spex_gmp_nlist, sizeof (void *),
            spex_gmp_list, &ok) ;
        if (!ok)
        {
            // failure to double the size of the SPEX_gmp_list.
            // The existing SPEX_gmp_list is still valid, with the old size,
            // (SPEX_gmp_nlist).  This is required so that the error handler
            // can traverse the SPEX_gmp_list to free all objects there.
            longjmp (spex_gmp_environment, 3) ;
        }
        // success; the old SPEX_gmp_list has been freed, and replaced with
        // the larger newlist.
        spex_gmp_nlist = newsize ;
    }

    //--------------------------------------------------------------------------
    // malloc the block
    //--------------------------------------------------------------------------

    void *p = SPEX_malloc (size) ;

    if (p == NULL)
    {
        // failure to allocate the new block
        longjmp (spex_gmp_environment, 4) ;
    }

    //--------------------------------------------------------------------------
    // save p in the SPEX_gmp_list and return result to GMP
    //--------------------------------------------------------------------------

    spex_gmp_list [spex_gmp_nmalloc++] = p ;

    #ifdef SPEX_GMP_MEMORY_DEBUG
    SPEX_PRINTF (" %p\n", p) ;
    spex_gmp_dump ( ) ;
    #endif

    // return p to SPEX_gmp_function (NEVER return a NULL pointer to GMP!)
    ASSERT (p != NULL) ;
    return (p) ;
}

//------------------------------------------------------------------------------
// spex_gmp_free: free space for gmp
//------------------------------------------------------------------------------

/* Purpose: Free space for GMP */
void spex_gmp_free
(
    void *p,        // Block to be freed
    size_t size     // Size of p (currently an unused parameter)
)
{
    #ifdef SPEX_GMP_MEMORY_DEBUG
    SPEX_PRINTF ("\n=================== free %p\n", p) ;
    spex_gmp_dump ( ) ;
    #endif

    if (p != NULL && spex_gmp_list != NULL)
    {
        // remove p from the SPEX_gmp_list
        for (int64_t i = 0 ; i < spex_gmp_nmalloc ; i++)
        {
            if (spex_gmp_list [i] == p)
            {
                #ifdef SPEX_GMP_MEMORY_DEBUG
                SPEX_PRINTF ("    found at i = %d\n", i) ;
                #endif
                spex_gmp_list [i] = spex_gmp_list [--spex_gmp_nmalloc] ;
                break ;
            }
        }
    }

    #ifdef SPEX_GMP_MEMORY_DEBUG
    spex_gmp_dump ( ) ;
    #endif

    // free p, even if it is not found in the SPEX_gmp_list.  p is only in the
    // SPEX_gmp_list if it was allocated inside the current GMP function.
    // If the block was allocated by one GMP function and freed by another,
    // it is not in the list.
    SPEX_GMP_SAFE_FREE (p) ;
}

//------------------------------------------------------------------------------
// spex_gmp_reallocate:  wrapper for realloc
//------------------------------------------------------------------------------

/* Purpose: Wrapper for GMP to call reallocation */
void *spex_gmp_reallocate
(
    void *p_old,        // Pointer to be realloc'd
    size_t old_size,    // Old size of p
    size_t new_size     // New size of p
)
{
    #ifdef SPEX_GMP_MEMORY_DEBUG
    SPEX_PRINTF ("spex_gmp_realloc (%p, %g, %g)\n", p_old,
        (double) old_size, (double) new_size) ;
    #endif

    if (p_old == NULL)
    {
        // realloc (NULL, size) is the same as malloc (size)
        return (spex_gmp_allocate (new_size)) ;
    }
    else if (new_size == 0)
    {
        // realloc (p, 0) is the same as free (p), and returns NULL
        spex_gmp_free (p_old, old_size) ;
        return (NULL) ;
    }
    else
    {
        // change the size of the block
        void *p_new = spex_gmp_allocate (new_size) ;
        // Note that p_new will never be NULL here, since SPEX_gmp_allocate
        // does not return if it fails.
        memcpy (p_new, p_old, SPEX_MIN (old_size, new_size)) ;
        spex_gmp_free (p_old, old_size) ;
        return (p_new) ;
    }
}

//------------------------------------------------------------------------------
// spex_gmp_dump: debug function
//------------------------------------------------------------------------------

/* Purpose: Dump the list of malloc'd objects */
#ifdef SPEX_GMP_MEMORY_DEBUG
void spex_gmp_dump ( )
{
    // dump the SPEX_gmp_list
    SPEX_PRINTF ("nmalloc = %g, SPEX_gmp_nlist = %g\n",
        (double) spex_gmp_nmalloc, (double) spex_gmp_nlist) ;
    if (spex_gmp_list != NULL)
    {
        for (int64_t i = 0 ; i < spex_gmp_nmalloc ; i++)
        {
            SPEX_PRINTF ("    spex_gmp_list [%d] = %p\n", i, spex_gmp_list [i]);
        }
    }
}
#endif

//------------------------------------------------------------------------------
// spex_gmp_failure: catch an error
//------------------------------------------------------------------------------

/* Purpose: Catch an error from longjmp */
void spex_gmp_failure
(
    int status      // Status returned from longjmp
                    // (unused parameter unless debugging)
)
{
    #ifdef SPEX_GMP_MEMORY_DEBUG
    SPEX_PRINTF ("failure from longjmp: status: %d\n", status) ;
    #endif

    // first free all caches
    mpfr_free_cache ( ) ;

    // Free the list
    if (spex_gmp_list != NULL)
    {
        for (int64_t i = 0 ; i < spex_gmp_nmalloc ; i++)
        {
            SPEX_GMP_SAFE_FREE (spex_gmp_list [i]) ;
        }
    }
    spex_gmp_finalize ( ) ;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//----------------------------Print and Scan functions--------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SPEX_gmp_fprintf
//------------------------------------------------------------------------------

/* Safely print to the stream fp. Return positive value (the number of
 * characters written) upon success, otherwise return negative value (error
 * code) */

#if 0
/* This function is currently unused, but kept here for future reference. */

SPEX_info SPEX_gmp_fprintf
(
    FILE *fp,
    const char *format,
    ...
)
{
    // Start the GMP wrapper
    SPEX_GMP_WRAPPER_START ;

    // call gmp_vfprintf
    va_list args;
    va_start (args, format) ;
    int n = gmp_vfprintf (fp, format, args) ;
    va_end (args) ;

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // gmp_vfprintf returns -1 if an error occurred.
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK) ;
}
#endif

//------------------------------------------------------------------------------
// SPEX_gmp_printf
//------------------------------------------------------------------------------

/* Safely print to the standard output stdout. Return positive value (the number
 * of characters written) upon success, otherwise return negative value (error
 * code) */
#if 0
/* This function is currently unused, but kept here for future reference. */
SPEX_info SPEX_gmp_printf
(
    const char *format,
    ...
)
{
    // Start the GMP wrapper
    SPEX_GMP_WRAPPER_START ;

    // call gmp_vprintf
    va_list args;
    va_start (args, format) ;
    int n = gmp_vprintf (format, args) ; 
    va_end (args) ;

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // gmp_vprintf returns -1 if an error occurred.
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK) ;
}
#endif

//------------------------------------------------------------------------------
// SPEX_gmp_asprintf
//------------------------------------------------------------------------------

/* Safely write the output as a null terminated string in a block of memory,
 * which is pointed to by a pointer stored in str. The block of memory must be
 * freed using SPEX_free. The return value is the number of characters
 * written in the string, excluding the null-terminator, or a negative value if
 * an error occurred */

#if 0
/* This function is currently unused, but kept here for future reference. */
/* Its functionality is provided by SPEX_mpfr_asprintf. */
SPEX_info SPEX_gmp_asprintf (char **str, const char *format, ... )
{
    // Start the GMP wrapper
    SPEX_GMP_WRAPPER_START ;

    // call gmp_vasprintf
    va_list args;
    va_start (args, format) ;
    int n = gmp_vasprintf (str, format, args) ;
    va_end (args) ;

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // gmp_vasprintf returns a negative value if an error occurred
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK) ;
}
#endif

//------------------------------------------------------------------------------
// SPEX_gmp_fscanf
//------------------------------------------------------------------------------

/* Safely scan the stream fp. Return positive value (the number of fields
 * successfully parsed and stored), otherwise return negative value (error
 * code) */

SPEX_info SPEX_gmp_fscanf
(
    FILE *fp,
    const char *format,
    ...
)
{
    // Start the GMP wrapper
    SPEX_GMP_WRAPPER_START ;

    // call gmp_vfscanf
    va_list args;
    va_start (args, format) ;
    int n = gmp_vfscanf (fp, format, args) ;
    va_end (args) ;

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // If end of input (or a file error) is reached before a character
    // for a field or a literal, and if no previous non-suppressed fields have
    // matched, then the return value is EOF instead of 0
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_asprintf
//------------------------------------------------------------------------------

/* Safely write the output as a null terminated string in a block of memory,
 * which is pointed to by a pointer stored in str. The block of memory must be
 * freed using SPEX_mpfr_free_str. The return value is the number of characters
 * written in the string, excluding the null-terminator, or a negative value if
 * an error occurred */

SPEX_info SPEX_mpfr_asprintf (char **str, const char *format, ... )
{
    // Start the GMP wrapper
    SPEX_GMP_WRAPPER_START ;

    // call mpfr_vasprintf
    va_list args;
    va_start (args, format) ;
    int n = mpfr_vasprintf (str, format, args) ;
    va_end (args) ;

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // mpfr_vasprintf returns a negative value if an error occurred
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_free_str
//------------------------------------------------------------------------------

/* Safely free a string allocated by SPEX_mpfr_asprintf. */
/* DONT TRY TO FREE NULL PONITER USING THIS FUNCTION*/

SPEX_info SPEX_mpfr_free_str (char *str)
{
    // Start the GMP wrapper
    SPEX_GMP_WRAPPER_START ;

    // call mpfr_free_str
    mpfr_free_str (str) ;

    // Finish the wrapper and return 0 if successful
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_fprintf
//------------------------------------------------------------------------------

/* Safely print to the stream fp. Return positive value (the number of
 * characters written) upon success, otherwise return negative value (error
 * code) */

#if 0
/* This function is currently unused, but kept here for future reference. */

SPEX_info SPEX_mpfr_fprintf
(
    FILE *fp,
    const char *format,
    ...
)
{
    // Start the GMP wrapper
    SPEX_GMP_WRAPPER_START ;

    // call mpfr_vfprintf
    va_list args;
    va_start (args, format) ;
    int n = mpfr_vfprintf (fp, format, args) ;
    va_end (args) ;
    // Free cache from mpfr_vfprintf. Even though mpfr_free_cache is
    // called in SPEX_finalize ( ), it has to be called here to
    // prevent memory leak in some rare situations.
    mpfr_free_cache ( ) ;

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // mpfr_vfprintf returns -1 if an error occurred.
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK) ;
}
#endif

//------------------------------------------------------------------------------
// SPEX_mpfr_printf
//------------------------------------------------------------------------------

/* Safely print to the standard output stdout. Return positive value (the number
 * of characters written) upon success, otherwise return negative value (error
 * code) */

#if 0
/* This function is currently unused, but kept here for future reference. */
SPEX_info SPEX_mpfr_printf
(
    const char *format,
    ...
)
{
    // Start the GMP wrapper
    SPEX_GMP_WRAPPER_START ;

    // call mpfr_vprintf
    va_list args;
    va_start (args, format) ;
    int n = mpfr_vprintf (format, args) ; 
    va_end (args) ;
    // Free cache from mpfr_vprintf. Even though mpfr_free_cache is
    // called in SPEX_finalize ( ), it has to be called here to
    // prevent memory leak in some rare situations.
    mpfr_free_cache ( ) ;

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // mpfr_vprintf returns -1 if an error occurred.
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK) ;
}
#endif
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------------Integer (mpz_t type) functions-----------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SPEX_mpz_init
//------------------------------------------------------------------------------

/* Purpose: Safely initialize an mpz_t number */

SPEX_info SPEX_mpz_init
(
    mpz_t x
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    mpz_init (x) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_init2
//------------------------------------------------------------------------------

/* Purpose: Safely initialize an mpz_t number with space for size bits */

SPEX_info SPEX_mpz_init2
(
    mpz_t x,                // Number to be initialized
    const size_t size       // size of the number
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    mpz_init2 (x, (mp_bitcnt_t) size) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_set
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpz number = to an mpz number, i.e., x = y */

SPEX_info SPEX_mpz_set
(
    mpz_t x,
    const mpz_t y
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    mpz_set (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_set_ui
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpz number = to uint64_t, i.e., x = y */

SPEX_info SPEX_mpz_set_ui
(
    mpz_t x,
    const uint64_t y
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    mpz_set_ui (x, (unsigned long int) y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_set_si
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpz number = a signed int64_t */

SPEX_info SPEX_mpz_set_si
(
    mpz_t x,
    const int64_t y
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    mpz_set_si (x, (signed long int) y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}
#if 0
/* This function is currently unused, but kept here for future reference. */
//------------------------------------------------------------------------------
// SPEX_mpz_set_d
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpz number = a double */

SPEX_info SPEX_mpz_set_d
(
    mpz_t x,
    const double y
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    mpz_set_d (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}
#endif

//------------------------------------------------------------------------------
// SPEX_mpz_get_d
//------------------------------------------------------------------------------

/* Purpose: Safely set a double number = a mpz */

SPEX_info SPEX_mpz_get_d
(
    double *x,
    const mpz_t y
)
{
    SPEX_GMP_WRAPPER_START ;
    *x = mpz_get_d (y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_get_si
//------------------------------------------------------------------------------

/* Purpose: Safely set an int64_t = a mpz */

SPEX_info SPEX_mpz_get_si
(
    int64_t *x,
    const mpz_t y
)
{
    SPEX_GMP_WRAPPER_START ;
    *x = mpz_get_si (y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_set_q
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpz number = mpq number */

SPEX_info SPEX_mpz_set_q
(
    mpz_t x,
    const mpq_t y
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    mpz_set_q (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_mul
//------------------------------------------------------------------------------

/* Purpose: Safely compute a = b*c */

SPEX_info SPEX_mpz_mul
(
    mpz_t a,
    const mpz_t b,
    const mpz_t c
)
{
    SPEX_GMPZ_WRAPPER_START (a) ;
    mpz_mul (a, b, c) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

#if 0
//------------------------------------------------------------------------------
// SPEX_mpz_add
//------------------------------------------------------------------------------

/* Purpose: Safely compute a = b+c */

SPEX_info SPEX_mpz_add
(
    mpz_t a,
    const mpz_t b,
    const mpz_t c
)
{
    SPEX_GMPZ_WRAPPER_START (a) ;
    mpz_add (a,b,c) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_addmul
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpz number += product of two mpz numbers,
 * i.e., x = x + y*z */

SPEX_info SPEX_mpz_addmul
(
    mpz_t x,
    const mpz_t y,
    const mpz_t z
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    mpz_addmul (x, y, z) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

#endif

//------------------------------------------------------------------------------
// SPEX_mpz_submul
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpz number = itself minus a product of
 * mpz numbers, i.e., x = x - y*z
 */

SPEX_info SPEX_mpz_submul
(
    mpz_t x,
    const mpz_t y,
    const mpz_t z
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    mpz_submul (x, y, z) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_divexact
//------------------------------------------------------------------------------

/* Purpose: Safe version of exact integer division, i.e., x = y / z */

SPEX_info SPEX_mpz_divexact
(
    mpz_t x,
    const mpz_t y,
    const mpz_t z
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    if (mpz_sgn(z) == 0)
    {
        SPEX_GMP_WRAPPER_FINISH;
        return SPEX_PANIC;
    }
    mpz_divexact (x, y, z) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_gcd
//------------------------------------------------------------------------------

/* Purpose: Safely compute the gcd of two mpz_t numbers, i.e., x = gcd (y, z) */

SPEX_info SPEX_mpz_gcd
(
    mpz_t x,
    const mpz_t y,
    const mpz_t z
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    mpz_gcd (x, y, z) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_lcm
//------------------------------------------------------------------------------

/* Purpose: Safely compute the lcm of two mpz numbers */

SPEX_info SPEX_mpz_lcm
(
    mpz_t lcm,   // lcm of x and y
    const mpz_t x,
    const mpz_t y
)
{
    SPEX_GMPZ_WRAPPER_START (lcm) ;
    mpz_lcm (lcm, x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_abs
//------------------------------------------------------------------------------

/* Purpose: Safely set x = |y| */

SPEX_info SPEX_mpz_abs
(
    mpz_t x,
    const mpz_t y
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    mpz_abs (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_cmp
//------------------------------------------------------------------------------

/* Purpose: Safely compare two mpz numbers,
 * r > 0 if x > y, r = 0 if x = y, and r < 0 if x < y */

SPEX_info SPEX_mpz_cmp
(
    int *r,
    const mpz_t x,
    const mpz_t y
)
{
    SPEX_GMP_WRAPPER_START ;
    *r = mpz_cmp (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_cmpabs
//------------------------------------------------------------------------------

/* Purpose: Safely compare the absolute value of two mpz numbers,
 * r > 0 if |x| > |y|, r = 0 if |x| = |y|, and r < 0 if |x| < |y| */

SPEX_info SPEX_mpz_cmpabs
(
    int *r,
    const mpz_t x,
    const mpz_t y
)
{
    SPEX_GMP_WRAPPER_START ;
    *r = mpz_cmpabs (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_cmp_ui
//------------------------------------------------------------------------------

/* Purpose: Safely compare a mpz number with a uint64_t integer
 * r > 0 if x > y, r = 0 if x = y, and r < 0 if x < y */
SPEX_info SPEX_mpz_cmp_ui
(
    int *r,
    const mpz_t x,
    const uint64_t y
)
{
    SPEX_GMP_WRAPPER_START ;
    *r = mpz_cmp_ui (x, (unsigned long int) y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_sgn
//------------------------------------------------------------------------------

/* Purpose: Safely set sgn = 0 if x = 0, otherwise, sgn = x/|x| */

SPEX_info SPEX_mpz_sgn
(
    int *sgn,
    const mpz_t x
)
{
    SPEX_GMP_WRAPPER_START ;
    *sgn = mpz_sgn (x) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpz_sizeinbase
//------------------------------------------------------------------------------

/* Purpose: Safely return the size of x measured in number of digits
 * in the given base */
SPEX_info SPEX_mpz_sizeinbase
(
    size_t *size,
    const mpz_t x,
    int64_t base
)
{
    SPEX_GMP_WRAPPER_START ;
    *size = mpz_sizeinbase (x, (int) base) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------------Rational  (mpq type) functions------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SPEX_mpq_init
//------------------------------------------------------------------------------

/* Purpose: Safely initialize an mpq_t number */

SPEX_info SPEX_mpq_init
(
    mpq_t x
)
{
    SPEX_GMPQ_WRAPPER_START (x) ;
    mpq_init (x) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_set
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpq number = to an mpq number, i.e., x = y */

SPEX_info SPEX_mpq_set
(
    mpq_t x,
    const mpq_t y
)
{
    SPEX_GMPQ_WRAPPER_START (x) ;
    mpq_set (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_set_z
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpq number = an mpz number. i.e., x = y */

SPEX_info SPEX_mpq_set_z
(
    mpq_t x,
    const mpz_t y
)
{
    SPEX_GMPQ_WRAPPER_START (x) ;
    mpq_set_z (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_set_d
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpq number = a double */

SPEX_info SPEX_mpq_set_d
(
    mpq_t x,
    const double y
)
{
    SPEX_GMPQ_WRAPPER_START (x) ;
    mpq_set_d (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_set_ui
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpq number as the fraction of two
 * unsigned ints. i.e., x = y / z
 */

SPEX_info SPEX_mpq_set_ui
(
    mpq_t x,
    const uint64_t y,
    const uint64_t z
)
{
    SPEX_GMPQ_WRAPPER_START (x) ;
    mpq_set_ui (x, (unsigned long int) y, (unsigned long int) z) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_set_si
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpq number = an int64_t */

SPEX_info SPEX_mpq_set_si
(
    mpq_t x,
    const int64_t y,
    const uint64_t z
)
{
    SPEX_GMPQ_WRAPPER_START (x) ;
    mpq_set_si (x, (signed long int) y, (unsigned long int) z) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_set_num
//------------------------------------------------------------------------------

/* Purpose: Safely set the numerator of an mpq number */

SPEX_info SPEX_mpq_set_num
(
    mpq_t x,
    const mpz_t y
)
{
    SPEX_GMPQ_WRAPPER_START (x) ;
    mpq_set_num (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_set_den
//------------------------------------------------------------------------------

/* Purpose: Safely set the denominator of an mpq number */

SPEX_info SPEX_mpq_set_den
(
    mpq_t x,
    const mpz_t y
)
{
    SPEX_GMPQ_WRAPPER_START (x) ;
    mpq_set_den (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_get_den
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpz number = denominator of an mpq number */

SPEX_info SPEX_mpq_get_den
(
    mpz_t x,
    const mpq_t y
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    mpq_get_den (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_get_d
//------------------------------------------------------------------------------

/* Purpose: Safely set a double = a mpq number*/

SPEX_info SPEX_mpq_get_d
(
    double *x,
    const mpq_t y
)
{
    SPEX_GMP_WRAPPER_START ;
    *x = mpq_get_d (y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_abs
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpq number = absolute value of mpq */

SPEX_info SPEX_mpq_abs
(
    mpq_t x,
    const mpq_t y
)
{
    SPEX_GMPQ_WRAPPER_START (x) ;
    mpq_abs (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_add
//------------------------------------------------------------------------------

/* Purpose: Safely add two mpq numbers, i.e., x = y+z */

SPEX_info SPEX_mpq_add
(
    mpq_t x,
    const mpq_t y,
    const mpq_t z
)
{
    SPEX_GMPQ_WRAPPER_START (x) ;
    mpq_add (x, y, z) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_mul
//------------------------------------------------------------------------------

/* Purpose: Safely multiply two mpq numbers, i.e., x = y*z */
SPEX_info SPEX_mpq_mul
(
    mpq_t x,
    const mpq_t y,
    const mpq_t z
)
{
    SPEX_GMPQ_WRAPPER_START (x) ;
    mpq_mul (x, y, z) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_div
//------------------------------------------------------------------------------

/* Purpose: Safely divide two mpq numbers, i.e., x = y/z */

SPEX_info SPEX_mpq_div
(
    mpq_t x,
    const mpq_t y,
    const mpq_t z
)
{
    SPEX_GMPQ_WRAPPER_START (x) ;
    mpq_div (x, y, z) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_cmp
//------------------------------------------------------------------------------

/* Purpose: Safely compare two mpq numbers,
 * r > 0 if x > y, r = 0 if x = y, and r < 0 if x < y */

SPEX_info SPEX_mpq_cmp
(
    int *r,
    const mpq_t x,
    const mpq_t y
)
{
    SPEX_GMP_WRAPPER_START ;
    *r = mpq_cmp (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_cmp_ui
//------------------------------------------------------------------------------

/* Purpose: Safely compare x and num/den. r > 0 if x > num/den,
 * r = 0 if x = num/den, and r < 0 if x < num/den */

SPEX_info SPEX_mpq_cmp_ui
(
    int *r,
    const mpq_t x,
    const uint64_t num,
    const uint64_t den
)
{
    SPEX_GMP_WRAPPER_START ;
    *r = mpq_cmp_ui (x, (unsigned long int) num, (unsigned long int) den) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_equal
//------------------------------------------------------------------------------

/* Purpose: Safely check if two mpq numbers equal,
 * r = 0 (r = false) if x != y, r != 0 (r = true) if x = y */

SPEX_info SPEX_mpq_equal
(
    int *r,
    const mpq_t x,
    const mpq_t y
)
{
    SPEX_GMP_WRAPPER_START ;
    *r = mpq_equal (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpq_sgn
//------------------------------------------------------------------------------

/* Purpose: Safely set sgn = 0 if x = 0, otherwise, sgn = x/|x| */

SPEX_info SPEX_mpq_sgn
(
    int *sgn,
    const mpq_t x
)
{
    SPEX_GMP_WRAPPER_START ;
    *sgn = mpq_sgn (x) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//-------------------------Floating Point (mpfr type) functions-----------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SPEX_mpfr_init2
//------------------------------------------------------------------------------

/* Purpose: Safely initialize an mpfr_t number */

SPEX_info SPEX_mpfr_init2
(
    mpfr_t x,       // Floating point number to initialize
    uint64_t size    // # of bits in x
)
{
    SPEX_GMPFR_WRAPPER_START (x) ;
    mpfr_init2 (x, (unsigned long int) size) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_set
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpfr number = to an mpfr number, i.e., x = y */

SPEX_info SPEX_mpfr_set
(
    mpfr_t x,
    const mpfr_t y,
    const mpfr_rnd_t rnd
)
{
    SPEX_GMPFR_WRAPPER_START (x) ;
    mpfr_set (x, y, rnd) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_set_d
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpfr number = to a double, i.e., x = y */

SPEX_info SPEX_mpfr_set_d
(
    mpfr_t x,
    const double y,
    const mpfr_rnd_t rnd  // MPFR rounding scheme used
)
{
    SPEX_GMPFR_WRAPPER_START (x) ;
    mpfr_set_d (x, y, rnd) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}


//------------------------------------------------------------------------------
// SPEX_mpfr_set_si
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpfr number = to a signed int, i.e., x = y */

SPEX_info SPEX_mpfr_set_si
(
    mpfr_t x,
    int64_t y,
    const mpfr_rnd_t rnd  // MPFR rounding scheme used
)
{
    SPEX_GMPFR_WRAPPER_START (x) ;
    mpfr_set_si (x, y, rnd) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_set_q
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpfr number = to an mpq number */

SPEX_info SPEX_mpfr_set_q
(
    mpfr_t x,
    const mpq_t y,
    const mpfr_rnd_t rnd
)
{
    SPEX_GMPFR_WRAPPER_START (x) ;
    mpfr_set_q (x, y, rnd) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_set_z
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpfr number = to an mpz number */

SPEX_info SPEX_mpfr_set_z
(
    mpfr_t x,
    const mpz_t y,
    const mpfr_rnd_t rnd
)
{
    SPEX_GMPFR_WRAPPER_START (x) ;
    mpfr_set_z (x, y, rnd) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_get_z
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpz number = to an mpfr number, i.e., x = y */

SPEX_info SPEX_mpfr_get_z
(
    mpz_t x,
    const mpfr_t y,
    const mpfr_rnd_t rnd  // MPFR rounding scheme used
)
{
    SPEX_GMPZ_WRAPPER_START (x) ;
    mpfr_get_z (x, y, rnd) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_get_q
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpq number = to an mpfr number, i.e., x = y */

SPEX_info SPEX_mpfr_get_q
(
    mpq_t x,
    const mpfr_t y,
    const mpfr_rnd_t rnd  // MPFR rounding scheme used
)
{
    SPEX_GMPQ_WRAPPER_START (x) ;
    mpfr_get_q (x, y) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_get_d
//------------------------------------------------------------------------------

/* Purpose: Safely set a double = to a mpfr number, i.e., x = y */

SPEX_info SPEX_mpfr_get_d
(
    double *x,
    const mpfr_t y,
    const mpfr_rnd_t rnd  // MPFR rounding scheme used
)
{
    SPEX_GMP_WRAPPER_START ;
    *x = mpfr_get_d (y, rnd) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_get_si
//------------------------------------------------------------------------------

/* Purpose: Safely set a signed int = to a mpfr number, i.e., x = y */

SPEX_info SPEX_mpfr_get_si
(
    int64_t *x,
    const mpfr_t y,
    const mpfr_rnd_t rnd  // MPFR rounding scheme used
)
{
    SPEX_GMP_WRAPPER_START ;
    *x = mpfr_get_si (y, rnd) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_mul
//------------------------------------------------------------------------------

/* Purpose: Safely multiply mpfr numbers, x = y*z */

SPEX_info SPEX_mpfr_mul
(
    mpfr_t x,
    const mpfr_t y,
    const mpfr_t z,
    const mpfr_rnd_t rnd  // MPFR rounding mode
)
{
    SPEX_GMPFR_WRAPPER_START (x) ;
    mpfr_mul (x, y, z, rnd) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_mul_d
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpfr number = to a product of an mpfr_t and double,
 * i.e., x = y*z
 */

SPEX_info SPEX_mpfr_mul_d
(
    mpfr_t x,
    const mpfr_t y,
    const double z,
    const mpfr_rnd_t rnd  // MPFR rounding scheme used
)
{
    SPEX_GMPFR_WRAPPER_START (x) ;
    mpfr_mul_d (x, y, z, rnd) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_div_d
//------------------------------------------------------------------------------

/* Purpose: Safely set a mpfr number = a mpfr number divided by a double,
 * i.e., x = y/z
 */

SPEX_info SPEX_mpfr_div_d
(
    mpfr_t x,
    const mpfr_t y,
    const double z,
    const mpfr_rnd_t rnd  // MPFR rounding scheme used
)
{
    SPEX_GMPFR_WRAPPER_START (x) ;
    mpfr_div_d (x, y, z, rnd) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_ui_pow_ui
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpfr number = power of two ints, i.e.,
 * x = y^z
 */

SPEX_info SPEX_mpfr_ui_pow_ui
(
    mpfr_t x,
    const uint64_t y,
    const uint64_t z,
    const mpfr_rnd_t rnd  // MPFR rounding mode
)
{
    SPEX_GMPFR_WRAPPER_START (x) ;
    mpfr_ui_pow_ui (x, (unsigned long int) y, (unsigned long int) z, rnd) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_log2
//------------------------------------------------------------------------------

/* Purpose: Safely take the log2 of an mpfr number */

#if 0
/* This function is currently unused, but kept here for future reference. */

SPEX_info SPEX_mpfr_log2
(
    mpfr_t x,
    const mpfr_t y,
    const mpfr_rnd_t rnd
)
{
    SPEX_GMPFR_WRAPPER_START (x) ;
    mpfr_log2 (x, y, rnd) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

#endif

//------------------------------------------------------------------------------
// SPEX_mpfr_sgn
//------------------------------------------------------------------------------

/* Purpose: Safely set sgn = 0 if x = 0, otherwise, sgn = x/|x| */

SPEX_info SPEX_mpfr_sgn
(
    int *sgn,
    const mpfr_t x
)
{
    SPEX_GMP_WRAPPER_START ;
    *sgn = mpfr_sgn (x) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

//------------------------------------------------------------------------------
// SPEX_mpfr_free_cache
//------------------------------------------------------------------------------

/* Purpose: Safely free all caches and pools used by MPFR internally */

SPEX_info SPEX_mpfr_free_cache ( void )
{
    SPEX_GMP_WRAPPER_START ;
    mpfr_free_cache ( ) ;
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK) ;
}

