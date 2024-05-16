//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_gmp.c: interface to the gmp library
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
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
        gmpfunc (args);
        SPEX_GMP_WRAPPER_FINISH ;
        return (SPEX_OK);
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
    SPEX_info SPEX_gmpfunc (int *result, args)
    {
        SPEX_GMP_WRAPPER_START ;
        (*result) = gmpfunc (args);
        SPEX_GMP_WRAPPER_FINISH ;
        return (SPEX_OK);
    }
*/

// The SPEX_GMP*_WRAPPER_START macros also take one or two 'archive' parameters,
// for the current mpz, mpq, or mpfr object being operated on.  A pointer
// parameter to this parameter is kept so that it can be safely freed in case
// a memory error occurs (avoiding a double-free), in spex_gmp_safe_free.
// See the examples below.

#include "spex_util_internal.h"

// ignore warnings about unused parameters in this file
#if defined (__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

//------------------------------------------------------------------------------
// thread-local-storage
//------------------------------------------------------------------------------

// SPEX is thread-safe as long as all of the following conditions hold:
//
// (1) GMP and MPFR are both thread-safe.  This is the typical case, but it is
//      possible to compile GMP and MPFR with thread-safety disabled.  See:
//      https://gmplib.org/manual/Reentrancy
//      https://www.mpfr.org/mpfr-3.1.0/
//
// (2) only one user thread calls SPEX_initialize and SPEX_finalize.
//
// (3) each subsequent user thread must call SPEX_thread_initialize when it
//      starts, and SPEX_thread_finalize when it finishes.
//
// (4) Multiple user threads may not write to the same SPEX objects.  If
//      declared as an input-only variable, multiple user threads may access
//      them in parallel.
//
// (5) SPEX is compiled with either OpenMP, or a compiler that supports
//      thread-local-storage (most of them do).

#if defined ( _OPENMP )

    // OpenMP threadprivate is preferred
    #include <omp.h>
    spex_gmp_t *spex_gmp = NULL ;
    #pragma omp threadprivate (spex_gmp)

#elif defined ( HAVE_KEYWORD__THREAD )

    // gcc and many other compilers support the __thread keyword
    __thread spex_gmp_t *spex_gmp = NULL ;

#elif defined ( HAVE_KEYWORD__DECLSPEC_THREAD )

    // Windows: __declspec (thread)
    __declspec ( thread ) spex_gmp_t *spex_gmp = NULL ;

#elif defined ( HAVE_KEYWORD__THREAD_LOCAL )

    // ANSI C11 threads
    #include <threads.h>
    _Thread_local spex_gmp_t *spex_gmp = NULL ;

#else

    // SPEX will not be thread-safe.
    spex_gmp_t *spex_gmp = NULL ;
    #ifndef MATLAB_MEX_FILE
    #warning "SPEX not compiled with OpenMP or thread keyword; SPEX will not be thread-safe!"
    #endif

#endif

//------------------------------------------------------------------------------
// GMP/MPFR wrapper macros
//------------------------------------------------------------------------------

#define SPEX_GMP_WRAPPER_START_HELPER(z1,z2,q,fr)                       \
    /* spex_gmp_t *spex_gmp = spex_gmp_get ( ) ; */                     \
    if (spex_gmp == NULL) return (SPEX_OUT_OF_MEMORY);                  \
    spex_gmp->mpz_archive  = z1 ;                                       \
    spex_gmp->mpz_archive2 = z2 ;                                       \
    spex_gmp->mpq_archive  = q  ;                                       \
    spex_gmp->mpfr_archive = fr ;                                       \
    /* setjmp returns 0 if called from here, or > 0 if from longjmp */  \
    int status = setjmp (spex_gmp->environment) ;                       \
    if (status != 0)                                                    \
    {                                                                   \
        /* failure from longjmp */                                      \
        return (spex_gmp_failure (status)) ;                            \
    }

#define SPEX_GMP_WRAPPER_START                                          \
    SPEX_GMP_WRAPPER_START_HELPER (NULL, NULL, NULL, NULL) ;

#define SPEX_GMPZ_WRAPPER_START(z1)                                     \
    SPEX_GMP_WRAPPER_START_HELPER (z1, NULL, NULL, NULL) ;

#define SPEX_GMPZ_WRAPPER_START2(z1,z2)                                 \
    SPEX_GMP_WRAPPER_START_HELPER (z1, z2, NULL, NULL) ;

#define SPEX_GMPQ_WRAPPER_START(q)                                      \
    SPEX_GMP_WRAPPER_START_HELPER (NULL, NULL, q, NULL) ;

#define SPEX_GMPFR_WRAPPER_START(fr)                                    \
    SPEX_GMP_WRAPPER_START_HELPER (NULL, NULL, NULL, fr) ;

#define SPEX_GMP_WRAPPER_FINISH                                         \
    spex_gmp->nmalloc = 0 ;                                             \
    spex_gmp->mpz_archive  = NULL ;                                     \
    spex_gmp->mpz_archive2 = NULL ;                                     \
    spex_gmp->mpq_archive  = NULL ;                                     \
    spex_gmp->mpfr_archive = NULL ;

//------------------------------------------------------------------------------
// spex_gmp_initialize: initialize the SPEX GMP interface
//------------------------------------------------------------------------------

// Called by SPEX_initialize* with primary == 1, and by SPEX_thread_initialize
// with primary == 0.  The object is not allocated if it already exists.

SPEX_info spex_gmp_initialize (int primary)
{
    if (spex_gmp == NULL)
    {
        // allocate the spex_gmp object
        spex_gmp = SPEX_calloc (1, sizeof (spex_gmp_t));
        if (spex_gmp == NULL)
        {
            // out of memory
            return (SPEX_OUT_OF_MEMORY);
        }

        // allocate an empty spex_gmp->list
        spex_gmp->list = (void **) SPEX_calloc (SPEX_GMP_LIST_INIT,
            sizeof (void *));

        if (spex_gmp->list == NULL)
        {
            // out of memory
            SPEX_FREE (spex_gmp);
            return (SPEX_OUT_OF_MEMORY);
        }

        // initialize the spex_gmp
        spex_gmp->nlist = SPEX_GMP_LIST_INIT ;
        spex_gmp->nmalloc = 0 ;
        spex_gmp->mpz_archive  = NULL ;
        spex_gmp->mpz_archive2 = NULL ;
        spex_gmp->mpq_archive  = NULL ;
        spex_gmp->mpfr_archive = NULL ;
        spex_gmp->primary = primary ;
    }
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// spex_gmp_finalize: finalize the SPEX GMP interface
//------------------------------------------------------------------------------

// called by SPEX_finalize* with primary == 1, and by SPEX_thread_finalize with
// primary == 0.

void spex_gmp_finalize (int primary)
{
    // free the spex_gmp object for this thread, if it exists.  If this function
    // is called by SPEX_finalize, then primary == 1 on input, and the spex_gmp
    // object is always freed.  If primary == 0 on input, then the caller is
    // SPEX_thread_finalize, and in this case the spex_gmp object is freed only
    // if spex_gmp->primary is also zero.

    if (spex_gmp != NULL && primary >= spex_gmp->primary)
    {
        // free the spex_gmp->list, if it exists
        SPEX_FREE (spex_gmp->list) ;
        // free the spex_gmp object itself
        SPEX_FREE (spex_gmp);
    }
}

//------------------------------------------------------------------------------
// spex_gmp_get: get the thread-local spex_gmp object and initialize it
//------------------------------------------------------------------------------

spex_gmp_t *spex_gmp_get (void)
{
    if (spex_gmp != NULL)
    {
        // clear the list of allocated objects in the spex_gmp->list
        spex_gmp->nmalloc = 0 ;
    }

    // return the spex_gmp object for this thread (or NULL if none)
    return (spex_gmp);
}

//------------------------------------------------------------------------------
// spex_gmp_ntrials: pretend to fail, for test coverage only
//------------------------------------------------------------------------------

#ifdef SPEX_GMP_TEST_COVERAGE
static int64_t spex_gmp_ntrials = -1 ;     // for test coverage only

void spex_set_gmp_ntrials (int64_t ntrials)
{
    spex_gmp_ntrials = ntrials ;
}

int64_t spex_get_gmp_ntrials (void)
{
    return (spex_gmp_ntrials) ;
}
#endif

//------------------------------------------------------------------------------
// spex_gmp_allocate: malloc space for gmp
//------------------------------------------------------------------------------

/* Purpose: malloc space for gmp. A NULL pointer is never returned to the GMP
 * library. If the allocation fails, all memory allocated since the
 * SPEX_GMP*_WRAPPER_START is freed and an error is thrown to the GMP wrapper
 * via longjmp
 */

void *spex_gmp_allocate
(
    size_t size // Amount of memory to be allocated
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // if spex_gmp does not exist, memory cannot be allocarted
    if (spex_gmp == NULL) return (NULL) ;

    //--------------------------------------------------------------------------
    // for testing only:
    //--------------------------------------------------------------------------

    #ifdef SPEX_GMP_TEST_COVERAGE
    {
        if (spex_gmp_ntrials == 0)
        {
            // pretend to fail
            #ifdef SPEX_GMP_MEMORY_DEBUG
            SPEX_PRINTF ("spex_gmp_allocate pretends to fail\n");
            #endif
            longjmp (spex_gmp->environment, 1);
        }
        else if (spex_gmp_ntrials > 0)
        {
            // one more malloc has been used up
            spex_gmp_ntrials-- ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // ensure the spex_gmp->list is large enough
    //--------------------------------------------------------------------------

    if (spex_gmp->nmalloc == spex_gmp->nlist)
    {
        // double the size of the spex_gmp->list
        bool ok ;
        int64_t newsize = 2 * spex_gmp->nlist ;
        spex_gmp->list = (void **)
            SPEX_realloc (newsize, spex_gmp->nlist, sizeof (void *),
            spex_gmp->list, &ok);
        if (!ok)
        {
            // failure to double the size of the spex_gmp->list.
            // The existing spex_gmp->list is still valid, with the old size,
            // (spex_gmp->nlist).  This is required so that the error handler
            // can traverse the spex_gmp->list to free all objects there.
            longjmp (spex_gmp->environment, 3);
        }
        // success:  the old spex_gmp->list has been freed, and replaced with
        // the larger newlist.
        spex_gmp->nlist = newsize ;
    }

    //--------------------------------------------------------------------------
    // malloc the block
    //--------------------------------------------------------------------------

    #ifdef SPEX_GMP_MEMORY_DEBUG
    SPEX_PRINTF ("spex_gmp_malloc (%g): ", (double) size);
    #endif

    void *p = SPEX_malloc (size);

    if (p == NULL)
    {
        // failure to allocate the new block
        longjmp (spex_gmp->environment, 4);
    }

    //--------------------------------------------------------------------------
    // save p in the spex_gmp->list and return result to GMP
    //--------------------------------------------------------------------------

    spex_gmp->list [spex_gmp->nmalloc++] = p ;

    #ifdef SPEX_GMP_MEMORY_DEBUG
    SPEX_PRINTF (" %p\n", p);
    spex_gmp_dump ( );
    #endif

    // return p to SPEX_gmp_function (NEVER return a NULL pointer to GMP!)
    ASSERT (p != NULL);
    return (p);
}

//------------------------------------------------------------------------------
// spex_gmp_safe_free:  free a block of memory and remove it from the archive
//------------------------------------------------------------------------------

// see mpfr-4.2.1/src/mpfr-impl.h, for MPFR_GET_REAL_PTR
#define SPEX_MPFR_GET_REAL_PTR(x) ((x)->_mpfr_d - 1)

static inline void spex_gmp_safe_free (void *p)
{
    if (spex_gmp != NULL)
    {
        if (spex_gmp->mpz_archive != NULL)
        {
            if (p == SPEX_MPZ_PTR((spex_gmp->mpz_archive)))
            {
                SPEX_MPZ_PTR((spex_gmp->mpz_archive)) = NULL ;
            }
        }
        if (spex_gmp->mpz_archive2 != NULL)
        {
            if (p == SPEX_MPZ_PTR((spex_gmp->mpz_archive2)))
            {
                SPEX_MPZ_PTR((spex_gmp->mpz_archive2)) = NULL ;
            }
        }
        if (spex_gmp->mpq_archive != NULL)
        {
            if (p == SPEX_MPZ_PTR(SPEX_MPQ_NUM(spex_gmp->mpq_archive)))
            {
                SPEX_MPZ_PTR(SPEX_MPQ_NUM(spex_gmp->mpq_archive)) = NULL ;
            }
            if (p == SPEX_MPZ_PTR(SPEX_MPQ_DEN(spex_gmp->mpq_archive)))
            {
                SPEX_MPZ_PTR(SPEX_MPQ_DEN(spex_gmp->mpq_archive)) = NULL ;
            }
        }
        if (spex_gmp->mpfr_archive != NULL)
        {
            if (p == SPEX_MPFR_GET_REAL_PTR(spex_gmp->mpfr_archive))
            {
                SPEX_MPFR_MANT(spex_gmp->mpfr_archive) = NULL ;
            }
        }
    }
    SPEX_FREE (p) ;
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

    //--------------------------------------------------------------------------
    // quick return if p is NULL
    //--------------------------------------------------------------------------

    if (p == NULL)
    {
        return ;
    }

    //--------------------------------------------------------------------------
    // remove the block from the spex_gmp->list
    //--------------------------------------------------------------------------

    if (spex_gmp != NULL)
    {

        #ifdef SPEX_GMP_MEMORY_DEBUG
        SPEX_PRINTF ("\n=================== free %p\n", p);
        spex_gmp_dump ( );
        #endif

        if (spex_gmp->list != NULL)
        {
            // remove p from the spex_gmp->list
            for (int64_t i = 0 ; i < spex_gmp->nmalloc ; i++)
            {
                if (spex_gmp->list [i] == p)
                {
                    #ifdef SPEX_GMP_MEMORY_DEBUG
                    SPEX_PRINTF ("    found at i = %d\n", i);
                    #endif
                    spex_gmp->list [i] = spex_gmp->list [--spex_gmp->nmalloc] ;
                    break ;
                }
            }
        }

        #ifdef SPEX_GMP_MEMORY_DEBUG
        spex_gmp_dump ( );
        #endif
    }

    //--------------------------------------------------------------------------
    // free the block
    //--------------------------------------------------------------------------

    // free p, even if it is not found in the spex_gmp->list.  p is only in the
    // spex_gmp->list if it was allocated inside the current GMP function.
    // If the block was allocated by one GMP function and freed by another,
    // it is not in the list.
    spex_gmp_safe_free (p);
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

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // if spex_gmp does not exist, memory cannot be allocated
    if (spex_gmp == NULL) return (NULL) ;

    //--------------------------------------------------------------------------
    // reallocate the space
    //--------------------------------------------------------------------------

    #ifdef SPEX_GMP_MEMORY_DEBUG
    SPEX_PRINTF ("spex_gmp_realloc (%p, %g, %g)\n", p_old,
        (double) old_size, (double) new_size);
    #endif

    if (p_old == NULL)
    {
        // realloc (NULL, size) is the same as malloc (size)
        return (spex_gmp_allocate (new_size));
    }
    else if (new_size == 0)
    {
        // realloc (p, 0) is the same as free (p), and returns NULL
        spex_gmp_free (p_old, old_size);
        return (NULL);
    }
    else
    {
        // change the size of the block
        void *p_new = spex_gmp_allocate (new_size);
        // Note that p_new will never be NULL here, since spex_gmp_allocate
        // does not return if it fails.
        memcpy (p_new, p_old, SPEX_MIN (old_size, new_size));
        spex_gmp_free (p_old, old_size);
        return (p_new);
    }
}

//------------------------------------------------------------------------------
// spex_gmp_dump: debug function
//------------------------------------------------------------------------------

/* Purpose: Dump the list of malloc'd objects */

#ifdef SPEX_GMP_MEMORY_DEBUG
void spex_gmp_dump ( )
{

    //--------------------------------------------------------------------------
    // dump the spex_gmp->list
    //--------------------------------------------------------------------------

    if (spex_gmp == NULL)
    {
        SPEX_PRINTF ("spex_gmp is NULL\n") ;
        return ;
    }

    SPEX_PRINTF ("nmalloc = %g, spex_gmp->nlist = %g\n",
        (double) spex_gmp->nmalloc, (double) spex_gmp->nlist);
    if (spex_gmp->list != NULL)
    {
        for (int64_t i = 0 ; i < spex_gmp->nmalloc ; i++)
        {
            SPEX_PRINTF ("    spex_gmp->list [%d] = %p\n", i,
                spex_gmp->list [i]);
        }
    }

    SPEX_PRINTF ("   spex_gmp->mpz_archive  : %p\n", spex_gmp->mpz_archive);
    SPEX_PRINTF ("   spex_gmp->mpz_archive2 : %p\n", spex_gmp->mpz_archive2);
    SPEX_PRINTF ("   spex_gmp->mpq_archive  : %p\n", spex_gmp->mpq_archive);
    SPEX_PRINTF ("   spex_gmp->mpfr_archive : %p\n", spex_gmp->mpfr_archive);
}
#endif

//------------------------------------------------------------------------------
// spex_gmp_failure: catch an error
//------------------------------------------------------------------------------

/* Purpose: Catch an error from longjmp */

SPEX_info spex_gmp_failure
(
    int status      // Status returned from longjmp
                    // (unused parameter unless debugging)
)
{

    //--------------------------------------------------------------------------
    // get the spex_gmp object for this thread
    //--------------------------------------------------------------------------

    #ifdef SPEX_GMP_MEMORY_DEBUG
    SPEX_PRINTF ("failure from longjmp: status: %d\n", status);
    #endif

    //--------------------------------------------------------------------------
    // free all MPFR caches
    //--------------------------------------------------------------------------

    mpfr_free_cache ( );

    //--------------------------------------------------------------------------
    // free the contents of the spex_gmp_t list
    //--------------------------------------------------------------------------

    if (spex_gmp != NULL)
    {
        if (spex_gmp->list != NULL)
        {
            for (int64_t i = 0 ; i < spex_gmp->nmalloc ; i++)
            {
                spex_gmp_safe_free (spex_gmp->list [i]);
                spex_gmp->list [i] = NULL ;
            }
        }
        SPEX_GMP_WRAPPER_FINISH ;
    }

    //--------------------------------------------------------------------------
    // tell the caller that the GMP/MPFR function ran out of memory
    //--------------------------------------------------------------------------

    return (SPEX_OUT_OF_MEMORY);
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
    va_list args ;
    va_start (args, format);
    int n = gmp_vfprintf (fp, format, args);
    va_end (args);

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // gmp_vfprintf returns -1 if an error occurred.
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK);
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
    va_list args ;
    va_start (args, format);
    int n = gmp_vprintf (format, args);
    va_end (args);

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // gmp_vprintf returns -1 if an error occurred.
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK);
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
    va_list args ;
    va_start (args, format);
    int n = gmp_vasprintf (str, format, args);
    va_end (args);

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // gmp_vasprintf returns a negative value if an error occurred
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK);
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
    va_list args ;
    va_start (args, format);
    int n = gmp_vfscanf (fp, format, args);
    va_end (args);

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // If end of input (or a file error) is reached before a character
    // for a field or a literal, and if no previous non-suppressed fields have
    // matched, then the return value is EOF instead of 0
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK);
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
    va_list args ;
    va_start (args, format);
    int n = mpfr_vasprintf (str, format, args);
    va_end (args);

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // mpfr_vasprintf returns a negative value if an error occurred
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpfr_free_str
//------------------------------------------------------------------------------

/* Safely free a string allocated by SPEX_mpfr_asprintf. */

SPEX_info SPEX_mpfr_free_str (char *str)
{
    if (str == NULL) return (SPEX_OK);     // nothing to do

    // Start the GMP wrapper
    SPEX_GMP_WRAPPER_START ;

    // call mpfr_free_str
    mpfr_free_str (str);

    // Finish the wrapper and return 0 if successful
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    va_list args ;
    va_start (args, format);
    int n = mpfr_vfprintf (fp, format, args);
    va_end (args);
    // Free cache from mpfr_vfprintf. Even though mpfr_free_cache is
    // called in SPEX_finalize ( ), it has to be called here to
    // prevent memory leak in some rare situations.
    mpfr_free_cache ( );

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // mpfr_vfprintf returns -1 if an error occurred.
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK);
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
    va_list args ;
    va_start (args, format);
    int n = mpfr_vprintf (format, args);
    va_end (args);
    // Free cache from mpfr_vprintf. Even though mpfr_free_cache is
    // called in SPEX_finalize ( ), it has to be called here to
    // prevent memory leak in some rare situations.
    mpfr_free_cache ( );

    // Finish the wrapper
    SPEX_GMP_WRAPPER_FINISH ;
    // mpfr_vprintf returns -1 if an error occurred.
    return ((n < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK);
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
// NOTE: This function never returns out-of-memory error with GMP-6.2.1 or
//       later versions (since there will be no memory allocation). But it could
//       return such error for GMP-6.1.2 or ealier versions.

SPEX_info SPEX_mpz_init
(
    mpz_t x
)
{
    SPEX_GMPZ_WRAPPER_START (x);
    mpz_init (x);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpz_init2
//------------------------------------------------------------------------------

/* Purpose: Safely initialize an mpz_t number with space for size bits */

SPEX_info SPEX_mpz_init2
(
    mpz_t x,                // Number to be initialized
    const uint64_t size     // size of the number
)
{
    SPEX_GMPZ_WRAPPER_START (x);
    mpz_init2 (x, (mp_bitcnt_t) size);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPZ_WRAPPER_START (x);
    mpz_set (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPZ_WRAPPER_START (x);
    mpz_set_ui (x, (unsigned long int) y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPZ_WRAPPER_START (x);
    mpz_set_si (x, (signed long int) y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}
//------------------------------------------------------------------------------
// SPEX_mpz_set_d
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpz number = a double */
#if 0
/* This function is currently unused, but kept here for future reference. */

SPEX_info SPEX_mpz_set_d
(
    mpz_t x,
    const double y
)
{
    SPEX_GMPZ_WRAPPER_START (x);
    mpz_set_d (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    *x = mpz_get_d (y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    *x = (int64_t) mpz_get_si (y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPZ_WRAPPER_START (a);
    mpz_mul (a, b, c);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}


//------------------------------------------------------------------------------
// SPEX_mpz_addmul
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpz number += product of two mpz numbers,
 * i.e., x = x + y*z */
#if 0
SPEX_info SPEX_mpz_addmul
(
    mpz_t x,
    const mpz_t y,
    const mpz_t z
)
{
    SPEX_GMPZ_WRAPPER_START (x);
    mpz_addmul (x, y, z);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}
#endif

//------------------------------------------------------------------------------
// SPEX_mpz_sub
//------------------------------------------------------------------------------

/* Purpose: Safely compute a = b-c */

SPEX_info SPEX_mpz_sub
(
    mpz_t a,
    const mpz_t b,
    const mpz_t c
)
{
    SPEX_GMPZ_WRAPPER_START (a);
    mpz_sub (a,b,c);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}


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
    SPEX_GMPZ_WRAPPER_START (x);
    mpz_submul (x, y, z);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}


//------------------------------------------------------------------------------
// SPEX_mpz_cdiv_qr
//------------------------------------------------------------------------------

/* Purpose: Safe version of dividing n by d, forming a quotient q and/or
 * remainder r.
 * cdiv rounds q up towards +infinity, and r will have the opposite sign to d.
 * The c stands for “ceil”. That is, q = ceil(n/d)
 */

SPEX_info SPEX_mpz_cdiv_qr
(
    mpz_t q,
    mpz_t r,
    const mpz_t n,
    const mpz_t d
)
{
    SPEX_GMPZ_WRAPPER_START2 (q, r);
    if (mpz_sgn (d) == 0)
    {
        SPEX_GMP_WRAPPER_FINISH ;
        return (SPEX_PANIC);
    }
    mpz_cdiv_qr (q, r, n, d);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPZ_WRAPPER_START (x);
    if (mpz_sgn (z) == 0)
    {
        SPEX_GMP_WRAPPER_FINISH ;
        return (SPEX_PANIC);
    }

#ifdef SPEX_DEBUG
        mpq_t r ;
        mpq_init (r); // r = 0/1
        mpz_fdiv_r (SPEX_MPQ_NUM (r), y, z);
        if (mpz_sgn (SPEX_MPQ_NUM (r)) != 0)
        {
            mpq_set_den (r, z);
            mpq_canonicalize (r);
            gmp_printf ("not exact division! remainder=%Qd\n", r);
            mpq_clear (r);
            SPEX_GMP_WRAPPER_FINISH ;
            return (SPEX_PANIC);
        }
        mpq_clear (r);
#endif

    mpz_divexact (x, y, z);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPZ_WRAPPER_START (x);
    mpz_gcd (x, y, z);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPZ_WRAPPER_START (lcm);
    mpz_lcm (lcm, x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpz_neg
//------------------------------------------------------------------------------

/* Purpose: Safely set x = -y */

SPEX_info SPEX_mpz_neg
(
    mpz_t x,
    const mpz_t y
)
{
    SPEX_GMPZ_WRAPPER_START (x);
    mpz_neg (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPZ_WRAPPER_START (x);
    mpz_abs (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    *r = mpz_cmp (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    *r = mpz_cmpabs (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    *r = mpz_cmp_ui (x, (unsigned long int) y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    *sgn = mpz_sgn (x);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    *size = mpz_sizeinbase (x, (int) base);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPQ_WRAPPER_START (x);
    mpq_init (x);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPQ_WRAPPER_START (x);
    mpq_set (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPQ_WRAPPER_START (x);
    mpq_set_z (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPQ_WRAPPER_START (x);
    mpq_set_d (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPQ_WRAPPER_START (x);
    mpq_set_ui (x, (unsigned long int) y, (unsigned long int) z);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    return (SPEX_OK);
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
    SPEX_GMPQ_WRAPPER_START (x);
    mpq_set_num (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPQ_WRAPPER_START (x);
    mpq_set_den (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpq_get_den
//------------------------------------------------------------------------------

#if 0
/* This function is currently unused, but kept here for future reference. */

/* Purpose: Safely set an mpz number = denominator of an mpq number */

SPEX_info SPEX_mpq_get_den
(
    mpz_t x,
    const mpq_t y
)
{
    SPEX_GMPZ_WRAPPER_START (x);
    mpq_get_den (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}
#endif

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
    *x = mpq_get_d (y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpq_swap
//------------------------------------------------------------------------------

#if 0
/* This function is currently unused, but kept here for future reference. */

/* Purpose: Safely swap the values x and y efficiently */

SPEX_info SPEX_mpq_swap
(
    mpq_t x,
    mpq_t y
)
{
    SPEX_GMP_WRAPPER_START ;
    mpq_swap (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}

#endif

//------------------------------------------------------------------------------
// SPEX_mpq_neg
//------------------------------------------------------------------------------

/* Purpose: Safely set an mpq number x = -y */

SPEX_info SPEX_mpq_neg
(
    mpq_t x,
    const mpq_t y
)
{
    SPEX_GMPQ_WRAPPER_START (x);
    mpq_neg (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPQ_WRAPPER_START (x);
    mpq_abs (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPQ_WRAPPER_START (x);
    mpq_add (x, y, z);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPQ_WRAPPER_START (x);
    mpq_mul (x, y, z);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPQ_WRAPPER_START (x);
    mpq_div (x, y, z);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    *r = mpq_cmp (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    *r = mpq_cmp_ui (x, (unsigned long int) num, (unsigned long int) den);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpq_cmp_z
//------------------------------------------------------------------------------

#if 0
/* This function is currently unused, but kept here for future reference. */

/* Purpose: Safely check if a mpq number equals to a mpz number,
 * r = 0 (r = false) if x != y, r < 0 if x < y, or r > 0 if x > y */

SPEX_info SPEX_mpq_cmp_z
(
    int *r,
    const mpq_t x,
    const mpz_t y
)
{
    SPEX_GMP_WRAPPER_START ;
    *r = mpq_cmp_z (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}

#endif

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
    *r = mpq_equal (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    *sgn = mpq_sgn (x);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    const uint64_t size    // # of bits in x
)
{
    // ensure the mpfr number is not too big
    if (size > MPFR_PREC_MAX/2)
    {
        return (SPEX_PANIC);
    }

    // initialize the mpfr number
    SPEX_GMPFR_WRAPPER_START (x);
    mpfr_init2 (x, (mpfr_prec_t) size);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpfr_set_prec
//------------------------------------------------------------------------------

/* Purpose: Set the precision of an mpfr_t number */

SPEX_info SPEX_mpfr_set_prec
(
    mpfr_t x,       // Floating point number to revise
    const uint64_t size    // # of bits in x
)
{
    // ensure the mpfr number is not too big
    if (size > MPFR_PREC_MAX/2)
    {
        return (SPEX_PANIC);
    }

    // set the precision of the mpfr number
    SPEX_GMPFR_WRAPPER_START (x);
    mpfr_set_prec (x, (mpfr_prec_t) size);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPFR_WRAPPER_START (x);
    mpfr_set (x, y, rnd);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPFR_WRAPPER_START (x);
    mpfr_set_d (x, y, rnd);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPFR_WRAPPER_START (x);
    mpfr_set_si (x, (long int) y, rnd);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPFR_WRAPPER_START (x);
    mpfr_set_q (x, y, rnd);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPFR_WRAPPER_START (x);
    mpfr_set_z (x, y, rnd);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPZ_WRAPPER_START (x);
    mpfr_get_z (x, y, rnd);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPQ_WRAPPER_START (x);
    mpfr_get_q (x, y);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    *x = mpfr_get_d (y, rnd);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    *x = (int64_t) mpfr_get_si (y, rnd);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPFR_WRAPPER_START (x);
    mpfr_mul (x, y, z, rnd);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPFR_WRAPPER_START (x);
    mpfr_mul_d (x, y, z, rnd);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPFR_WRAPPER_START (x);
    mpfr_div_d (x, y, z, rnd);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPFR_WRAPPER_START (x);
    mpfr_ui_pow_ui (x, (unsigned long int) y, (unsigned long int) z, rnd);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    SPEX_GMPFR_WRAPPER_START (x);
    mpfr_log2 (x, y, rnd);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
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
    *sgn = mpfr_sgn (x);
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpfr_free_cache
//------------------------------------------------------------------------------

/* Purpose: Safely free all caches and pools used by MPFR internally */

SPEX_info SPEX_mpfr_free_cache ( void )
{
    SPEX_GMP_WRAPPER_START ;
    mpfr_free_cache ( );
    SPEX_GMP_WRAPPER_FINISH ;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpz_set_null
//------------------------------------------------------------------------------

// Purpose: initialize the contents of an mpz_t value

SPEX_info SPEX_mpz_set_null
(
    mpz_t x
)
{
    SPEX_MPZ_SET_NULL (x) ;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpq_set_null
//------------------------------------------------------------------------------

// Purpose: initialize the contents of an mpq_t value

SPEX_info SPEX_mpq_set_null
(
    mpq_t x
)
{
    SPEX_MPQ_SET_NULL (x) ;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpfr_set_null
//------------------------------------------------------------------------------

// Purpose: initialize the contents of an mpfr_t value

SPEX_info SPEX_mpfr_set_null
(
    mpfr_t x
)
{
    SPEX_MPFR_SET_NULL (x) ;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpz_clear
//------------------------------------------------------------------------------

// Purpose: safely clear an mpz_t value

SPEX_info SPEX_mpz_clear
(
    mpz_t x
)
{
    SPEX_MPZ_CLEAR (x) ;
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpq_clear
//------------------------------------------------------------------------------

// Purpose: safely clear an mpq_t value

SPEX_info SPEX_mpq_clear
(
    mpq_t x
)
{
    if (x != NULL)
    {
        SPEX_MPQ_CLEAR (x) ;
    }
    return (SPEX_OK);
}

//------------------------------------------------------------------------------
// SPEX_mpfr_clear
//------------------------------------------------------------------------------

// Purpose: safely clear an mpfr_t value

SPEX_info SPEX_mpfr_clear
(
    mpfr_t x
)
{
    SPEX_MPFR_CLEAR (x) ;
    return (SPEX_OK);
}

