//------------------------------------------------------------------------------
// SuiteSparse_config/SuiteSparse_config.c: common utilites for SuiteSparse
//------------------------------------------------------------------------------

// SuiteSparse_config, Copyright (c) 2012-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

/* SuiteSparse configuration : memory manager and printf functions.
 */

#include "SuiteSparse_config.h"

/* -------------------------------------------------------------------------- */
/* SuiteSparse_config : a static struct */
/* -------------------------------------------------------------------------- */

/* The SuiteSparse_config struct is indirectly available to all SuiteSparse
    functions and to all applications that use those functions.  In v6.x and
    earlier, it was globally visible, but it is now hidden and accessible only
    by functions in this file (SuiteSparse v7.0.0 and later).

    It must be modified with care, particularly in a multithreaded context.
    Normally, the application will initialize this object once, via
    SuiteSparse_start, possibily followed by application-specific modifications
    if the applications wants to use alternative memory manager functions.

    The user can redefine these pointers at run-time to change the
    memory manager and printf function used by SuiteSparse.

    If -DNMALLOC is defined at compile-time, then no memory-manager is
    specified.  You must define them at run-time, after calling
    SuiteSparse_start.

    If -DPRINT is defined a compile time, then printf is disabled, and
    SuiteSparse will not use printf.
 */

struct SuiteSparse_config_struct
{
    void *(*malloc_func) (size_t) ;             // pointer to malloc
    void *(*calloc_func) (size_t, size_t) ;     // pointer to calloc
    void *(*realloc_func) (void *, size_t) ;    // pointer to realloc
    void (*free_func) (void *) ;                // pointer to free
    int (*printf_func) (const char *, ...) ;    // pointer to printf
    double (*hypot_func) (double, double) ;     // pointer to hypot
    int (*divcomplex_func) (double, double, double, double, double *, double *);
} ;

static struct SuiteSparse_config_struct SuiteSparse_config =
{

    /* memory management functions */
    #ifndef NMALLOC
        #ifdef MATLAB_MEX_FILE
            /* MATLAB mexFunction: */
            mxMalloc, mxCalloc, mxRealloc, mxFree,
        #else
            /* standard ANSI C: */
            malloc, calloc, realloc, free,
        #endif
    #else
        /* no memory manager defined; you must define one at run-time: */
        NULL, NULL, NULL, NULL,
    #endif

    /* printf function */
    #ifndef NPRINT
        #ifdef MATLAB_MEX_FILE
            /* MATLAB mexFunction: */
            mexPrintf,
        #else
            /* standard ANSI C: */
            printf,
        #endif
    #else
        /* printf is disabled */
        NULL,
    #endif

    hypot, // was SuiteSparse_hypot in v5 and earlier
    SuiteSparse_divcomplex

} ;

//------------------------------------------------------------------------------
// SuiteSparse_config_*_get methods
//------------------------------------------------------------------------------

// Methods that return the contents of the SuiteSparse_config struct.

void *(*SuiteSparse_config_malloc_func_get (void)) (size_t)
{
    return (SuiteSparse_config.malloc_func) ;
}

void *(*SuiteSparse_config_calloc_func_get (void)) (size_t, size_t)
{
    return (SuiteSparse_config.calloc_func) ;
}

void *(*SuiteSparse_config_realloc_func_get (void)) (void *, size_t)
{
    return (SuiteSparse_config.realloc_func) ;
}

void (*SuiteSparse_config_free_func_get (void)) (void *)
{
    return (SuiteSparse_config.free_func) ;
}

int (*SuiteSparse_config_printf_func_get (void)) (const char *, ...)
{
    return (SuiteSparse_config.printf_func) ;
}

double (*SuiteSparse_config_hypot_func_get (void)) (double, double)
{
    return (SuiteSparse_config.hypot_func) ;
}

int (*SuiteSparse_config_divcomplex_func_get (void)) (double, double, double, double, double *, double *)
{
    return (SuiteSparse_config.divcomplex_func) ;
}

//------------------------------------------------------------------------------
// SuiteSparse_config_*_set methods
//------------------------------------------------------------------------------

// Methods that set the contents of the SuiteSparse_config struct.

void SuiteSparse_config_malloc_func_set (void *(*malloc_func) (size_t))
{
    SuiteSparse_config.malloc_func = malloc_func ;
}

void SuiteSparse_config_calloc_func_set (void *(*calloc_func) (size_t, size_t))
{
    SuiteSparse_config.calloc_func = calloc_func ;
}

void SuiteSparse_config_realloc_func_set (void *(*realloc_func) (void *, size_t))
{
    SuiteSparse_config.realloc_func = realloc_func ;
}

void SuiteSparse_config_free_func_set (void (*free_func) (void *))
{
    SuiteSparse_config.free_func = free_func ;
}

void SuiteSparse_config_printf_func_set (int (*printf_func) (const char *, ...))
{
    SuiteSparse_config.printf_func = printf_func ;
}

void SuiteSparse_config_hypot_func_set (double (*hypot_func) (double, double))
{
    SuiteSparse_config.hypot_func = hypot_func ;
}

void SuiteSparse_config_divcomplex_func_set (int (*divcomplex_func) (double, double, double, double, double *, double *))
{
    SuiteSparse_config.divcomplex_func = divcomplex_func ;
}

//------------------------------------------------------------------------------
// SuiteSparse_config_*_call methods
//------------------------------------------------------------------------------

// Methods that directly call the functions in the SuiteSparse_config struct.
// Note that there is no wrapper for the printf_func.

void *SuiteSparse_config_malloc (size_t s)
{
    return (SuiteSparse_config.malloc_func (s)) ;
}

void *SuiteSparse_config_calloc (size_t n, size_t s)
{
    return (SuiteSparse_config.calloc_func (n, s)) ;
}

void *SuiteSparse_config_realloc (void *p, size_t s)
{
    return (SuiteSparse_config.realloc_func (p, s)) ;
}

void SuiteSparse_config_free (void *p)
{
    SuiteSparse_config.free_func (p) ;
}

double SuiteSparse_config_hypot (double x, double y)
{
    return (SuiteSparse_config.hypot_func (x, y)) ;
}

int SuiteSparse_config_divcomplex
(
    double xr, double xi, double yr, double yi, double *zr, double *zi
)
{
    return (SuiteSparse_config.divcomplex_func (xr, xi, yr, yi, zr, zi)) ;
}

/* -------------------------------------------------------------------------- */
/* SuiteSparse_start */
/* -------------------------------------------------------------------------- */

/* All applications that use SuiteSparse should call SuiteSparse_start prior
   to using any SuiteSparse function.  Only a single thread should call this
   function, in a multithreaded application.  Currently, this function is
   optional, since all this function currently does is to set the four memory
   function pointers to NULL (which tells SuiteSparse to use the default
   functions).  In a multi- threaded application, only a single thread should
   call this function.

   Future releases of SuiteSparse might enforce a requirement that
   SuiteSparse_start be called prior to calling any SuiteSparse function.
 */

void SuiteSparse_start ( void )
{

    /* memory management functions */
    #ifndef NMALLOC
        #ifdef MATLAB_MEX_FILE
            /* MATLAB mexFunction: */
            SuiteSparse_config.malloc_func  = mxMalloc ;
            SuiteSparse_config.calloc_func  = mxCalloc ;
            SuiteSparse_config.realloc_func = mxRealloc ;
            SuiteSparse_config.free_func    = mxFree ;
        #else
            /* standard ANSI C: */
            SuiteSparse_config.malloc_func  = malloc ;
            SuiteSparse_config.calloc_func  = calloc ;
            SuiteSparse_config.realloc_func = realloc ;
            SuiteSparse_config.free_func    = free ;
        #endif
    #else
        /* no memory manager defined; you must define one after calling
           SuiteSparse_start */
        SuiteSparse_config.malloc_func  = NULL ;
        SuiteSparse_config.calloc_func  = NULL ;
        SuiteSparse_config.realloc_func = NULL ;
        SuiteSparse_config.free_func    = NULL ;
    #endif

    /* printf function */
    #ifndef NPRINT
        #ifdef MATLAB_MEX_FILE
            /* MATLAB mexFunction: */
            SuiteSparse_config.printf_func = mexPrintf ;
        #else
            /* standard ANSI C: */
            SuiteSparse_config.printf_func = printf ;
        #endif
    #else
        /* printf is disabled */
        SuiteSparse_config.printf_func = NULL ;
    #endif

    /* math functions */
    SuiteSparse_config.hypot_func = hypot ; // was SuiteSparse_hypot in v5
    SuiteSparse_config.divcomplex_func = SuiteSparse_divcomplex ;
}

/* -------------------------------------------------------------------------- */
/* SuiteSparse_finish */
/* -------------------------------------------------------------------------- */

/* This currently does nothing, but in the future, applications should call
   SuiteSparse_start before calling any SuiteSparse function, and then
   SuiteSparse_finish after calling the last SuiteSparse function, just before
   exiting.  In a multithreaded application, only a single thread should call
   this function.

   Future releases of SuiteSparse might use this function for any
   SuiteSparse-wide cleanup operations or finalization of statistics.
 */

void SuiteSparse_finish ( void )
{
    /* do nothing */ ;
}

/* -------------------------------------------------------------------------- */
/* SuiteSparse_malloc: malloc wrapper */
/* -------------------------------------------------------------------------- */

void *SuiteSparse_malloc    /* pointer to allocated block of memory */
(
    size_t nitems,          /* number of items to malloc */
    size_t size_of_item     /* sizeof each item */
)
{
    void *p ;
    size_t size ;
    if (nitems < 1) nitems = 1 ;
    if (size_of_item < 1) size_of_item = 1 ;
    size = nitems * size_of_item  ;

    if (size != ((double) nitems) * size_of_item)
    {
        /* size_t overflow */
        p = NULL ;
    }
    else
    {
        p = (void *) (SuiteSparse_config.malloc_func) (size) ;
    }
    return (p) ;
}

/* -------------------------------------------------------------------------- */
/* SuiteSparse_calloc: calloc wrapper */
/* -------------------------------------------------------------------------- */

void *SuiteSparse_calloc    /* pointer to allocated block of memory */
(
    size_t nitems,          /* number of items to calloc */
    size_t size_of_item     /* sizeof each item */
)
{
    void *p ;
    size_t size ;
    if (nitems < 1) nitems = 1 ;
    if (size_of_item < 1) size_of_item = 1 ;
    size = nitems * size_of_item  ;

    if (size != ((double) nitems) * size_of_item)
    {
        /* size_t overflow */
        p = NULL ;
    }
    else
    {
        p = (void *) (SuiteSparse_config.calloc_func) (nitems, size_of_item) ;
    }
    return (p) ;
}

/* -------------------------------------------------------------------------- */
/* SuiteSparse_realloc: realloc wrapper */
/* -------------------------------------------------------------------------- */

/* If p is non-NULL on input, it points to a previously allocated object of
   size nitems_old * size_of_item.  The object is reallocated to be of size
   nitems_new * size_of_item.  If p is NULL on input, then a new object of that
   size is allocated.  On success, a pointer to the new object is returned,
   and ok is returned as 1.  If the allocation fails, ok is set to 0 and a
   pointer to the old (unmodified) object is returned.
 */

void *SuiteSparse_realloc   /* pointer to reallocated block of memory, or
                               to original block if the realloc failed. */
(
    size_t nitems_new,      /* new number of items in the object */
    size_t nitems_old,      /* old number of items in the object */
    size_t size_of_item,    /* sizeof each item */
    void *p,                /* old object to reallocate */
    int *ok                 /* 1 if successful, 0 otherwise */
)
{
    size_t size ;
    if (nitems_old < 1) nitems_old = 1 ;
    if (nitems_new < 1) nitems_new = 1 ;
    if (size_of_item < 1) size_of_item = 1 ;
    size = nitems_new * size_of_item  ;

    if (size != ((double) nitems_new) * size_of_item)
    {
        /* size_t overflow */
        (*ok) = 0 ;
    }
    else if (p == NULL)
    {
        /* a fresh object is being allocated */
        p = SuiteSparse_malloc (nitems_new, size_of_item) ;
        (*ok) = (p != NULL) ;
    }
    else if (nitems_old == nitems_new)
    {
        /* the object does not change; do nothing */
        (*ok) = 1 ;
    }
    else
    {
        /* change the size of the object from nitems_old to nitems_new */
        void *pnew ;
        pnew = (void *) (SuiteSparse_config.realloc_func) (p, size) ;
        if (pnew == NULL)
        {
            if (nitems_new < nitems_old)
            {
                /* the attempt to reduce the size of the block failed, but
                   the old block is unchanged.  So pretend to succeed. */
                (*ok) = 1 ;
            }
            else
            {
                /* out of memory */
                (*ok) = 0 ;
            }
        }
        else
        {
            /* success */
            p = pnew ;
            (*ok) = 1 ;
        }
    }
    return (p) ;
}

/* -------------------------------------------------------------------------- */
/* SuiteSparse_free: free wrapper */
/* -------------------------------------------------------------------------- */

void *SuiteSparse_free      /* always returns NULL */
(
    void *p                 /* block to free */
)
{
    if (p)
    {
        (SuiteSparse_config.free_func) (p) ;
    }
    return (NULL) ;
}

/* -------------------------------------------------------------------------- */
/* SuiteSparse_tic: return current wall clock time */
/* -------------------------------------------------------------------------- */

/* Returns the number of seconds (tic [0]) and nanoseconds (tic [1]) since some
 * unspecified but fixed time in the past.  If no timer is installed, zero is
 * returned.  A scalar double precision value for 'tic' could be used, but this
 * might cause loss of precision because clock_getttime returns the time from
 * some distant time in the past.  Thus, an array of size 2 is used.
 *
 * The timer is enabled by default.  To disable the timer, compile with
 * -DNTIMER.  If enabled on a POSIX C 1993 system, the timer requires linking
 * with the -lrt library.
 *
 * example:
 *
 *      double tic [2], r, s, t ;
 *      SuiteSparse_tic (tic) ;     // start the timer
 *      // do some work A
 *      t = SuiteSparse_toc (tic) ; // t is time for work A, in seconds
 *      // do some work B
 *      s = SuiteSparse_toc (tic) ; // s is time for work A and B, in seconds
 *      SuiteSparse_tic (tic) ;     // restart the timer
 *      // do some work C
 *      r = SuiteSparse_toc (tic) ; // s is time for work C, in seconds
 *
 * A double array of size 2 is used so that this routine can be more easily
 * ported to non-POSIX systems.  The caller does not rely on the POSIX
 * <time.h> include file.
 */

#if !defined ( SUITESPARSE_TIMER_ENABLED )

    /* ---------------------------------------------------------------------- */
    /* no timer */
    /* ---------------------------------------------------------------------- */

    void SuiteSparse_tic
    (
        double tic [2]      /* output, contents undefined on input */
    )
    {
        /* no timer installed */
        tic [0] = 0 ;
        tic [1] = 0 ;
    }

#elif defined ( _OPENMP )

    /* ---------------------------------------------------------------------- */
    /* OpenMP timer */
    /* ---------------------------------------------------------------------- */

    void SuiteSparse_tic
    (
        double tic [2]      /* output, contents undefined on input */
    )
    {
        tic [0] = omp_get_wtime ( ) ;
        tic [1] = 0 ;
    }

#else

    /* ---------------------------------------------------------------------- */
    /* POSIX timer */
    /* ---------------------------------------------------------------------- */

    #include <time.h>
    void SuiteSparse_tic
    (
        double tic [2]      /* output, contents undefined on input */
    )
    {
        /* POSIX C 1993 timer, requires -lrt */
        struct timespec t ;
        clock_gettime (CLOCK_MONOTONIC, &t) ;
        tic [0] = (double) (t.tv_sec) ;
        tic [1] = (double) (t.tv_nsec) ;
    }

#endif

/* -------------------------------------------------------------------------- */
/* SuiteSparse_toc: return time since last tic */
/* -------------------------------------------------------------------------- */

/* Assuming SuiteSparse_tic is accurate to the nanosecond, this function is
 * accurate down to the nanosecond for 2^53 nanoseconds since the last call to
 * SuiteSparse_tic, which is sufficient for SuiteSparse (about 104 days).  If
 * additional accuracy is required, the caller can use two calls to
 * SuiteSparse_tic and do the calculations differently.
 */

double SuiteSparse_toc  /* returns time in seconds since last tic */
(
    double tic [2]  /* input, not modified from last call to SuiteSparse_tic */
)
{
    double toc [2] ;
    SuiteSparse_tic (toc) ;
    return ((toc [0] - tic [0]) + 1e-9 * (toc [1] - tic [1])) ;
}

/* -------------------------------------------------------------------------- */
/* SuiteSparse_time: return current wallclock time in seconds */
/* -------------------------------------------------------------------------- */

/* This function might not be accurate down to the nanosecond. */

double SuiteSparse_time  /* returns current wall clock time in seconds */
(
    void
)
{
    double toc [2] ;
    SuiteSparse_tic (toc) ;
    return (toc [0] + 1e-9 * toc [1]) ;
}

/* -------------------------------------------------------------------------- */
/* SuiteSparse_version: return the current version of SuiteSparse */
/* -------------------------------------------------------------------------- */

int SuiteSparse_version
(
    int version [3]
)
{
    if (version != NULL)
    {
        version [0] = SUITESPARSE_MAIN_VERSION ;
        version [1] = SUITESPARSE_SUB_VERSION ;
        version [2] = SUITESPARSE_SUBSUB_VERSION ;
    }
    return (SUITESPARSE_VERSION) ;
}

//------------------------------------------------------------------------------
// SuiteSparse_hypot
//------------------------------------------------------------------------------

// SuiteSparse_config v5 and earlier used SuiteSparse_hypot, defined below.
// SuiteSparse_config v6 now uses the hypot method in <math.h>, by default.
// The hypot function appears in ANSI C99 and later, and SuiteSparse now
// assumes ANSI C11.

// s = hypot (x,y) computes s = sqrt (x*x + y*y) but does so more accurately.
// The NaN cases for the double relops x >= y and x+y == x are safely ignored.

// Source: Algorithm 312, "Absolute value and square root of a complex number,"
// P. Friedland, Comm. ACM, vol 10, no 10, October 1967, page 665.

// This method below is kept for historical purposes.

double SuiteSparse_hypot (double x, double y)
{
    double s, r ;
    x = fabs (x) ;
    y = fabs (y) ;
    if (x >= y)
    {
        if (x + y == x)
        {
            s = x ;
        }
        else
        {
            r = y / x ;
            s = x * sqrt (1.0 + r*r) ;
        }
    }
    else
    {
        if (y + x == y)
        {
            s = y ;
        }
        else
        {
            r = x / y ;
            s = y * sqrt (1.0 + r*r) ;
        }
    }
    return (s) ;
}

//------------------------------------------------------------------------------
// SuiteSparse_divcomplex
//------------------------------------------------------------------------------

// z = x/y where z, x, and y are complex.  The real and imaginary parts are
// passed as separate arguments to this routine.  The NaN case is ignored
// for the double relop yr >= yi.  Returns 1 if the denominator is zero,
// 0 otherwise.
//
// This uses ACM Algo 116, by R. L. Smith, 1962, which tries to avoid
// underflow and overflow.
//
// z can be the same variable as x or y.
//
// Default value of the SuiteSparse_config.divcomplex_func pointer is
// SuiteSparse_divcomplex.
//
// This function is identical to GB_divcomplex in GraphBLAS/Source/GB_math.h.
// The only difference is the name of the function.

int SuiteSparse_divcomplex
(
    double xr, double xi,       // real and imaginary parts of x
    double yr, double yi,       // real and imaginary parts of y
    double *zr, double *zi      // real and imaginary parts of z
)
{
    double tr, ti, r, den ;

    int yr_class = fpclassify (yr) ;
    int yi_class = fpclassify (yi) ;

    if (yi_class == FP_ZERO)
    {
        den = yr ;
        if (xi == 0)
        {
            tr = xr / den ;
            ti = 0 ;
        }
        else if (xr == 0)
        {
            tr = 0 ;
            ti = xi / den ;
        }
        else
        {
            tr = xr / den ;
            ti = xi / den ;
        }
    }
    else if (yr_class == FP_ZERO)
    {
        den = yi ;
        if (xr == 0)
        {
            tr = xi / den ;
            ti = 0 ;
        }
        else if (xi == 0)
        {
            tr = 0 ;
            ti = -xr / den ;
        }
        else
        {
            tr = xi / den ;
            ti = -xr / den ;
        }
    }
    else if (yi_class == FP_INFINITE && yr_class == FP_INFINITE)
    {

        if (signbit (yr) == signbit (yi))
        {
            // r = 1
            den = yr + yi ;
            tr = (xr + xi) / den ;
            ti = (xi - xr) / den ;
        }
        else
        {
            // r = -1
            den = yr - yi ;
            tr = (xr - xi) / den ;
            ti = (xi + xr) / den ;
        }

    }
    else
    {

        if (fabs (yr) >= fabs (yi))
        {
            r = yi / yr ;
            den = yr + r * yi ;
            tr = (xr + xi * r) / den ;
            ti = (xi - xr * r) / den ;
        }
        else
        {
            r = yr / yi ;
            den = r * yr + yi ;
            tr = (xr * r + xi) / den ;
            ti = (xi * r - xr) / den ;
        }

    }
    (*zr) = tr ;
    (*zi) = ti ;
    return (den == 0) ;
}

//------------------------------------------------------------------------------
// SuiteSparse_BLAS_library: return name of BLAS library found
//------------------------------------------------------------------------------

// Returns the name of the BLAS library found by SuiteSparse_config

const char *SuiteSparse_BLAS_library ( void )
{
    #if defined ( BLAS_Intel10_64ilp )
        return ("Intel MKL 64ilp BLAS (64-bit integers)") ;
    #elif defined ( BLAS_Intel10_64lp )
        return ("Intel MKL 64lp BLAS (32-bit integers)") ;
    #elif defined ( BLAS_Apple )
        return ("Apple Accelerate Framework BLAS (32-bit integers)") ;
    #elif defined ( BLAS_Arm_ilp64_mp )
        return ("ARM MP BLAS (64-bit integers)") ;
    #elif defined ( BLAS_Arm_mp )
        return ("ARM MP BLAS (32-bit integers)") ;
    #elif defined ( BLAS_IBMESSL_SMP )
        return ((sizeof (SUITESPARSE_BLAS_INT) == 8) ?
            "IBMESSL_SMP BLAS (64-bit integers)" :
            "IBMESSL_SMP BLAS (32-bit integers)") ;
    #elif defined ( BLAS_OpenBLAS )
        return ((sizeof (SUITESPARSE_BLAS_INT) == 8) ?
            "OpenBLAS (64-bit integers)" :
            "OpenBLAS (32-bit integers)") ;
    #elif defined ( BLAS_FLAME )
        return ((sizeof (SUITESPARSE_BLAS_INT) == 8) ?
            "FLAME (64-bit integers)" :
            "FLAME (32-bit integers)") ;
    #elif defined ( BLAS_Generic )
        return ((sizeof (SUITESPARSE_BLAS_INT) == 8) ?
            "Reference BLAS (64-bit integers)" :
            "Reference BLAS (32-bit integers)") ;
    #else
        return ((sizeof (SUITESPARSE_BLAS_INT) == 8) ?
            "Other BLAS (64-bit integers)" :
            "Other BLAS (32-bit integers)") ;
    #endif
}

//------------------------------------------------------------------------------
// SuiteSparse_BLAS_integer: return size of BLAS integer
//------------------------------------------------------------------------------

size_t SuiteSparse_BLAS_integer_size ( void )
{
    return (sizeof (SUITESPARSE_BLAS_INT)) ;
}

