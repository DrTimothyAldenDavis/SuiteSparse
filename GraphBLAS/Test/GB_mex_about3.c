//------------------------------------------------------------------------------
// GB_mex_about3: still more basic tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Test lots of random stuff.  The function otherwise serves no purpose.

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_about3"

int myprintf (const char *restrict format, ...) ;

int myprintf (const char *restrict format, ...)
{
    printf ("[[myprintf:") ;
    va_list ap ;
    va_start (ap, format) ;
    vprintf (format, ap) ;
    va_end (ap) ;
    printf ("]]") ;
}

int myflush (void) ;

int myflush (void)
{
    printf ("myflush\n") ;
    fflush (stdout) ;
}

typedef int (* printf_func_t) (const char *restrict format, ...) ;
typedef int (* flush_func_t)  (void) ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    GrB_Matrix C = NULL ;
    GrB_Vector w = NULL ;
    char *err ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    bool malloc_debug = GB_mx_get_global (true) ;
    FILE *f = fopen ("errlog4.txt", "w") ;
    int expected = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // GxB_set/get for printf and flush
    //--------------------------------------------------------------------------

    OK (GxB_Global_Option_set (GxB_BURBLE, true)) ;
    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;

    printf ("\nBurble with standard printf/flush:\n") ;
    GrB_Index nvals ;
    OK (GrB_Matrix_nvals (&nvals, C)) ;
    CHECK (nvals == 0) ;

    OK (GxB_Global_Option_set (GxB_PRINTF, myprintf)) ;
    OK (GxB_Global_Option_set (GxB_FLUSH, myflush)) ;

    printf_func_t mypr ;
    OK (GxB_Global_Option_get (GxB_PRINTF, &mypr)) ;
    CHECK (mypr == myprintf) ;

    flush_func_t myfl ;
    OK (GxB_Global_Option_get (GxB_FLUSH, &myfl)) ;
    CHECK (myfl == myflush) ;

    printf ("\nBurble with myprintf/myflush:\n") ;
    OK (GrB_Matrix_nvals (&nvals, C)) ;
    CHECK (nvals == 0) ;
    OK (GxB_Global_Option_set (GxB_BURBLE, false)) ;

    //--------------------------------------------------------------------------
    // test GxB_set/get for free_pool_limit
    //--------------------------------------------------------------------------

    int64_t free_pool_limit [64] ;
    OK (GxB_Global_Option_set (GxB_MEMORY_POOL, NULL)) ;
    OK (GxB_Global_Option_get (GxB_MEMORY_POOL, free_pool_limit)) ;
    printf ("\ndefault memory pool limits:\n") ;
    for (int k = 0 ; k < 64 ; k++)
    {
        if (free_pool_limit [k] > 0)
        {
            printf ("pool %2d: limit %ld\n", k, free_pool_limit [k]) ;
        }
    }
    for (int k = 0 ; k < 64 ; k++)
    {
        free_pool_limit [k] = k ;
    }
    OK (GxB_Global_Option_set (GxB_MEMORY_POOL, free_pool_limit)) ;
    OK (GxB_Global_Option_get (GxB_MEMORY_POOL, free_pool_limit)) ;
    for (int k = 0 ; k < 3 ; k++)
    {
        CHECK (free_pool_limit [k] == 0) ;
    }
    for (int k = 3 ; k < 64 ; k++)
    {
        CHECK (free_pool_limit [k] == k) ;
    }
    for (int k = 0 ; k < 64 ; k++)
    {
        free_pool_limit [k] = 0 ;
    }
    OK (GxB_Global_Option_set (GxB_MEMORY_POOL, free_pool_limit)) ;
    OK (GxB_Global_Option_get (GxB_MEMORY_POOL, free_pool_limit)) ;
    for (int k = 0 ; k < 64 ; k++)
    {
        CHECK (free_pool_limit [k] == 0) ;
    }

    //--------------------------------------------------------------------------
    // GrB_reduce with invalid binary op
    //--------------------------------------------------------------------------

    OK (GrB_Vector_new (&w, GrB_FP32, 10)) ;
    info = GrB_Matrix_reduce_BinaryOp (w, NULL, NULL, GrB_LT_FP32, C, NULL) ;
    CHECK (info == GrB_DOMAIN_MISMATCH) ;
    const char *s ;
    OK (GrB_error (&s, w)) ;
    printf ("expected error: [%s]\n", s) ;
    GrB_Vector_free_(&w) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GrB_Matrix_free_(&C) ;
    GB_mx_put_global (true) ;   
    fclose (f) ;
    printf ("\nGB_mex_about3: all tests passed\n\n") ;
}

