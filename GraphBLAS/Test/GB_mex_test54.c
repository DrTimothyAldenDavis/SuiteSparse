//------------------------------------------------------------------------------
// GB_mex_test54: select NONZOMBIE with user-defined type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Regression test for a bug in versions 10.5.3 and earlier, when JIT-ing
// the NONZOMBIE operator for GrB_select, with a user-defined type.
// See issue #480.

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL                    \
{                                   \
    mxFree (in) ;                   \
    mxFree (out) ;                  \
    GrB_Matrix_free (&A) ;          \
}

//------------------------------------------------------------------------------

static GrB_Info GB_mex_test54_worker (GrB_Type type, size_t size)
{

    //--------------------------------------------------------------------------
    // construct a test matrix with the given type, and allocate workspace
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix A = NULL ;
    OK (GrB_Matrix_new (&A, type, 8, 8)) ;
    OK (GrB_set (A, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;

    unsigned char *in = mxMalloc (size), *out = mxMalloc (size) ;
    CHECK (in != NULL && out != NULL) ;

    //--------------------------------------------------------------------------
    // assign entries to the diagonal of A
    //--------------------------------------------------------------------------

    for (GrB_Index k = 0 ; k < 8 ; k++)
    {
        memset (in, (int) k + 1, size) ;
        OK (GrB_Matrix_setElement_UDT (A, in, k, k)) ;
    }

    //--------------------------------------------------------------------------
    // wait, remove the A(3,3) entry, and wait again
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_removeElement (A, 3, 3)) ;
    // Materialization invokes the internal NONZOMBIE selector.
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // check the result
    //--------------------------------------------------------------------------

    GrB_Index n = 0 ;
    OK (GrB_Matrix_nvals (&n, A)) ;
    CHECK (n == 7) ;
    for (GrB_Index k = 0 ; k < 8 ; k++)
    {
        if (k == 3) continue ;
        memset (in, (int) k + 1, size) ;
        OK (GrB_Matrix_extractElement_UDT (out, A, k, k)) ;
        CHECK (memcmp (in, out, size) == 0) ;
    }

    //--------------------------------------------------------------------------
    // ensure the JIT is still on
    //--------------------------------------------------------------------------

    int32_t jit = 0 ;
    OK (GxB_Global_Option_get_INT32 (GxB_JIT_C_CONTROL, &jit)) ;
    CHECK (jit == GxB_JIT_ON) ;

    //--------------------------------------------------------------------------
    // free everything and return result
    //--------------------------------------------------------------------------

    FREE_ALL ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_mex_test54 mexFunction
//------------------------------------------------------------------------------

#undef  FREE_ALL
#define FREE_ALL                    \
{                                   \
    GrB_Type_free (&t) ;            \
}

typedef struct { uint64_t x [4] ; } select_large ;
typedef struct { uint64_t x [2] ; } select_small ;


void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Type t = NULL ;

    //--------------------------------------------------------------------------
    // ensure the JIT is on, and turn on the burble
    //--------------------------------------------------------------------------

    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    OK (GrB_set (GrB_GLOBAL, true, GxB_BURBLE)) ;

    //--------------------------------------------------------------------------
    // test an unnamed type; this should not use the JIT
    //--------------------------------------------------------------------------

    OK (GrB_Type_new (&t, 24)) ;
    OK (GB_mex_test54_worker (t, 24)) ;
    FREE_ALL ;

    //--------------------------------------------------------------------------
    // test a 32-byte type; this should use the JIT
    //--------------------------------------------------------------------------

    OK (GxB_Type_new (&t, sizeof (select_large), "select_large",
        "#include <stdint.h>\n"
        "typedef struct { uint64_t x [4] ; } select_large ;")) ;
    OK (GB_mex_test54_worker (t, sizeof (select_large))) ;
    FREE_ALL ;

    //--------------------------------------------------------------------------
    // test a 16-byte type; this should use the JIT
    //--------------------------------------------------------------------------

    OK (GxB_Type_new (&t, sizeof (select_small), "select_small",
        "#include <stdint.h>\n"
        "typedef struct { uint64_t x [2] ; } select_small ;")) ;
    OK (GB_mex_test54_worker (t, sizeof (select_small))) ;
    FREE_ALL ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    OK (GrB_set (GrB_GLOBAL, false, GxB_BURBLE)) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test54: all tests passed\n\n") ;
}

