//------------------------------------------------------------------------------
// GB_Semiring_check: check and print a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_Semiring_check          // check a GraphBLAS semiring
(
    const GrB_Semiring semiring,    // GraphBLAS semiring to print and check
    const char *name,               // name of the semiring, optional
    int pr,                         // print level
    FILE *f,                        // file for output
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBPR0 ("\n    GraphBLAS Semiring: %s ", GB_NAME) ;

    if (semiring == NULL)
    { 
        // GrB_error status not modified since this may be an optional argument
        GBPR0 ("NULL\n") ;
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // check object
    //--------------------------------------------------------------------------

    GB_CHECK_MAGIC (semiring, "Semiring") ;
    GBPR0 (semiring->builtin ? "(built-in)" : "(user-defined)") ;

    GrB_Info info ;
    info = GB_Monoid_check (semiring->add, "semiring->add", pr, f, Context) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    Semiring->add invalid\n") ;
        return (GB_ERROR (GrB_INVALID_OBJECT, (GB_LOG,
            "Semiring->add is an invalid monoid: [%s]", GB_NAME))) ;
    }

    info = GB_BinaryOp_check (semiring->multiply, "semiring->multiply", pr, f,
        Context) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    Semiring->multiply invalid\n") ;
        return (GB_ERROR (GrB_INVALID_OBJECT, (GB_LOG,
            "Semiring->multiply is an invalid operator: [%s]", GB_NAME))) ;
    }

    // z = multiply(x,y); type of z must match monoid type
    if (semiring->multiply->ztype != semiring->add->op->ztype)
    { 
        GBPR0 ("    Semiring multiply output domain must match monoid"
            " domain\n") ;
        return (GB_ERROR (GrB_INVALID_OBJECT, (GB_LOG,
            "Semiring multiply output domain must match monoid domain: [%s]",
            GB_NAME))) ;
    }

    return (GrB_SUCCESS) ;
}

