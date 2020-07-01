//------------------------------------------------------------------------------
// GB_BinaryOp_new: create a new binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Create a new a binary operator: z = f (x,y).  The binary function signature
// must be void f (void *z, const void *x, const void *y), and then it must
// recast its input and output arguments internally as needed.

// This function is not directly user-callable.  Use GrB_BinaryOp_new instead.

#include "GB.h"
#include <ctype.h>

GrB_Info GB_BinaryOp_new
(
    GrB_BinaryOp *binaryop,         // handle for the new binary operator
    GxB_binary_function function,   // pointer to the binary function
    GrB_Type ztype,                 // type of output z
    GrB_Type xtype,                 // type of input x
    GrB_Type ytype,                 // type of input y
    const char *name                // name of the function
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE ("GrB_BinaryOp_new (binaryop, function, ztype, xtype, ytype)") ;
    GB_RETURN_IF_NULL (binaryop) ;
    (*binaryop) = NULL ;
    GB_RETURN_IF_NULL (function) ;
    GB_RETURN_IF_NULL_OR_FAULTY (ztype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (xtype) ;
    GB_RETURN_IF_NULL_OR_FAULTY (ytype) ;

    //--------------------------------------------------------------------------
    // create the binary op
    //--------------------------------------------------------------------------

    // allocate the binary operator
    (*binaryop) = GB_CALLOC (1, struct GB_BinaryOp_opaque) ;
    if (*binaryop == NULL)
    { 
        // out of memory
        return (GB_OUT_OF_MEMORY) ;
    }

    // initialize the binary operator
    GrB_BinaryOp op = *binaryop ;
    op->magic = GB_MAGIC ;
    op->xtype = xtype ;
    op->ytype = ytype ;
    op->ztype = ztype ;
    op->function = function ;
    op->opcode = GB_USER_opcode ;     // user-defined operator

    //--------------------------------------------------------------------------
    // find the name of the operator
    //--------------------------------------------------------------------------

    if (name == NULL)
    { 
        // if no name , a generic name is used instead
        strncpy (op->name, "user_binary_operator", GB_LEN-1) ;
    }
    else
    {
        // see if the typecast "(GxB_binary_function)" appears in the name
        char *p = NULL ;
        p = strstr ((char *) name, "GxB_binary_function") ;
        if (p != NULL)
        { 
            // skip past the typecast, the left parenthesis, and any whitespace
            p += 19 ;
            while (isspace (*p)) p++ ;
            if (*p == ')') p++ ;
            while (isspace (*p)) p++ ;
            strncpy (op->name, p, GB_LEN-1) ;
        }
        else
        { 
            // copy the entire name as-is
            strncpy (op->name, name, GB_LEN-1) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_BINARYOP_OK (op, "new user-defined binary op", GB0) ;
    return (GrB_SUCCESS) ;
}

