//------------------------------------------------------------------------------
// gbfull: convert a GraphBLAS matrix struct into a MATLAB dense matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard MATLAB
// sparse or dense matrix.  The output is a GraphBLAS matrix by default, with
// all entries present, of the given type.  Entries are filled in with the id
// value, whose default value is zero.  If desc.kind = 'full', the output is a
// MATLAB dense matrix.

// Usage:
//  C = gbfull (A)
//  C = gbfull (A, type)
//  C = gbfull (A, type, id)
//  C = gbfull (A, type, id, desc)

#include "gb_matlab.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin >= 1 && nargin <= 4 && nargout <= 2,
        "usage: C = gbfull (A, type, id, desc)") ;

    //--------------------------------------------------------------------------
    // get a shallow copy of the input matrix
    //--------------------------------------------------------------------------

    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    GrB_Index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    //--------------------------------------------------------------------------
    // get the type of C
    //--------------------------------------------------------------------------

    GrB_Matrix type ;
    if (nargin > 1)
    { 
        type = gb_mxstring_to_type (pargin [1]) ;
    }
    else
    { 
        // the output type defaults to the same as the input type
        OK (GxB_Matrix_type (&type, A)) ;
    }

    //--------------------------------------------------------------------------
    // get the identity scalar
    //--------------------------------------------------------------------------

    GrB_Matrix id ;
    if (nargin > 2)
    { 
        id = gb_get_shallow (pargin [2]) ;
    }
    else
    { 
        // Assume the identity is zero, of the same type as C.
        // The format does not matter, since only id (0,0) will be used.
        OK (GrB_Matrix_new (&id, type, 1, 1)) ;
    }

    //--------------------------------------------------------------------------
    // get the descriptor
    //--------------------------------------------------------------------------

    base_enum_t base = BASE_DEFAULT ;
    kind_enum_t kind = KIND_GRB ;
    GxB_Format_Value fmt = GxB_NO_FORMAT ;
    GrB_Descriptor desc = NULL ;
    if (nargin > 3)
    { 
        desc = gb_mxarray_to_descriptor (pargin [nargin-1], &kind, &fmt, &base);
    }
    OK (GrB_Descriptor_free (&desc)) ;

    // A determines the format of C, unless defined by the descriptor
    fmt = gb_get_format (nrows, ncols, A, NULL, fmt) ;

    //--------------------------------------------------------------------------
    // expand the identity into a dense matrix B the same size as C
    //--------------------------------------------------------------------------

    GrB_Matrix B ;
    OK (GrB_Matrix_new (&B, type, nrows, ncols)) ;
    OK (GxB_Matrix_Option_set (B, GxB_FORMAT, fmt)) ;
    gb_matrix_assign_scalar (B, NULL, NULL, id, GrB_ALL, 0, GrB_ALL, 0, NULL,
        false) ;

    //--------------------------------------------------------------------------
    // typecast A from float to integer using the MATLAB rules
    //--------------------------------------------------------------------------

    GrB_Matrix S, T = NULL ;
    GrB_Type atype ;
    OK (GxB_Matrix_type (&atype, A)) ;
    if (gb_is_integer (type) && gb_is_float (atype))
    { 
        // T = (type) round (A)
        OK (GrB_Matrix_new (&T, type, nrows, ncols)) ;
        OK (GxB_Matrix_Option_set (T, GxB_FORMAT, fmt)) ;
        OK (GrB_Matrix_apply (T, NULL, NULL, gb_round_binop (atype), A, NULL)) ;
        S = T ;
    }
    else
    { 
        // T = A, and let GrB_Matrix_eWiseAdd_BinaryOp do the typecasting
        S = A ;
    }

    //--------------------------------------------------------------------------
    // C = first (S, B)
    //--------------------------------------------------------------------------

    GrB_Matrix C ;
    OK (GrB_Matrix_new (&C, type, nrows, ncols)) ;
    OK (GxB_Matrix_Option_set (C, GxB_FORMAT, fmt)) ;
    OK (GrB_Matrix_eWiseAdd_BinaryOp (C, NULL, NULL,
        gb_first_binop (type), S, B, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&id)) ;
    OK (GrB_Matrix_free (&B)) ;
    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Matrix_free (&T)) ;

    //--------------------------------------------------------------------------
    // export C to a MATLAB dense matrix
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    GB_WRAPUP ;
}

