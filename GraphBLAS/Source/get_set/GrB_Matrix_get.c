//------------------------------------------------------------------------------
// GrB_Matrix_get_*: get a field in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Matrix_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_get_Scalar
(
    GrB_Matrix A,
    GrB_Scalar value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_get_Scalar (A, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_MATRIX_OK (A, "A to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    int32_t i ;
    GrB_Info info = GB_matvec_enum_get (A, &i, field) ;
    if (info == GrB_SUCCESS)
    { 
        // field specifies an int: assign it to the scalar
        info = GB_setElement ((GrB_Matrix) value, NULL, &i, 0, 0,
            GB_INT32_code, Werk) ;
    }
    else
    { 
        double x ;
        switch ((int) field)
        {
            case GxB_HYPER_SWITCH : 
                x = (double) (A->hyper_switch) ;
                break ;

            case GxB_BITMAP_SWITCH : 
                x = (double) (A->bitmap_switch) ;
                break ;

            default : 
                return (GrB_INVALID_VALUE) ;
        }
        // field specifies a double: assign it to the scalar
        info = GB_setElement ((GrB_Matrix) value, NULL, &x, 0, 0,
            GB_FP64_code, Werk) ;
    }

    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_get_String
(
    GrB_Matrix A,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_get_String (A, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_MATRIX_OK (A, "A to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_matvec_name_get (A, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_get_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_get_INT32
(
    GrB_Matrix A,
    int32_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_get_INT32 (A, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_MATRIX_OK (A, "A to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_matvec_enum_get (A, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_get_SIZE
(
    GrB_Matrix A,
    size_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_get_SIZE (A, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_MATRIX_OK (A, "A to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_matvec_name_size_get (A, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_get_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_get_VOID
(
    GrB_Matrix A,
    void * value,
    GrB_Field field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

