//------------------------------------------------------------------------------
// GrB_Type_set_*: set a field in a type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Type_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Type_set_Scalar
(
    GrB_Type type,
    GrB_Scalar value,
    GrB_Field field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GrB_Type_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Type_set_String
(
    GrB_Type type,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Type_set_String (type, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_TYPE_OK (type, "unaryop for get", GB0) ;

    //--------------------------------------------------------------------------
    // set the name or defn of a user-defined type
    //--------------------------------------------------------------------------

    bool user_defined = (type->code == GB_UDT_code) ;

    return (GB_op_or_type_string_set (user_defined, true, value, field,
        &(type->user_name), &(type->user_name_size),
        type->name, &(type->name_len), &(type->defn), &(type->defn_size),
        &(type->hash))) ;
}

//------------------------------------------------------------------------------
// GrB_Type_set_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Type_set_INT32
(
    GrB_Type type,
    int32_t value,
    GrB_Field field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GrB_Type_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Type_set_VOID
(
    GrB_Type type,
    void * value,
    GrB_Field field,
    size_t size
)
{ 
    return (GrB_INVALID_VALUE) ;
}

