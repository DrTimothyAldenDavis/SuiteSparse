//------------------------------------------------------------------------------
// GrB_Semiring_get_*: get a field in a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Semiring_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_get_Scalar
(
    GrB_Semiring semiring,
    GrB_Scalar value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Semiring_get_Scalar (semiring, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_SEMIRING_OK (semiring, "semiring to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    switch ((int) field)
    {
        case GxB_MONOID_IDENTITY : 
        case GxB_MONOID_TERMINAL : 
            return (GB_monoid_get (semiring->add, value, field, Werk)) ;
        default : 
            return (GB_op_scalar_get ((GB_Operator) (semiring->multiply),
                value, field, Werk)) ;
    }
}

//------------------------------------------------------------------------------
// GrB_Semiring_get_String
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_get_String
(
    GrB_Semiring semiring,
    char * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Semiring_get_String (semiring, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_SEMIRING_OK (semiring, "semiring to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    (*value) = '\0' ;
    const char *name ;

    switch ((int) field)
    {
        case GrB_NAME : 

            name = GB_semiring_name_get (semiring) ;
            if (name != NULL)
            { 
                // get the name of the semiring
                strcpy (value, name) ;
            }
            #pragma omp flush
            return (GrB_SUCCESS) ;

        case GrB_INP0_TYPE_STRING : 
        case GrB_INP1_TYPE_STRING : 
        case GrB_OUTP_TYPE_STRING : 
            return (GB_op_string_get ((GB_Operator) (semiring->multiply),
                value, field)) ;

        default : ;
            return (GrB_INVALID_VALUE) ;
    }
}

//------------------------------------------------------------------------------
// GrB_Semiring_get_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_get_INT32
(
    GrB_Semiring semiring,
    int32_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Semiring_get_INT32 (semiring, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_SEMIRING_OK (semiring, "semiring to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_op_enum_get ((GB_Operator) (semiring->multiply), value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Semiring_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_get_SIZE
(
    GrB_Semiring semiring,
    size_t * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Semiring_get_SIZE (semiring, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_SEMIRING_OK (semiring, "semiring to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    const char *name ;

    switch ((int) field)
    {

        case GrB_NAME : 

            // get the length of the semiring user_name, or built-in name
            name = GB_semiring_name_get (semiring) ;
            break ;

        case GrB_INP0_TYPE_STRING : 
            name = GB_type_name_get (semiring->multiply->xtype) ;
            break ;

        case GrB_INP1_TYPE_STRING : 
            name = GB_type_name_get (semiring->multiply->ytype) ;
            break ;

        case GrB_OUTP_TYPE_STRING : 
            name = GB_type_name_get (semiring->multiply->ztype) ;
            break ;

        case GxB_MONOID_OPERATOR : 
        case GxB_SEMIRING_MULTIPLY : 
            (*value) = sizeof (GrB_BinaryOp) ;
            return (GrB_SUCCESS) ;

        case GxB_SEMIRING_MONOID : 
            (*value) = sizeof (GrB_Monoid) ;
            return (GrB_SUCCESS) ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    (*value) = (name == NULL) ? 1 : (strlen (name) + 1) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Semiring_get_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Semiring_get_VOID
(
    GrB_Semiring semiring,
    void * value,
    GrB_Field field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Semiring_get_VOID (semiring, value, field)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_SEMIRING_OK (semiring, "semiring to get option", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    switch ((int) field)
    {

        case GxB_MONOID_OPERATOR : 
            memcpy (value, &(semiring->add->op), sizeof (GrB_BinaryOp)) ;
            break ;

        case GxB_SEMIRING_MONOID : 
            memcpy (value, &(semiring->add), sizeof (GrB_Monoid)) ;
            break ;

        case GxB_SEMIRING_MULTIPLY : 
            memcpy (value, &(semiring->multiply), sizeof (GrB_BinaryOp)) ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

