//------------------------------------------------------------------------------
// UMFPACK/Source/umfpack_deserialize_numeric: deserialize a Numeric object
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    User-callable.  Loads a Numeric object from a serialized blob created by
    umfpack_*_serialize_numeric.
    Initial contribution by Will Kimmerer (MIT); revised by Tim Davis.
*/

#include "umf_internal.h"
#include "umf_valid_numeric.h"
#include "umf_malloc.h"
#include "umf_free.h"

// get a component of the Numeric object from the blob
#define DESERIALIZE(object,type,n)                          \
{                                                           \
    object = (type *) UMF_malloc (n, sizeof (type)) ;       \
    if (object == (type *) NULL)                            \
    {                                                       \
	UMFPACK_free_numeric ((void **) &Numeric) ;         \
	return (UMFPACK_ERROR_out_of_memory) ;              \
    }                                                       \
    memcpy (object, blob + offset, (n) * sizeof (type)) ;   \
    offset += (n) * sizeof (type) ;                         \
}

// get a single scalar from the blob
#define DESERIALIZE_SCALAR(type,scalar)                     \
    type scalar = 0 ;                                       \
    if (offset + sizeof (type) > blobsize)                  \
    {                                                       \
        return (UMFPACK_ERROR_invalid_blob) ;               \
    }                                                       \
    memcpy (&scalar, blob + offset, sizeof (type)) ;        \
    offset += sizeof (type) ;                               \

//==============================================================================
//==== UMFPACK_deserialize_numeric =============================================
//==============================================================================

int UMFPACK_deserialize_numeric
(
    void **NumericHandle,   // output: Numeric object created from the blob
    int8_t *blob,           // input: serialized blob, not modified
    int64_t blobsize        // size of the blob in bytes
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (NumericHandle == NULL || blob == NULL)
    {
        return (UMFPACK_ERROR_argument_missing) ;
    }

    NumericType *Numeric ;
    (*NumericHandle) = (void *) NULL ;
    int64_t offset = 0 ;

    // read the blob header:
    DESERIALIZE_SCALAR (int64_t, required) ;
    DESERIALIZE_SCALAR (int32_t, valid) ;
    DESERIALIZE_SCALAR (int32_t, version_main) ;
    DESERIALIZE_SCALAR (int32_t, version_sub) ;
    DESERIALIZE_SCALAR (int32_t, version_subsub) ;
    DESERIALIZE_SCALAR (int32_t, sizeof_Numeric) ;
    DESERIALIZE_SCALAR (int32_t, sizeof_Entry) ;
    DESERIALIZE_SCALAR (int32_t, sizeof_Int) ;
    DESERIALIZE_SCALAR (int32_t, sizeof_Unit) ;
    DESERIALIZE_SCALAR (int32_t, sizeof_double) ;
    DESERIALIZE_SCALAR (int32_t, sizeof_void_star) ;

    if (required > blobsize || valid != NUMERIC_VALID
        || sizeof_Numeric != sizeof (NumericType)
        || sizeof_Entry != sizeof (Entry)
        || sizeof_Int != sizeof (Int)
        || sizeof_Unit != sizeof (Unit)
        || sizeof_double != sizeof (double)
        || sizeof_void_star != sizeof (void *))
    {
        return (UMFPACK_ERROR_invalid_blob) ;
    }

    //--------------------------------------------------------------------------
    // read the Numeric header from the blob
    //--------------------------------------------------------------------------

    Numeric = (NumericType *) UMF_malloc (1, sizeof (NumericType)) ;
    if (Numeric == (NumericType *) NULL)
    {
	return (UMFPACK_ERROR_out_of_memory) ;
    }
    memcpy (Numeric, blob + offset, sizeof (NumericType)) ;
    offset += sizeof (NumericType) ;

    if (Numeric->valid != NUMERIC_VALID || Numeric->n_row <= 0 ||
	Numeric->n_col <= 0 || Numeric->npiv < 0 || Numeric->ulen < 0 ||
	Numeric->size < 0)
    {
	// Numeric does not contain a valid Numeric object
	(void) UMF_free ((void *) Numeric) ;
	return (UMFPACK_ERROR_invalid_Numeric_object) ;
    }

    Numeric->D        = (Entry *) NULL ;
    Numeric->Rperm    = (Int *) NULL ;
    Numeric->Cperm    = (Int *) NULL ;
    Numeric->Lpos     = (Int *) NULL ;
    Numeric->Lilen    = (Int *) NULL ;
    Numeric->Lip      = (Int *) NULL ;
    Numeric->Upos     = (Int *) NULL ;
    Numeric->Uilen    = (Int *) NULL ;
    Numeric->Uip      = (Int *) NULL ;
    Numeric->Rs       = (double *) NULL ;
    Numeric->Memory   = (Unit *) NULL ;
    Numeric->Upattern = (Int *) NULL ;

    // UMFPACK_free_numeric can now be safely called if an error occurs

    //--------------------------------------------------------------------------
    // read the rest of the Numeric object
    //--------------------------------------------------------------------------

    Int n_inner = MIN (Numeric->n_row, Numeric->n_col) ;
    DESERIALIZE (Numeric->D,     Entry, n_inner+1) ;
    DESERIALIZE (Numeric->Rperm, Int,   Numeric->n_row+1) ;
    DESERIALIZE (Numeric->Cperm, Int,   Numeric->n_col+1) ;
    DESERIALIZE (Numeric->Lpos,  Int,   Numeric->npiv+1) ;
    DESERIALIZE (Numeric->Lilen, Int,   Numeric->npiv+1) ;
    DESERIALIZE (Numeric->Lip,   Int,   Numeric->npiv+1) ;
    DESERIALIZE (Numeric->Upos,  Int,   Numeric->npiv+1) ;
    DESERIALIZE (Numeric->Uilen, Int,   Numeric->npiv+1) ;
    DESERIALIZE (Numeric->Uip,   Int,   Numeric->npiv+1) ;
    if (Numeric->scale != UMFPACK_SCALE_NONE)
    {
	DESERIALIZE (Numeric->Rs, double, Numeric->n_row) ;
    }
    if (Numeric->ulen > 0)
    {
	DESERIALIZE (Numeric->Upattern, Int, Numeric->ulen+1) ;
    }
    DESERIALIZE (Numeric->Memory, Unit, Numeric->size) ;

    // make sure the Numeric object is valid
    ASSERT (UMF_valid_numeric (Numeric)) ;

    // return new Numeric object
    (*NumericHandle) = (void *) Numeric ;
    return (UMFPACK_OK) ;
}

