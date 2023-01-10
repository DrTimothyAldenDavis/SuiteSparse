//------------------------------------------------------------------------------
// UMFPACK/Source/umfpack_serialize_numeric: serialize a Numeric object
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    User-callable.  Saves a Numeric object to a single int8_t array of bytes
    (the "blob").  It can later be read back to reconstruct the Numeric object
    via a call to umfpack_*_deserialize_numeric.
    Initial contribution by Will Kimmerer (MIT); revised by Tim Davis.
*/

#include "umf_internal.h"
#include "umf_valid_numeric.h"

//------------------------------------------------------------------------------
// UMFPACK_serialize_numeric_size: return size of blob for a Numeric object
//------------------------------------------------------------------------------

int UMFPACK_serialize_numeric_size
(
    int64_t *blobsize,          // output: required size of blob
    void *NumericHandle         // input: Numeric object to serialize
)
{

    // check inputs
    if (blobsize == NULL || NumericHandle == NULL)
    {
        return (UMFPACK_ERROR_argument_missing) ;
    }

    NumericType *Numeric ;
    Numeric = (NumericType *) NumericHandle ;
    (*blobsize) = 0 ;

    // make sure the Numeric object is valid
    if (!UMF_valid_numeric (Numeric))
    {
	return (UMFPACK_ERROR_invalid_Numeric_object) ;
    }

    // blob header
    (*blobsize) += sizeof (int64_t) + 10 * sizeof (int32_t) ;

    // Numeric header struct:
    (*blobsize) += sizeof (NumericType) ;

    // Lpos, Lilen, Lip, Upos, Uilen, and Uip:
    (*blobsize) += 6*(Numeric->npiv+1) * sizeof (Int) ;

    // Rperm and Cperm:
    (*blobsize) += (Numeric->n_row+1) * sizeof (Int) ;
    (*blobsize) += (Numeric->n_col+1) * sizeof (Int) ;

    // D:
    Int n_inner = MIN (Numeric->n_row, Numeric->n_col) ;
    (*blobsize) += (n_inner+1) * sizeof (Entry) ;

    // Rs, if present:
    if (Numeric->scale != UMFPACK_SCALE_NONE)
    {
        (*blobsize) += (Numeric->n_row) * sizeof (double) ;
    }
    // Upattern, if present:
    if (Numeric->ulen > 0)
    {
        (*blobsize) += (Numeric->ulen+1) * sizeof (Int) ;
    }

    // Numeric->Memory
    (*blobsize) += (Numeric->size) * sizeof (Unit) ;

    return (UMFPACK_OK) ;
}

//------------------------------------------------------------------------------
// UMFPACK_serialize_numeric: serialize a Numeric object into a blob
//------------------------------------------------------------------------------

#define SERIALIZE(object, type, n)                          \
{                                                           \
    memcpy (blob + offset, object, (n) * sizeof (type)) ;   \
    offset += (n) * sizeof (type) ;                         \
}

#define SERIALIZE_INT32(x)                                  \
{                                                           \
    int32_t scalar = (int32_t) (x) ;                        \
    SERIALIZE (&scalar, int32_t, 1) ;                       \
}

int UMFPACK_serialize_numeric
(
    int8_t *blob,           // output: serialized blob of size blobsize,
                            // allocated but unitialized on input.
    int64_t blobsize,       // input: size of the blob
    void *NumericHandle     // input: Numeric object to serialize
)
{

    // determine the required size of the blob
    if (blob == NULL)
    {
        return (UMFPACK_ERROR_argument_missing) ;
    }
    int64_t required ;
    int status = UMFPACK_serialize_numeric_size (&required, NumericHandle) ;
    if (status != UMFPACK_OK)
    {
        // Numeric object is invalid or NULL
	return (status) ;
    }
    if (required > blobsize)
    {
        // blob is not large enough
        return (UMFPACK_ERROR_invalid_blob) ;
    }

    // get the Numeric object
    NumericType *Numeric = (NumericType *) NumericHandle ;

    // write the blob header:
    int64_t offset = 0 ;
    SERIALIZE (&required, int64_t, 1) ;         // required size of this blob
    SERIALIZE_INT32 (NUMERIC_VALID) ;           // tag as a Numeric object
    SERIALIZE_INT32 (UMFPACK_MAIN_VERSION) ;    // gaurd against version changes
    SERIALIZE_INT32 (UMFPACK_SUB_VERSION) ;
    SERIALIZE_INT32 (UMFPACK_SUBSUB_VERSION) ;
    SERIALIZE_INT32 (sizeof (NumericType)) ;    // size of Numeric header
    SERIALIZE_INT32 (sizeof (Entry)) ;          // double or double complex
    SERIALIZE_INT32 (sizeof (Int)) ;            // Int is int32_t or int64_t
    SERIALIZE_INT32 (sizeof (Unit)) ;
    SERIALIZE_INT32 (sizeof (double)) ;
    SERIALIZE_INT32 (sizeof (void *)) ;         // 32-bit vs 64-bit OS

    // write the Numeric object to the blob
    Int n_inner = MIN (Numeric->n_row, Numeric->n_col) ;
    SERIALIZE (Numeric,             NumericType, 1) ;
    SERIALIZE (Numeric->D,          Entry, n_inner+1) ;
    SERIALIZE (Numeric->Rperm,      Int, Numeric->n_row+1) ;
    SERIALIZE (Numeric->Cperm,      Int, Numeric->n_col+1) ;
    SERIALIZE (Numeric->Lpos,       Int, Numeric->npiv+1) ;
    SERIALIZE (Numeric->Lilen,      Int, Numeric->npiv+1) ;
    SERIALIZE (Numeric->Lip,        Int, Numeric->npiv+1) ;
    SERIALIZE (Numeric->Upos,       Int, Numeric->npiv+1) ;
    SERIALIZE (Numeric->Uilen,      Int, Numeric->npiv+1) ;
    SERIALIZE (Numeric->Uip,        Int, Numeric->npiv+1) ;
    if (Numeric->scale != UMFPACK_SCALE_NONE)
    {
	/* only when dense rows are present */
	SERIALIZE (Numeric->Rs, double, Numeric->n_row) ;
    }
    if (Numeric->ulen > 0)
    {
	/* only when diagonal pivoting is prefered */
	SERIALIZE (Numeric->Upattern, Int, Numeric->ulen+1) ;
    }
    /* It is possible that some parts of Numeric->Memory are
       unitialized and unused; this is OK, but it can generate
       a valgrind warning. */
    SERIALIZE (Numeric->Memory, Unit, Numeric->size) ;

    return (UMFPACK_OK) ;
}

