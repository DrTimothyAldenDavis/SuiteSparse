/* ========================================================================== */
/* === UMFPACK_serialize_numeric ============================================ */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Saves a Numeric object to a buffer.  It can later be read
    back in via a call to umfpack_*_deserialize_numeric.
*/

#include "umf_internal.h"
#include "umf_valid_numeric.h"

/* ========================================================================== */
/* === UMFPACK_serialize_numeric =========================================== */
/* ========================================================================== */

GLOBAL Int UMFPACK_serialize_numeric_size
(
    Int *size,
    void *NumericHandle
)
{
    Int isize = 0 ;
    NumericType *Numeric ;
    Numeric = (NumericType *) NumericHandle ;
    /* make sure the Numeric object is valid */
    if (!UMF_valid_numeric (Numeric))
    {
	return (UMFPACK_ERROR_invalid_Numeric_object) ;
    }
    isize += sizeof(NumericType) ;
    size_t intsize = sizeof(Int) ;
    isize += 6*(Numeric->npiv) * intsize ;
    isize += (Numeric->n_row+1) * intsize ;
    isize += (Numeric->n_col+1) * intsize ;
    isize += (MIN (Numeric->n_row, Numeric->n_col)+1) * sizeof (Entry) ;
    if (Numeric->scale != UMFPACK_SCALE_NONE)
    {
        isize += (Numeric->n_row) * intsize ;
    }
    if (Numeric->ulen > 0)
    {
        isize += (Numeric->ulen+1) * intsize ;
    }
    isize += (Numeric->size) * sizeof (Unit) ;
    *size = isize ;
    return (UMFPACK_OK) ;
}


#define SERIALIZE(object, type, n) \
{ \
    typesize = sizeof(type) ; \
    memcpy(buffer + offset, object, typesize * n) ; \
    offset += typesize * n ; \
}

GLOBAL Int UMFPACK_serialize_numeric
(
    void *NumericHandle,
    void *buffer
)
{
    NumericType *Numeric ;
    size_t typesize ;
    int offset = 0;
    /* get the Numeric object */
    Numeric = (NumericType *) NumericHandle ;

    /* make sure the Numeric object is valid */
    if (!UMF_valid_numeric (Numeric))
    {
	return (UMFPACK_ERROR_invalid_Numeric_object) ;
    }
    /* write the Numeric object to the file, in binary */
    SERIALIZE (Numeric,             NumericType, 1) ;
    SERIALIZE (Numeric->D,          Entry, MIN (Numeric->n_row, Numeric->n_col)+1) ;
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
    SERIALIZE(Numeric->Memory, Unit, Numeric->size) ;

    return (UMFPACK_OK) ;
}
