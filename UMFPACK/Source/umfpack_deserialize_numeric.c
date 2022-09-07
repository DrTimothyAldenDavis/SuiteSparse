/* ========================================================================== */
/* === UMFPACK_deserialize_numeric ========================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Loads a Numeric object from a buffer created by
    umfpack_*_serialize_numeric
*/

#include "umf_internal.h"
#include "umf_valid_numeric.h"
#include "umf_malloc.h"
#include "umf_free.h"

#define DESERIALIZE(object,type,n) \
{ \
    object = (type *) UMF_malloc (n, sizeof (type)) ; \
    if (object == (type *) NULL) \
    { \
	UMFPACK_free_numeric ((void **) &Numeric) ; \
	return (UMFPACK_ERROR_out_of_memory) ; \
    } \
    memcpy(object, buffer+offset, n * sizeof(type)) ; \
    offset += n * sizeof(type) ; \
}

/* ========================================================================== */
/* === UMFPACK_deserialize_numeric ========================================== */
/* ========================================================================== */

GLOBAL Int UMFPACK_deserialize_numeric
(
    void **NumericHandle,
    char *buffer
)
{
    NumericType *Numeric ;

    *NumericHandle = (void *) NULL ;
    size_t offset = 0 ;
    /* ---------------------------------------------------------------------- */
    /* read the Numeric header from the buffer, in binary */
    /* ---------------------------------------------------------------------- */

    Numeric = (NumericType *) UMF_malloc (1, sizeof (NumericType)) ;
    if (Numeric == (NumericType *) NULL)
    {
	return (UMFPACK_ERROR_out_of_memory) ;
    }
    memcpy(Numeric, buffer, sizeof(NumericType)) ;
    offset += sizeof(NumericType) ;

    if (Numeric->valid != NUMERIC_VALID || Numeric->n_row <= 0 ||
	Numeric->n_col <= 0 || Numeric->npiv < 0 || Numeric->ulen < 0 ||
	Numeric->size < 0)
    {
	/* Numeric does not point to a Numeric object */
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

    /* umfpack_free_numeric can now be safely called if an error occurs */

    /* ---------------------------------------------------------------------- */
    /* read the rest of the Numeric object */
    /* ---------------------------------------------------------------------- */

    DESERIALIZE (Numeric->D,     Entry, MIN (Numeric->n_row, Numeric->n_col)+1) ;
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

    /* make sure the Numeric object is valid */
    if (!UMF_valid_numeric (Numeric))
    {
	UMFPACK_free_numeric ((void **) &Numeric) ;
	return (UMFPACK_ERROR_invalid_Numeric_object) ;
    }

    *NumericHandle = (void *) Numeric ;
    return (UMFPACK_OK) ;
}
