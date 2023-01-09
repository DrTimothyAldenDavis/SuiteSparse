/* ========================================================================== */
/* === UMFPACK_copy_numeric ================================================= */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Copy a Numeric object.
*/

#include "umf_internal.h"
#include "umf_valid_numeric.h"
#include "umf_malloc.h"
#include "umf_free.h"

#define COPY(object,original,type,n) \
{ \
    object = (type *) UMF_malloc (n, sizeof (type)) ; \
    if (object == (type *) NULL) \
    { \
	UMFPACK_free_numeric ((void **) &Numeric) ; \
	return (UMFPACK_ERROR_out_of_memory) ; \
    } \
    memcpy(object, original, n * sizeof(type)) ; \
}

/* ========================================================================== */
/* === UMFPACK_copy_numeric ================================================= */
/* ========================================================================== */

GLOBAL Int UMFPACK_copy_numeric
(
    void **NumericHandle,
    void *NumericOriginal
)
{
    NumericType *Numeric ;
    NumericType *Original = (NumericType *) NumericOriginal ;
    *NumericHandle = (void *) NULL ;
    
    /* ---------------------------------------------------------------------- */
    /* read the Numeric header from the buffer, in binary */
    /* ---------------------------------------------------------------------- */

    Numeric = (NumericType *) UMF_malloc (1, sizeof (NumericType)) ;
    if (Numeric == (NumericType *) NULL)
    {
	return (UMFPACK_ERROR_out_of_memory) ;
    }
    memcpy(Numeric, Original, sizeof(NumericType)) ;

    /* @DrTimothyAldenDavis does this check need 
    to be done in the case of memcpy? */
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

    COPY (Numeric->D,     Original->D,     Entry, 
        MIN (Numeric->n_row, Numeric->n_col)+1) ;
    COPY (Numeric->Rperm, Original->Rperm, Int,   Numeric->n_row+1) ;
    COPY (Numeric->Cperm, Original->Cperm, Int,   Numeric->n_col+1) ;
    COPY (Numeric->Lpos,  Original->Lpos,  Int,   Numeric->npiv+1) ;
    COPY (Numeric->Lilen, Original->Lilen, Int,   Numeric->npiv+1) ;
    COPY (Numeric->Lip,   Original->Lip,   Int,   Numeric->npiv+1) ;
    COPY (Numeric->Upos,  Original->Upos,  Int,   Numeric->npiv+1) ;
    COPY (Numeric->Uilen, Original->Uilen, Int,   Numeric->npiv+1) ;
    COPY (Numeric->Uip,   Original->Uip,   Int,   Numeric->npiv+1) ;
    if (Numeric->scale != UMFPACK_SCALE_NONE)
    {
	COPY (Numeric->Rs, Original->Rs, double, Numeric->n_row) ;
    }
    if (Numeric->ulen > 0)
    {
	COPY (Numeric->Upattern, Original->Upattern, Int, Numeric->ulen+1) ;
    }
    COPY (Numeric->Memory, Original->Memory, Unit, Numeric->size) ;

    /* make sure the Numeric object is valid */
    if (!UMF_valid_numeric (Numeric))
    {
	UMFPACK_free_numeric ((void **) &Numeric) ;
	return (UMFPACK_ERROR_invalid_Numeric_object) ;
    }

    *NumericHandle = (void *) Numeric ;
    return (UMFPACK_OK) ;
}
