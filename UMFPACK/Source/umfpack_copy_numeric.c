//------------------------------------------------------------------------------
// UMFPACK/Source/umfpack_copy_numeric: copy a Numeric object
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    User-callable.  Copy a Numeric object.
    Initial contribution by Will Kimmerer (MIT); revised by Tim Davis.
*/

#include "umf_internal.h"
#include "umf_valid_numeric.h"
#include "umf_malloc.h"
#include "umf_free.h"

#define COPY(component,type,n)                                              \
{                                                                           \
    Numeric->component = (type *) UMF_malloc (n, sizeof (type)) ;           \
    if (Numeric->component == (type *) NULL)                                \
    {                                                                       \
        UMFPACK_free_numeric ((void **) &Numeric) ;                         \
        return (UMFPACK_ERROR_out_of_memory) ;                              \
    }                                                                       \
    memcpy (Numeric->component, Original->component, (n) * sizeof(type)) ;  \
}

/* ========================================================================== */
/* === UMFPACK_copy_numeric ================================================= */
/* ========================================================================== */

int UMFPACK_copy_numeric
(
    void **NumericHandle,   // output: new copy of the input object
    void *NumericOriginal   // input: Numeric object to copy (not modified)
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (NumericHandle == NULL || NumericOriginal == NULL)
    {
        return (UMFPACK_ERROR_argument_missing) ;
    }

    NumericType *Numeric ;
    NumericType *Original = (NumericType *) NumericOriginal ;
    *NumericHandle = (void *) NULL ;

    if (!UMF_valid_numeric (Original))
    {
        return (UMFPACK_ERROR_invalid_Numeric_object) ;
    }

    //--------------------------------------------------------------------------
    // allocate and copy the header of the new Numeric object
    //--------------------------------------------------------------------------

    Numeric = (NumericType *) UMF_malloc (1, sizeof (NumericType)) ;
    if (Numeric == (NumericType *) NULL)
    {
        return (UMFPACK_ERROR_out_of_memory) ;
    }
    memcpy (Numeric, Original, sizeof (NumericType)) ;

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

    // umfpack_free_numeric can now be safely called if an error occurs

    //--------------------------------------------------------------------------
    // copy the rest of the Numeric object
    //--------------------------------------------------------------------------

    Int n_inner = MIN (Numeric->n_row, Numeric->n_col) ;
    COPY (D,     Entry, n_inner+1) ;
    COPY (Rperm, Int,   Numeric->n_row+1) ;
    COPY (Cperm, Int,   Numeric->n_col+1) ;
    COPY (Lpos,  Int,   Numeric->npiv+1) ;
    COPY (Lilen, Int,   Numeric->npiv+1) ;
    COPY (Lip,   Int,   Numeric->npiv+1) ;
    COPY (Upos,  Int,   Numeric->npiv+1) ;
    COPY (Uilen, Int,   Numeric->npiv+1) ;
    COPY (Uip,   Int,   Numeric->npiv+1) ;
    if (Numeric->scale != UMFPACK_SCALE_NONE)
    {
        COPY (Rs, double, Numeric->n_row) ;
    }
    if (Numeric->ulen > 0)
    {
        COPY (Upattern, Int, Numeric->ulen+1) ;
    }
    COPY (Memory, Unit, Numeric->size) ;

    // make sure the Numeric object is valid
    ASSERT (UMF_valid_numeric (Numeric)) ;

    // return the new Numeric object
    *NumericHandle = (void *) Numeric ;
    return (UMFPACK_OK) ;
}

