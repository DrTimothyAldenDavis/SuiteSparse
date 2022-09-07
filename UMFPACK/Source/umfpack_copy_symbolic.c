/* ========================================================================== */
/* === UMFPACK_copy_symbolic ================================================ */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Copy a Symbolic object.
*/

#include "umf_internal.h"
#include "umf_valid_symbolic.h"
#include "umf_malloc.h"
#include "umf_free.h"

#define COPY(object,original,type,n) \
{ \
    object = (type *) UMF_malloc (n, sizeof (type)) ; \
    if (object == (type *) NULL) \
    { \
	UMFPACK_free_symbolic ((void **) &Symbolic) ; \
	return (UMFPACK_ERROR_out_of_memory) ; \
    } \
    memcpy(object, original, n * sizeof(type)) ; \
}

/* ========================================================================== */
/* === UMFPACK_copy_symbolic ================================================ */
/* ========================================================================== */

GLOBAL Int UMFPACK_copy_symbolic
(
    void **SymbolicHandle,
    void *SymbolicOriginal
)
{
    SymbolicType *Symbolic ;
    SymbolicType *Original = (SymbolicType *) SymbolicOriginal ;
    *SymbolicHandle = (void *) NULL ;
    
    /* ---------------------------------------------------------------------- */
    /* read the Symbolic header from the buffer, in binary */
    /* ---------------------------------------------------------------------- */

    Symbolic = (SymbolicType *) UMF_malloc (1, sizeof (SymbolicType)) ;
    if (Symbolic == (SymbolicType *) NULL)
    {
	return (UMFPACK_ERROR_out_of_memory) ;
    }
    memcpy(Symbolic, Original, sizeof(SymbolicType)) ;

    /* @DrTimothyAldenDavis does this check need 
    to be done in the case of memcpy? */
    if (Symbolic->valid != SYMBOLIC_VALID || Symbolic->n_row <= 0 ||
	Symbolic->n_col <= 0 || Symbolic->nfr < 0 || Symbolic->nchains < 0 ||
	Symbolic->esize < 0)
    {
	/* Symbolic does not point to a Symbolic object */
	(void) UMF_free ((void *) Symbolic) ;
	return (UMFPACK_ERROR_invalid_Symbolic_object) ;
    }

    Symbolic->Cperm_init         = (Int *) NULL ;
    Symbolic->Rperm_init         = (Int *) NULL ;
    Symbolic->Front_npivcol      = (Int *) NULL ;
    Symbolic->Front_parent       = (Int *) NULL ;
    Symbolic->Front_1strow       = (Int *) NULL ;
    Symbolic->Front_leftmostdesc = (Int *) NULL ;
    Symbolic->Chain_start        = (Int *) NULL ;
    Symbolic->Chain_maxrows      = (Int *) NULL ;
    Symbolic->Chain_maxcols      = (Int *) NULL ;
    Symbolic->Cdeg               = (Int *) NULL ;
    Symbolic->Rdeg               = (Int *) NULL ;
    Symbolic->Esize              = (Int *) NULL ;
    Symbolic->Diagonal_map       = (Int *) NULL ;

    /* umfpack_free_symbolic can now be safely called if an error occurs */

    /* ---------------------------------------------------------------------- */
    /* read the rest of the Symbolic object */
    /* ---------------------------------------------------------------------- */

    COPY (Symbolic->Cperm_init,         Original->Cperm_init,         Int, Symbolic->n_col+1) ;
    COPY (Symbolic->Rperm_init,         Original->Rperm_init,         Int, Symbolic->n_row+1) ;
    COPY (Symbolic->Front_npivcol,      Original->Front_npivcol,      Int, Symbolic->nfr+1) ;
    COPY (Symbolic->Front_parent,       Original->Front_parent,       Int, Symbolic->nfr+1) ;
    COPY (Symbolic->Front_1strow,       Original->Front_1strow,       Int, Symbolic->nfr+1) ;
    COPY (Symbolic->Front_leftmostdesc, Original->Front_leftmostdesc, Int, Symbolic->nfr+1) ;
    COPY (Symbolic->Chain_start,        Original->Chain_start,        Int, Symbolic->nchains+1) ;
    COPY (Symbolic->Chain_maxrows,      Original->Chain_maxrows,      Int, Symbolic->nchains+1) ;
    COPY (Symbolic->Chain_maxcols,      Original->Chain_maxcols,      Int, Symbolic->nchains+1) ;
    COPY (Symbolic->Cdeg,               Original->Cdeg,               Int, Symbolic->n_col+1) ;
    COPY (Symbolic->Rdeg,               Original->Rdeg,               Int, Symbolic->n_row+1) ;
    if (Symbolic->esize > 0)
    {
	/* only when dense rows are present */
	COPY (Symbolic->Esize, Original->Esize, Int, Symbolic->esize) ;
    }
    if (Symbolic->prefer_diagonal)
    {
	/* only when diagonal pivoting is prefered */
	COPY (Symbolic->Diagonal_map, Original->Diagonal_map, Int, Symbolic->n_col+1) ;
    }

    /* make sure the Symbolic object is valid */
    if (!UMF_valid_symbolic (Symbolic))
    {
	UMFPACK_free_symbolic ((void **) &Symbolic) ;
	return (UMFPACK_ERROR_invalid_Symbolic_object) ;
    }

    *SymbolicHandle = (void *) Symbolic ;
    return (UMFPACK_OK) ;
}
