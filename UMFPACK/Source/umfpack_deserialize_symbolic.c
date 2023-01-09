/* ========================================================================== */
/* === UMFPACK_deserialize_symbolic ========================================= */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Loads a Symbolic object from a buffer created by
    umfpack_*_serialize_symbolic.
*/

#include "umf_internal.h"
#include "umf_valid_symbolic.h"
#include "umf_malloc.h"
#include "umf_free.h"

#define DESERIALIZE(object,type,n) \
{ \
    object = (type *) UMF_malloc (n, sizeof (type)) ; \
    if (object == (type *) NULL) \
    { \
	UMFPACK_free_symbolic ((void **) &Symbolic) ; \
	return (UMFPACK_ERROR_out_of_memory) ; \
    } \
    memcpy(object, buffer+offset, n * sizeof(type)) ; \
    offset += n * sizeof(type) ; \
}

/* ========================================================================== */
/* === UMFPACK_deserialize_symbolic ========================================= */
/* ========================================================================== */

GLOBAL Int UMFPACK_deserialize_symbolic
(
    void **SymbolicHandle,
    char *buffer
)
{
    SymbolicType *Symbolic ;

    *SymbolicHandle = (void *) NULL ;
    size_t offset = 0;
    /* ---------------------------------------------------------------------- */
    /* read the Symbolic header from the buffer, in binary */
    /* ---------------------------------------------------------------------- */

    Symbolic = (SymbolicType *) UMF_malloc (1, sizeof (SymbolicType)) ;
    if (Symbolic == (SymbolicType *) NULL)
    {
	return (UMFPACK_ERROR_out_of_memory) ;
    }
    memcpy(Symbolic, buffer, sizeof(SymbolicType)) ;
    offset += sizeof(SymbolicType) ;

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

    DESERIALIZE (Symbolic->Cperm_init,         Int, Symbolic->n_col+1) ;
    DESERIALIZE (Symbolic->Rperm_init,         Int, Symbolic->n_row+1) ;
    DESERIALIZE (Symbolic->Front_npivcol,      Int, Symbolic->nfr+1) ;
    DESERIALIZE (Symbolic->Front_parent,       Int, Symbolic->nfr+1) ;
    DESERIALIZE (Symbolic->Front_1strow,       Int, Symbolic->nfr+1) ;
    DESERIALIZE (Symbolic->Front_leftmostdesc, Int, Symbolic->nfr+1) ;
    DESERIALIZE (Symbolic->Chain_start,        Int, Symbolic->nchains+1) ;
    DESERIALIZE (Symbolic->Chain_maxrows,      Int, Symbolic->nchains+1) ;
    DESERIALIZE (Symbolic->Chain_maxcols,      Int, Symbolic->nchains+1) ;
    DESERIALIZE (Symbolic->Cdeg,               Int, Symbolic->n_col+1) ;
    DESERIALIZE (Symbolic->Rdeg,               Int, Symbolic->n_row+1) ;
    if (Symbolic->esize > 0)
    {
	/* only when dense rows are present */
	DESERIALIZE (Symbolic->Esize, Int, Symbolic->esize) ;
    }
    if (Symbolic->prefer_diagonal)
    {
	/* only when diagonal pivoting is prefered */
	DESERIALIZE (Symbolic->Diagonal_map, Int, Symbolic->n_col+1) ;
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
