/* ========================================================================== */
/* === UMFPACK_serialize_symbolic ================================================ */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Saves a Symbolic object to a buffer.  It can later be read
    back in via a call to umfpack_*_deserialize_symbolic.
*/

#include "umf_internal.h"
#include "umf_valid_symbolic.h"

/* ========================================================================== */
/* === UMFPACK_serialize_symbolic =========================================== */
/* ========================================================================== */

GLOBAL Int UMFPACK_serialize_symbolic_size
(
    Int *size,
    void *SymbolicHandle
)
{
    SymbolicType *Symbolic ;
    Symbolic = (SymbolicType *) SymbolicHandle ;
    /* make sure the Symbolic object is valid */
    if (!UMF_valid_symbolic (Symbolic))
    {
	return (UMFPACK_ERROR_invalid_Symbolic_object) ;
    }
    Int isize = 0 ;
    isize += sizeof(SymbolicType) ;
    size_t intsize = sizeof(Int) ;
    isize += 2*(Symbolic->n_col+1) * intsize ;
    isize += 2*(Symbolic->n_row+1) * intsize ;
    isize += 4*(Symbolic->nfr+1) * intsize ;
    isize += 3*(Symbolic->nchains+1) * intsize ;
    if (Symbolic->esize > 0)
    {
    isize += Symbolic->esize * intsize ;
    }
    if (Symbolic->prefer_diagonal)
    {
        isize += (Symbolic->n_col+1) * intsize ;
    }
    *size = isize ;
    return (UMFPACK_OK) ;
}


#define SERIALIZE(object, type, n) \
{ \
    typesize = sizeof(type) ; \
    memcpy(buffer + offset, object, typesize * n) ; \
    offset += typesize * n ; \
}

GLOBAL Int UMFPACK_serialize_symbolic
(
    void *SymbolicHandle,
    void *buffer
)
{
    SymbolicType *Symbolic ;
    size_t typesize ;
    int offset = 0;
    /* get the Symbolic object */
    Symbolic = (SymbolicType *) SymbolicHandle ;

    /* make sure the Symbolic object is valid */
    if (!UMF_valid_symbolic (Symbolic))
    {
	return (UMFPACK_ERROR_invalid_Symbolic_object) ;
    }
    /* write the Symbolic object to the file, in binary */
    SERIALIZE (Symbolic,                     SymbolicType, 1) ;
    SERIALIZE (Symbolic->Cperm_init,         Int, Symbolic->n_col+1) ;
    SERIALIZE (Symbolic->Rperm_init,         Int, Symbolic->n_row+1) ;
    SERIALIZE (Symbolic->Front_npivcol,      Int, Symbolic->nfr+1) ;
    SERIALIZE (Symbolic->Front_parent,       Int, Symbolic->nfr+1) ;
    SERIALIZE (Symbolic->Front_1strow,       Int, Symbolic->nfr+1) ;
    SERIALIZE (Symbolic->Front_leftmostdesc, Int, Symbolic->nfr+1) ;
    SERIALIZE (Symbolic->Chain_start,        Int, Symbolic->nchains+1) ;
    SERIALIZE (Symbolic->Chain_maxrows,      Int, Symbolic->nchains+1) ;
    SERIALIZE (Symbolic->Chain_maxcols,      Int, Symbolic->nchains+1) ;
    SERIALIZE (Symbolic->Cdeg,               Int, Symbolic->n_col+1) ;
    SERIALIZE (Symbolic->Rdeg,               Int, Symbolic->n_row+1) ;
    if (Symbolic->esize > 0)
    {
	/* only when dense rows are present */
	SERIALIZE (Symbolic->Esize, Int, Symbolic->esize) ;
    }
    if (Symbolic->prefer_diagonal)
    {
	/* only when diagonal pivoting is prefered */
	SERIALIZE (Symbolic->Diagonal_map, Int, Symbolic->n_col+1) ;
    }

    return (UMFPACK_OK) ;
}
