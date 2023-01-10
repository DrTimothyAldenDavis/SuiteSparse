//------------------------------------------------------------------------------
// UMFPACK/Source/umfpack_deserialize_symbolic: deserialize a Symbolic object
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    User-callable.  Loads a Symbolic object from a serialized blob created by
    umfpack_*_serialize_symbolic.
    Initial contribution by Will Kimmerer (MIT); revised by Tim Davis.
*/

#include "umf_internal.h"
#include "umf_valid_symbolic.h"
#include "umf_malloc.h"
#include "umf_free.h"

// get a component of the Symbolic object from the blob
#define DESERIALIZE(object,type,n)                          \
{                                                           \
    object = (type *) UMF_malloc (n, sizeof (type)) ;       \
    if (object == (type *) NULL)                            \
    {                                                       \
	UMFPACK_free_symbolic ((void **) &Symbolic) ;       \
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
//=== UMFPACK_deserialize_symbolic =============================================
//==============================================================================

int UMFPACK_deserialize_symbolic
(
    void **SymbolicHandle,  // output: Symbolic object created from the blob
    int8_t *blob,           // input: serialized blob, not modified
    int64_t blobsize        // size of the blob in bytes
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (SymbolicHandle == NULL || blob == NULL)
    {
        return (UMFPACK_ERROR_argument_missing) ;
    }

    SymbolicType *Symbolic ;
    (*SymbolicHandle) = (void *) NULL ;
    int64_t offset = 0 ;

    // read the blob header:
    DESERIALIZE_SCALAR (int64_t, required) ;
    DESERIALIZE_SCALAR (int32_t, valid) ;
    DESERIALIZE_SCALAR (int32_t, version_main) ;
    DESERIALIZE_SCALAR (int32_t, version_sub) ;
    DESERIALIZE_SCALAR (int32_t, version_subsub) ;
    DESERIALIZE_SCALAR (int32_t, sizeof_Symbolic) ;
    DESERIALIZE_SCALAR (int32_t, sizeof_Entry) ;
    DESERIALIZE_SCALAR (int32_t, sizeof_Int) ;
    DESERIALIZE_SCALAR (int32_t, sizeof_Unit) ;
    DESERIALIZE_SCALAR (int32_t, sizeof_double) ;
    DESERIALIZE_SCALAR (int32_t, sizeof_void_star) ;

    if (required > blobsize || valid != SYMBOLIC_VALID
        || sizeof_Symbolic != sizeof (SymbolicType)
        || sizeof_Entry != sizeof (Entry)
        || sizeof_Int != sizeof (Int)
        || sizeof_Unit != sizeof (Unit)
        || sizeof_double != sizeof (double)
        || sizeof_void_star != sizeof (void *))
    {
        return (UMFPACK_ERROR_invalid_blob) ;
    }

    //--------------------------------------------------------------------------
    // read the Symbolic header from the blob
    //--------------------------------------------------------------------------

    Symbolic = (SymbolicType *) UMF_malloc (1, sizeof (SymbolicType)) ;
    if (Symbolic == (SymbolicType *) NULL)
    {
	return (UMFPACK_ERROR_out_of_memory) ;
    }
    memcpy (Symbolic, blob + offset, sizeof (SymbolicType)) ;
    offset += sizeof (SymbolicType) ;

    if (Symbolic->valid != SYMBOLIC_VALID || Symbolic->n_row <= 0 ||
	Symbolic->n_col <= 0 || Symbolic->nfr < 0 || Symbolic->nchains < 0 ||
	Symbolic->esize < 0)
    {
	// Symbolic does not point to a Symbolic object
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

    // umfpack_free_symbolic can now be safely called if an error occurs

    //--------------------------------------------------------------------------
    // read the rest of the Symbolic object
    //--------------------------------------------------------------------------

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
	// only when dense rows are present
	DESERIALIZE (Symbolic->Esize, Int, Symbolic->esize) ;
    }
    if (Symbolic->prefer_diagonal)
    {
	// only when diagonal pivoting is prefered
	DESERIALIZE (Symbolic->Diagonal_map, Int, Symbolic->n_col+1) ;
    }

    // make sure the Symbolic object is valid
    ASSERT (UMF_valid_symbolic (Symbolic)) ;

    // return new Symbolic object
    (*SymbolicHandle) = (void *) Symbolic ;
    return (UMFPACK_OK) ;
}

