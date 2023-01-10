//------------------------------------------------------------------------------
// UMFPACK/Source/umfpack_serialize_symbolic: serialize a Symbolic object
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    User-callable.  Saves a Symbolic object to a single int8_t array of bytes
    (the "blob").  It can later be read back to reconstruct the Symbolic object
    via a call to umfpack_*_deserialize_symbolic.
    Initial contribution by Will Kimmerer (MIT); revised by Tim Davis.
*/

#include "umf_internal.h"
#include "umf_valid_symbolic.h"

//------------------------------------------------------------------------------
// UMFPACK_serialize_symbolic_size: return size of blob for a Symbolic object
//------------------------------------------------------------------------------

int UMFPACK_serialize_symbolic_size
(
    int64_t *blobsize,          // output: required size of blob
    void *SymbolicHandle        // input: Symbolic object to serialize
)
{

    // check inputs
    if (blobsize == NULL || SymbolicHandle == NULL)
    {
        return (UMFPACK_ERROR_argument_missing) ;
    }

    SymbolicType *Symbolic ;
    Symbolic = (SymbolicType *) SymbolicHandle ;
    (*blobsize) = 0 ;

    // make sure the Symbolic object is valid
    if (!UMF_valid_symbolic (Symbolic))
    {
        return (UMFPACK_ERROR_invalid_Symbolic_object) ;
    }

    // blob header
    (*blobsize) += sizeof (int64_t) + 10 * sizeof (int32_t) ;

    // Symbolic header struct:
    (*blobsize) += sizeof (SymbolicType) ;

    // Rperm_init, Cperm_init, Rdeg, Cdeg:
    (*blobsize) += 2 * (Symbolic->n_row+1) * sizeof (Int) ;
    (*blobsize) += 2 * (Symbolic->n_col+1) * sizeof (Int) ;

    // Front_*
    (*blobsize) += 4*(Symbolic->nfr+1) * sizeof (Int) ;

    // Chain_*
    (*blobsize) += 3*(Symbolic->nchains+1) * sizeof (Int) ;

    // Esize, if present:
    if (Symbolic->esize > 0)
    {
        (*blobsize) += (Symbolic->esize) * sizeof (Int) ;
    }
    // Diagonal_map, if present:
    if (Symbolic->prefer_diagonal)
    {
        (*blobsize) += (Symbolic->n_col+1) * sizeof (Int) ;
    }

    return (UMFPACK_OK) ;
}

//------------------------------------------------------------------------------
// UMFPACK_serialize_symbolic: serialize a Symbolic object into a blob
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

int UMFPACK_serialize_symbolic
(
    int8_t *blob,           // output: serialized blob of size blobsize,
                            // allocated but unitialized on input.
    int64_t blobsize,       // input: size of the blob
    void *SymbolicHandle    // input: Symbolic object to serialize
)
{

    // determine the required size of the blob
    if (blob == NULL)
    {
        return (UMFPACK_ERROR_argument_missing) ;
    }
    int64_t required ;
    int status = UMFPACK_serialize_symbolic_size (&required, SymbolicHandle) ;
    if (status != UMFPACK_OK)
    {
        // Symbolic object is invalid or NULL
        return (status) ;
    }
    if (required > blobsize)
    {
        // blob is not large enough
        return (UMFPACK_ERROR_invalid_blob) ;
    }

    // get the Symbolic object
    SymbolicType *Symbolic = (SymbolicType *) SymbolicHandle ;

    // write the blob header:
    int64_t offset = 0 ;
    SERIALIZE (&required, int64_t, 1) ;         // required size of this blob
    SERIALIZE_INT32 (SYMBOLIC_VALID) ;          // tag as a Symbolic object
    SERIALIZE_INT32 (UMFPACK_MAIN_VERSION) ;    // gaurd against version changes
    SERIALIZE_INT32 (UMFPACK_SUB_VERSION) ;
    SERIALIZE_INT32 (UMFPACK_SUBSUB_VERSION) ;
    SERIALIZE_INT32 (sizeof (SymbolicType)) ;   // size of Symbolic header
    SERIALIZE_INT32 (sizeof (Entry)) ;          // double or double complex
    SERIALIZE_INT32 (sizeof (Int)) ;            // Int is int32_t or int64_t
    SERIALIZE_INT32 (sizeof (Unit)) ;
    SERIALIZE_INT32 (sizeof (double)) ;
    SERIALIZE_INT32 (sizeof (void *)) ;         // 32-bit vs 64-bit OS

    // write the Symbolic object to the blob
    Int n_inner = MIN (Symbolic->n_row, Symbolic->n_col) ;
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
        // only when dense rows are present
        SERIALIZE (Symbolic->Esize, Int, Symbolic->esize) ;
    }
    if (Symbolic->prefer_diagonal)
    {
        // only when diagonal pivoting is prefered
        SERIALIZE (Symbolic->Diagonal_map, Int, Symbolic->n_col+1) ;
    }

    return (UMFPACK_OK) ;
}

