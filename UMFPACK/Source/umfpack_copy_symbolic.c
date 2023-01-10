//------------------------------------------------------------------------------
// UMFPACK/Source/umfpack_copy_symbolic: copy a Symbolic object
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    User-callable.  Copy a Symbolic object.
    Initial contribution by Will Kimmerer (MIT); revised by Tim Davis.
*/

#include "umf_internal.h"
#include "umf_valid_symbolic.h"
#include "umf_malloc.h"
#include "umf_free.h"

#define COPY(component,type,n)                                              \
{                                                                           \
    Symbolic->component = (type *) UMF_malloc (n, sizeof (type)) ;          \
    if (Symbolic->component == (type *) NULL)                               \
    {                                                                       \
        UMFPACK_free_symbolic ((void **) &Symbolic) ;                       \
        return (UMFPACK_ERROR_out_of_memory) ;                              \
    }                                                                       \
    memcpy (Symbolic->component, Original->component, (n) * sizeof(type)) ; \
}

/* ========================================================================== */
/* === UMFPACK_copy_symbolic ================================================ */
/* ========================================================================== */

int UMFPACK_copy_symbolic
(
    void **SymbolicHandle,  // output: new copy of the input object
    void *SymbolicOriginal  // input: Symbolic object to copy (not modified)
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (SymbolicHandle == NULL || SymbolicOriginal == NULL)
    {
        return (UMFPACK_ERROR_argument_missing) ;
    }

    SymbolicType *Symbolic ;
    SymbolicType *Original = (SymbolicType *) SymbolicOriginal ;
    (*SymbolicHandle) = (void *) NULL ;

    if (!UMF_valid_symbolic (Original))
    {
        return (UMFPACK_ERROR_invalid_Symbolic_object) ;
    }

    //--------------------------------------------------------------------------
    // allocate and copy the header of the new Symbolic object
    //--------------------------------------------------------------------------

    Symbolic = (SymbolicType *) UMF_malloc (1, sizeof (SymbolicType)) ;
    if (Symbolic == (SymbolicType *) NULL)
    {
        return (UMFPACK_ERROR_out_of_memory) ;
    }
    memcpy (Symbolic, Original, sizeof (SymbolicType)) ;

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
    // copy the rest of the Symbolic object
    //--------------------------------------------------------------------------

    COPY (Cperm_init,         Int, Symbolic->n_col+1) ;
    COPY (Rperm_init,         Int, Symbolic->n_row+1) ;
    COPY (Front_npivcol,      Int, Symbolic->nfr+1) ;
    COPY (Front_parent,       Int, Symbolic->nfr+1) ;
    COPY (Front_1strow,       Int, Symbolic->nfr+1) ;
    COPY (Front_leftmostdesc, Int, Symbolic->nfr+1) ;
    COPY (Chain_start,        Int, Symbolic->nchains+1) ;
    COPY (Chain_maxrows,      Int, Symbolic->nchains+1) ;
    COPY (Chain_maxcols,      Int, Symbolic->nchains+1) ;
    COPY (Cdeg,               Int, Symbolic->n_col+1) ;
    COPY (Rdeg,               Int, Symbolic->n_row+1) ;
    if (Symbolic->esize > 0)
    {
        // only when dense rows are present
        COPY (Esize, Int, Symbolic->esize) ;
    }
    if (Symbolic->prefer_diagonal)
    {
        // only when diagonal pivoting is prefered
        COPY (Diagonal_map, Int, Symbolic->n_col+1) ;
    }

    // make sure the Symbolic object is valid
    ASSERT (UMF_valid_symbolic (Symbolic)) ;

    // return the new Symbolic object
    (*SymbolicHandle) = (void *) Symbolic ;
    return (UMFPACK_OK) ;
}
