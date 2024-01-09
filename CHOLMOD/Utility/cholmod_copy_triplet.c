//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_copy_triplet: copy a triplet matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_triplet) (&C, Common) ;    \
        return (NULL) ;                         \
    }

cholmod_triplet *CHOLMOD(copy_triplet)  // return new triplet matrix
(
    // input:
    cholmod_triplet *T,     // triplet matrix to copy
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_TRIPLET_MATRIX_INVALID (T, NULL) ;
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // allocate the copy
    //--------------------------------------------------------------------------

    cholmod_triplet *C = CHOLMOD(allocate_triplet) (T->nrow, T->ncol,
        T->nzmax, T->stype, T->xtype + T->dtype, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // get the sizes of the entries
    //--------------------------------------------------------------------------

    size_t ei = sizeof (Int) ;
    size_t e = (T->dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ex = e * ((T->xtype == CHOLMOD_PATTERN) ? 0 :
                    ((T->xtype == CHOLMOD_COMPLEX) ? 2 : 1)) ;
    size_t ez = e * ((T->xtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;

    //--------------------------------------------------------------------------
    // copy the contents from T to C
    //--------------------------------------------------------------------------

    C->nnz = T->nnz ;
    if (T->i != NULL) memcpy (C->i, T->i, T->nnz * ei) ;
    if (T->j != NULL) memcpy (C->j, T->j, T->nnz * ei) ;
    if (T->x != NULL) memcpy (C->x, T->x, T->nnz * ex) ;
    if (T->z != NULL) memcpy (C->z, T->z, T->nnz * ez) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_triplet) (C, "copy_triplet:C", Common)) ;
    return (C) ;
}

