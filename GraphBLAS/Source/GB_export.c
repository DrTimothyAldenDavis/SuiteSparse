//------------------------------------------------------------------------------
// GB_export: export a matrix or vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// No conversion is done, and the matrix is exported in its current sparsity
// structure and by-row/by-col format.

#include "GB_export.h"

GrB_Info GB_export      // export a matrix in any format
(
    GrB_Matrix *A,      // handle of matrix to export and free
    GrB_Type *type,     // type of matrix to export
    GrB_Index *vlen,    // vector length
    GrB_Index *vdim,    // vector dimension
    bool is_sparse_vector,      // true if A is a sparse GrB_Vector

    // the 5 arrays:
    GrB_Index **Ap,     // pointers
    GrB_Index *Ap_size, // size of Ap in bytes

    GrB_Index **Ah,     // vector indices
    GrB_Index *Ah_size, // size of Ah in bytes

    int8_t **Ab,        // bitmap
    GrB_Index *Ab_size, // size of Ab in bytes

    GrB_Index **Ai,     // indices
    GrB_Index *Ai_size, // size of Ai in bytes

    void **Ax,          // values
    GrB_Index *Ax_size, // size of Ax in bytes

    // additional information for specific formats:
    GrB_Index *nvals,   // # of entries for bitmap format.
    bool *jumbled,      // if true, sparse/hypersparse may be jumbled.
    GrB_Index *nvec,    // size of Ah for hypersparse format.

    // information for all formats:
    int *sparsity,      // hypersparse, sparse, bitmap, or full
    bool *is_csc,       // if true then matrix is by-column, else by-row
    bool *is_uniform,   // if true then A has uniform values and only one
                        // entry is returned in Ax, regardless of nvals(A).
                        // TODO::: uniform valued matrices not yet supported
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A != NULL) ;
    GB_RETURN_IF_NULL_OR_FAULTY (*A) ;
    ASSERT_MATRIX_OK (*A, "A to export", GB0) ;
    ASSERT (!GB_ZOMBIES (*A)) ;
    ASSERT (GB_JUMBLED_OK (*A)) ;
    ASSERT (!GB_PENDING (*A)) ;

    GB_RETURN_IF_NULL (type) ;
    GB_RETURN_IF_NULL (vlen) ;
    GB_RETURN_IF_NULL (vdim) ;
    GB_RETURN_IF_NULL (Ax) ;
    GB_RETURN_IF_NULL (Ax_size) ;

    int s = GB_sparsity (*A) ;

    switch (s)
    {
        case GxB_HYPERSPARSE : 
            GB_RETURN_IF_NULL (nvec) ;
            GB_RETURN_IF_NULL (Ah) ; GB_RETURN_IF_NULL (Ah_size) ;

        case GxB_SPARSE : 
            if (is_sparse_vector)
            {
                GB_RETURN_IF_NULL (nvals) ;
            }
            else
            {
                GB_RETURN_IF_NULL (Ap) ; GB_RETURN_IF_NULL (Ap_size) ;
            }
            GB_RETURN_IF_NULL (Ai) ; GB_RETURN_IF_NULL (Ai_size) ;
            break ;

        case GxB_BITMAP : 
            GB_RETURN_IF_NULL (nvals) ;
            GB_RETURN_IF_NULL (Ab) ; GB_RETURN_IF_NULL (Ab_size) ;

        case GxB_FULL : 
            break ;

        default: ;
    }

    //--------------------------------------------------------------------------
    // export the matrix
    //--------------------------------------------------------------------------

    (*type) = (*A)->type ;
    (*vlen) = (*A)->vlen ;
    (*vdim) = (*A)->vdim ;

    // export A->x
    #ifdef GB_DEBUG
    GB_Global_memtable_remove ((*A)->x) ;
    #endif
    (*Ax) = (*A)->x ; (*A)->x = NULL ;
    (*Ax_size) = (*A)->x_size ;

    switch (s)
    {
        case GxB_HYPERSPARSE : 
            (*nvec) = (*A)->nvec ;

            // export A->h
            #ifdef GB_DEBUG
            GB_Global_memtable_remove ((*A)->h) ;
            #endif
            (*Ah) = (GrB_Index *) ((*A)->h) ; (*A)->h = NULL ;
            (*Ah_size) = (*A)->h_size ;

        case GxB_SPARSE : 
            if (jumbled != NULL)
            { 
                (*jumbled) = (*A)->jumbled ;
            }

            // export A->p, unless A is a sparse vector in CSC format
            if (is_sparse_vector)
            {
                (*nvals) = (*A)->p [1] ;
            }
            else
            {
                #ifdef GB_DEBUG
                GB_Global_memtable_remove ((*A)->p) ;
                #endif
                (*Ap) = (GrB_Index *) ((*A)->p) ; (*A)->p = NULL ;
                (*Ap_size) = (*A)->p_size ;
            }

            // export A->i
            #ifdef GB_DEBUG
            GB_Global_memtable_remove ((*A)->i) ;
            #endif
            (*Ai) = (GrB_Index *) ((*A)->i) ; (*A)->i = NULL ;
            (*Ai_size) = (*A)->i_size ;
            break ;

        case GxB_BITMAP : 
            (*nvals) = (*A)->nvals ;

            // export A->b
            #ifdef GB_DEBUG
            GB_Global_memtable_remove ((*A)->b) ;
            #endif
            (*Ab) = (*A)->b ; (*A)->b = NULL ;
            (*Ab_size) = (*A)->b_size ;

        case GxB_FULL : 

        default: ;
    }

    if (sparsity != NULL)
    { 
        (*sparsity) = s ;
    }
    if (is_csc != NULL)
    { 
        (*is_csc) = (*A)->is_csc ;
    }
    if (is_uniform != NULL)
    { 
        (*is_uniform) = false ;     // TODO::: uniform-valued matrices
    }

    //--------------------------------------------------------------------------
    // free the GrB_Matrix
    //--------------------------------------------------------------------------

    // frees the header of A, and A->p if A is a sparse GrB_Vector
    GB_Matrix_free (A) ;
    ASSERT ((*A) == NULL) ;
    return (GrB_SUCCESS) ;
}

