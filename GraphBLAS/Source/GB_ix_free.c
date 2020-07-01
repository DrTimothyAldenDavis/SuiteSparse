//------------------------------------------------------------------------------
// GB_ix_free: free A->i, A->x, pending tuples, zombies; A->p, A->h unchanged
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Since A->p and A->h are unchanged, the matrix is still valid (unless it was
// invalid on input).  nnz(A) would report zero, and so would GrB_Matrix_nvals.

#include "GB_Pending.h"

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_ix_free             // free A->i and A->x of a matrix
(
    GrB_Matrix A                // matrix with content to free
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (A == NULL)
    { 
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // free all but A->p and A->h
    //--------------------------------------------------------------------------

    // zombies and pending tuples are about to be deleted
    ASSERT (GB_PENDING_OK (A)) ; ASSERT (GB_ZOMBIES_OK (A)) ;

    // free A->i unless it is shallow
    if (!A->i_shallow)
    { 
        GB_FREE (A->i) ;
    }
    A->i = NULL ;
    A->i_shallow = false ;

    // free A->x unless it is shallow
    if (!A->x_shallow)
    { 
        GB_FREE (A->x) ;
    }
    A->x = NULL ;
    A->x_shallow = false ;

    A->nzmax = 0 ;

    // no zombies remain
    A->nzombies = 0 ;

    // free the list of pending tuples
    GB_Pending_free (&(A->Pending)) ;

    if (!GB_queue_remove (A)) return (GrB_PANIC) ;  // TODO in 4.0: delete

    return (GrB_SUCCESS) ;
}

