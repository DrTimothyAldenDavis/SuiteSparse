//------------------------------------------------------------------------------
// GB_nvals: number of entries in a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GB_nvals           // get the number of entries in a matrix
(
    GrB_Index *nvals,       // matrix has nvals entries
    const GrB_Matrix A,     // matrix to query
    GB_Context Context
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    // delete any lingering zombies and assemble any pending tuples
    // TODO in 4.0: delete this line of code:
    GB_MATRIX_WAIT (A) ; ASSERT (!GB_ZOMBIES (A)) ; ASSERT (!GB_PENDING (A)) ;

    GB_RETURN_IF_NULL (nvals) ;

    // leave zombies alone, but assemble any pending tuples
    if (GB_PENDING (A))
    {
        ASSERT (GB_DEAD_CODE) ; // TODO in 4.0: delete this line
        GB_MATRIX_WAIT (A) ;
    }

    //--------------------------------------------------------------------------
    // return the number of entries in the matrix
    //--------------------------------------------------------------------------

    // Pending tuples are disjoint from the zombies and the live entries in the
    // matrix.  However, there can be duplicates in the pending tuples, and the
    // number of duplicates has not yet been determined.  Thus, zombies can be
    // tolerated but pending tuples cannot.

    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    (*nvals) = GB_NNZ (A) - (A->nzombies) ;
    return (GrB_SUCCESS) ;
}

