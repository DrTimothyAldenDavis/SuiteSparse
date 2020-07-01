//------------------------------------------------------------------------------
// GrB_Matrix_removeElement: remove a single entry from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Removes a single entry, C (row,col), from the matrix C.

#include "GB.h"

#define GB_FREE_ALL ;
#define GB_WHERE_STRING "GrB_Matrix_removeElement (C, row, col)"

//------------------------------------------------------------------------------
// GB_removeElement: remove C(i,j) if it exists
//------------------------------------------------------------------------------

static inline bool GB_removeElement
(
    GrB_Matrix C,
    GrB_Index i,
    GrB_Index j
)
{

    //--------------------------------------------------------------------------
    // binary search in C->h for vector j
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Cp = C->p ;
    const int64_t *GB_RESTRICT Ci = C->i ;
    bool found ;

    // remove an entry from vector j of a GrB_Matrix
    int64_t k ;
    if (C->is_hyper)
    {
        // look for vector j in hyperlist C->h [0 ... C->nvec-1]
        const int64_t *Ch = C->h ;
        int64_t pleft = 0 ;
        int64_t pright = C->nvec-1 ;
        GB_BINARY_SEARCH (j, Ch, pleft, pright, found) ;
        if (!found)
        { 
            // vector j is empty
            return (false) ;
        }
        ASSERT (j == Ch [pleft]) ;
        k = pleft ;
    }
    else
    { 
        k = j ;
    }
    int64_t pleft = Cp [k] ;
    int64_t pright = Cp [k+1] - 1 ;

    //--------------------------------------------------------------------------
    // binary search in kth vector for index i
    //--------------------------------------------------------------------------

    // Time taken for this step is at most O(log(nnz(C(:,j))).
    bool is_zombie ;
    int64_t nzombies = C->nzombies ;
    GB_BINARY_SEARCH_ZOMBIE (i, Ci, pleft, pright, found, nzombies, is_zombie) ;

    //--------------------------------------------------------------------------
    // remove the entry
    //--------------------------------------------------------------------------

    if (found && !is_zombie)
    { 
        // C(i,j) becomes a zombie
        C->i [pleft] = GB_FLIP (i) ;
        C->nzombies++ ;
        // TODO in 4.0: delete:
        if (!(C->enqueued)) { if (!GB_queue_insert (C)) GB_PANIC ; }
    }

    return (found) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_removeElement: remove a single entry from a matrix
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_removeElement
(
    GrB_Matrix C,               // matrix to remove entry from
    GrB_Index row,              // row index
    GrB_Index col               // column index
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CONTEXT_RETURN_IF_NULL (C) ;
    GB_CONTEXT_RETURN_IF_FAULTY (C) ;

    // look for index i in vector j
    int64_t i, j, nrows, ncols ;
    if (C->is_csc)
    { 
        // C is stored by column
        i = row ;
        j = col ;
        nrows = C->vlen ;
        ncols = C->vdim ;
    }
    else
    { 
        // C is stored by row
        i = col ;
        j = row ;
        nrows = C->vdim ;
        ncols = C->vlen ;
    }

    // check row and column indices
    if (row >= nrows)
    { 
        GB_WHERE (GB_WHERE_STRING) ;
        return (GB_ERROR (GrB_INVALID_INDEX, (GB_LOG, "Row index "
            GBu " out of range; must be < " GBd, row, nrows))) ;
    }
    if (col >= ncols)
    { 
        GB_WHERE (GB_WHERE_STRING) ;
        return (GB_ERROR (GrB_INVALID_INDEX, (GB_LOG, "Column index "
            GBu " out of range; must be < " GBd, col, ncols))) ;
    }

    bool C_is_pending = GB_PENDING (C) ;
    if (C->nzmax == 0 && !C_is_pending)
    { 
        // quick return
        return (GrB_SUCCESS) ;
    }

    // remove the entry
    if (GB_removeElement (C, i, j))
    { 
        // found it; no need to assemble pending tuples
        return (GrB_SUCCESS) ;
    }

    // assemble any pending tuples; zombies are OK
    if (C_is_pending)
    { 
        GrB_Info info ;
        GB_WHERE (GB_WHERE_STRING) ;
        GB_BURBLE_START ("GrB_Matrix_removeElement") ;
        GB_OK (GB_Matrix_wait (C, Context)) ;
        ASSERT (!GB_ZOMBIES (C)) ;
        ASSERT (!GB_PENDING (C)) ;
        GB_BURBLE_END ;
    }

    // look again; remove the entry if it was a pending tuple
    GB_removeElement (C, i, j) ;
    return (GrB_SUCCESS) ;
}

