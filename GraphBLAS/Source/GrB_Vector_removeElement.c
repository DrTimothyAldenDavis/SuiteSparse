//------------------------------------------------------------------------------
// GrB_Vector_removeElement: remove a single entry from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Removes a single entry, V (i), from the vector V.

#include "GB.h"

#define GB_FREE_ALL ;
#define GB_WHERE_STRING "GrB_Vector_removeElement (v, i)"

//------------------------------------------------------------------------------
// GB_removeElement: remove V(i) if it exists
//------------------------------------------------------------------------------

static inline bool GB_removeElement
(
    GrB_Vector V,
    GrB_Index i
)
{

    //--------------------------------------------------------------------------
    // get the pattern of the vector
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Vp = V->p ;
    const int64_t *GB_RESTRICT Vi = V->i ;
    bool found ;

    // remove from a GrB_Vector
    int64_t pleft = 0 ;
    int64_t pright = Vp [1] - 1 ;

    //--------------------------------------------------------------------------
    // binary search in kth vector for index i
    //--------------------------------------------------------------------------

    // Time taken for this step is at most O(log(nnz(V))).
    bool is_zombie ;
    int64_t nzombies = V->nzombies ;
    GB_BINARY_SEARCH_ZOMBIE (i, Vi, pleft, pright, found, nzombies, is_zombie) ;

    //--------------------------------------------------------------------------
    // remove the entry
    //--------------------------------------------------------------------------

    if (found && !is_zombie)
    { 
        // V(i) becomes a zombie
        V->i [pleft] = GB_FLIP (i) ;
        V->nzombies++ ;
        // TODO in 4.0: delete:
        if (!(V->enqueued)) { if (!GB_queue_insert ((GrB_Matrix) V)) GB_PANIC ;}
    }

    return (found) ;
}

//------------------------------------------------------------------------------
// GrB_Vector_removeElement: remove a single entry from a vector
//------------------------------------------------------------------------------

GrB_Info GrB_Vector_removeElement
(
    GrB_Vector V,               // vector to remove entry from
    GrB_Index i                 // index
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // GB_WHERE ("Vec here") ;
    GB_CONTEXT_RETURN_IF_NULL (V) ;
    GB_CONTEXT_RETURN_IF_FAULTY (V) ;

    // check index
    if (i >= V->vlen)
    { 
        GB_WHERE (GB_WHERE_STRING) ;
        return (GB_ERROR (GrB_INVALID_INDEX, (GB_LOG, "Row index "
            GBu " out of range; must be < " GBd, i, V->vlen))) ;
    }

    bool V_is_pending = GB_PENDING (V) ; 
    if (V->nzmax == 0 && !V_is_pending)
    { 
        // quick return
        return (GrB_SUCCESS) ;
    }

    // remove the entry
    if (GB_removeElement (V, i))
    { 
        // found it; no need to assemble pending tuples
        return (GrB_SUCCESS) ;
    }

    // assemble any pending tuples; zombies are OK
    if (V_is_pending)
    { 
        GrB_Info info ;
        GB_WHERE (GB_WHERE_STRING) ;
        GB_BURBLE_START ("GrB_Vector_removeElement") ;
        GB_OK (GB_Matrix_wait ((GrB_Matrix) V, Context)) ;
        ASSERT (!GB_ZOMBIES (V)) ;
        ASSERT (!GB_PENDING (V)) ;
        GB_BURBLE_END ;
    }

    // look again; remove the entry if it was a pending tuple
    GB_removeElement (V, i) ;
    return (GrB_SUCCESS) ;
}

