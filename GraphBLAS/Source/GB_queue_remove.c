//------------------------------------------------------------------------------
// GB_queue_remove: remove a matrix from the matrix queue
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

void GB_queue_remove            // remove matrix from queue
(
    GrB_Matrix A                // matrix to remove
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A != NULL) ;

    //--------------------------------------------------------------------------
    // remove the matrix from the queue
    //--------------------------------------------------------------------------

    void *prev = A->queue_prev ;
    void *next = A->queue_next ;

    if (IS_NOT_IN_QUEUE (A))
    {
        // matrix is not yet in the queue; do nothing.  This case can occur
        // if GB_add_pending ran out of memory.  In that case, the matrix may
        // have already been in the queue from a prior operation, so it must be
        // removed.
        ;
    }
    else
    {
        // remove the matrix from the queue
        if (prev == NULL)
        {
            // matrix is at the head of the queue; update the head
            GB_thread_local.queue_head = next ;
        }
        else
        {
            // matrix is not the first in the queue
            GrB_Matrix Prev = (GrB_Matrix) prev ;
            Prev->queue_next = next ;
        }
        if (next != NULL)
        {
            // update the previous link of the next matrix, if any
            GrB_Matrix Next = (GrB_Matrix) next ;
            Next->queue_prev = prev ;
        }
    }

    // matrix has been removed from the queue
    A->queue_prev = NULL ;
    A->queue_next = NULL ;
}

