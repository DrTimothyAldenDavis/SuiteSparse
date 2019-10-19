//------------------------------------------------------------------------------
// GB_queue_insert:  insert a matrix at the head of the matrix queue
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// check if the matrix has pending computations (either pending tuples or
// zombies, or both).  If it has any, and if it is not already in the queue,
// then insert it into the queue.

#include "GB.h"

void GB_queue_insert            // insert matrix at the head of queue
(
    GrB_Matrix A                // matrix to insert
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A != NULL) ;

    #ifndef NDEBUG
    {
        // walk the whole list to make sure it's OK
        GrB_Matrix A = (GrB_Matrix) GB_thread_local.queue_head ;
        while (A != NULL)
        {
            ASSERT_OK (GB_check (A, "A in the queue", 0)) ;
            ASSERT (PENDING (A) || ZOMBIES (A)) ;
            A = A->queue_next ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // insert the matrix at the head of the queue
    //--------------------------------------------------------------------------

    if ((A->npending > 0 || A->nzombies > 0) && IS_NOT_IN_QUEUE (A))
    {
        GrB_Matrix Head = GB_thread_local.queue_head ;
        A->queue_next = Head ;
        A->queue_prev = NULL ;
        if (Head != NULL)
        {
            Head->queue_prev = A ;
        }
        GB_thread_local.queue_head = A ;
    }
}

