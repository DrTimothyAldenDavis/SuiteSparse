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

    //--------------------------------------------------------------------------
    // insert the matrix at the head of the queue
    //--------------------------------------------------------------------------

    if ((A->npending > 0 || A->nzombies > 0) && !(A->enqueued))
    {
        // A is not in the queue yet, but needs to be there

        #pragma omp critical GB_queue
        {

            // GraphBLAS is not (yet) parallel, but the user application might
            // be.  This update to the global queue must be done in a critical
            // section.  If both GraphBLAS and the user application are
            // compiled with OpenMP, then the #pragma will protect the queue
            // from a race condition of simulateneous updates.

            if ((A->npending > 0 || A->nzombies > 0) && !(A->enqueued))
            {

                // check the condition again, inside the critical section,
                // just to be safe

                // add the matrix to the head of the queue
                GrB_Matrix head = (GrB_Matrix) (GB_Global.queue_head) ;
                A->queue_next = head ;
                A->queue_prev = NULL ;
                A->enqueued = true ;
                if (head != NULL)
                {
                    head->queue_prev = A ;
                }
                GB_Global.queue_head = A ;
            }
        }
    }
}

