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

    if (A->enqueued)
    {
        // remove the matrix from the queue

        #pragma omp critical GB_queue
        {

            // GraphBLAS is not (yet) parallel, but the user application might
            // be.  This update to the global queue must be done in a critical
            // section.  If both GraphBLAS and the user application are
            // compiled with OpenMP, then the #pragma will protect the queue
            // from a race condition of simulateneous updates.

            if (A->enqueued)
            {
                // check the condition again, since GrB_wait could have been
                // called by another thread, which removes all matrices from
                // the queue, including this one.

                void *prev = A->queue_prev ;
                void *next = A->queue_next ;
                if (prev == NULL)
                {
                    // matrix is at the head of the queue; update the head
                    GB_Global.queue_head = next ;
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

                // matrix has been removed from the queue
                A->queue_prev = NULL ;
                A->queue_next = NULL ;
                A->enqueued = false ;
            }
        }
    }
}

