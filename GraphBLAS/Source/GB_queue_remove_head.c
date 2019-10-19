//------------------------------------------------------------------------------
// GB_queue_remove_head: remove the matrix at the head of the matrix queue
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Matrix GB_queue_remove_head ( )   // return matrix or NULL if queue empty
{

    //--------------------------------------------------------------------------
    // remove the matrix at the head the queue
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL ;

    #pragma omp critical GB_queue
    {

        // GraphBLAS is not (yet) parallel, but the user application might
        // be.  This update to the global queue must be done in a critical
        // section.  If both GraphBLAS and the user application are
        // compiled with OpenMP, then the #pragma will protect the queue
        // from a race condition of simulateneous updates.

        // get the matrix at the head of the queue
        A = (GrB_Matrix) (GB_Global.queue_head) ;

        // remove it from the queue
        if (A != NULL)
        {
            // shift the head to the next matrix in the queue
            GB_Global.queue_head = A->queue_next ;

            // mark this matrix has no longer in the queue
            ASSERT (A->queue_prev == NULL) ;
            A->queue_next = NULL ;
            A->enqueued = false ;
        }
    }

    //--------------------------------------------------------------------------
    // return the matrix that was just removed from the head the queue
    //--------------------------------------------------------------------------

    return (A) ;
}

