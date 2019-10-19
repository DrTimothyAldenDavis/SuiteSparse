//------------------------------------------------------------------------------
// GB_queue_init:  initialize the queue
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

//------------------------------------------------------------------------------
// All Global storage is declared and initialized here
//------------------------------------------------------------------------------

// If the user creates threads that work on GraphBLAS matrices, then all of
// those threads must share the same matrix queue, and the same mode.

GB_Global_struct GB_Global =
{

    // queued matrices with work to do
    .queue_head = NULL,         // pointer to first queued matrix

    // GraphBLAS mode
    .mode = GrB_NONBLOCKING,    // default is nonblocking

} ;

//------------------------------------------------------------------------------
// GB_queue_init
//------------------------------------------------------------------------------

void GB_queue_init
(
    const GrB_Mode mode         // blocking or non-blocking mode
)
{

    #pragma omp critical GB_queue
    {
        // clear the queue
        GB_Global.queue_head = NULL ;

        // set the mode: blocking or nonblocking
        GB_Global.mode = mode ;             // default is non-blocking
    }
}

