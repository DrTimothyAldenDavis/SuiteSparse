//------------------------------------------------------------------------------
// GrB_wait: finish all pending computations
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The GrB_wait function forces all pending operations to complete.  Blocking
// mode is as if the GrB_wait operation is called whenever a GraphBLAS
// operation returns to the user.

// The non-blocking mode can have side effects if user-defined functions have
// side effects or if they rely on global variables, which are not under the
// control of GraphBLAS.  Suppose the user creates a user-defined operator that
// accesses a global variable.  That operator is then used in a GraphBLAS
// operation, which is left pending.  If the user then changes the global
// variable before pending operations complete, the pending operations will be
// eventually computed with this different value.

// Worse yet, a user-defined operator can be freed before it is needed to
// finish a pending operation.  To avoid this, call GrB_wait before modifying
// any global variables relied upon by user-defined operators, or before
// freeing any user-defined types, operators, monoids, or semirings.

#include "GB.h"

GrB_Info GrB_wait ( )       // finish all pending computations
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    WHERE ("GrB_wait ( )") ;

    #ifndef NDEBUG
    // walk the whole list to make sure it's OK
    GrB_Matrix A = (GrB_Matrix) GB_thread_local.queue_head ;
    while (A != NULL)
    {
        ASSERT_OK (GB_check (A, "to assemble in GrB_wait", 0)) ;
        ASSERT (PENDING (A) || ZOMBIES (A)) ;
        A = A->queue_next ;
    }
    #endif

    //--------------------------------------------------------------------------
    // assemble all matrices with lingering zombies and/or pending tuples
    //--------------------------------------------------------------------------

    while (GB_thread_local.queue_head != NULL)
    {
        // get the head of the queue
        GrB_Matrix A = (GrB_Matrix) GB_thread_local.queue_head ;
        ASSERT_OK (GB_check (A, "to assemble in GrB_wait", 0)) ;
        ASSERT (PENDING (A) || ZOMBIES (A)) ;
        // delete any lingering zombies and assemble any pending tuples
        // this also removes A from the queue
        APPLY_PENDING_UPDATES (A) ;
    }

    return (REPORT_SUCCESS) ;
}

