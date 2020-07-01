// GB_queue_status:  TODO in 4.0: delete
// DEPRECATED:  all GB_queue_* will be removed when GrB_wait() is gone.

#include "GB.h"

bool GB_queue_status            // get the queue status of a matrix
(
    GrB_Matrix A,               // matrix to check
    GrB_Matrix *p_head,         // head of the queue
    GrB_Matrix *p_prev,         // prev from A
    GrB_Matrix *p_next,         // next after A
    bool *p_enqd                // true if A is in the queue
)
{ 
    bool ok = true ;
    (*p_head) = NULL ;
    (*p_prev) = NULL ;
    (*p_next) = NULL ;
    (*p_enqd) = NULL ;
    #define GB_CRITICAL_SECTION                                             \
    {                                                                       \
        (*p_head) = (GrB_Matrix) (GB_Global_queue_head_get ( )) ;           \
        (*p_prev) = (GrB_Matrix) (A->queue_prev) ;                          \
        (*p_next) = (GrB_Matrix) (A->queue_next) ;                          \
        (*p_enqd) = A->enqueued ;                                           \
    }
    #include "GB_critical_section.c"
    return (ok) ;
}

