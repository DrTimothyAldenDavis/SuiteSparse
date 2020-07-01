// GB_queue_remove_head: TODO in 4.0: delete this
// DEPRECATED:  all GB_queue_* will be removed when GrB_wait() is gone.

#include "GB.h"

bool GB_queue_remove_head       // remove matrix at the head of queue
(
    GrB_Matrix *Ahandle         // return matrix or NULL if queue empty
)
{
    GrB_Matrix A = NULL ;
    bool ok = true ;
    #define GB_CRITICAL_SECTION                                             \
    {                                                                       \
        A = (GrB_Matrix) (GB_Global_queue_head_get ( )) ;                   \
        if (A != NULL)                                                      \
        {                                                                   \
            GrB_Matrix Next = (GrB_Matrix) A->queue_next ;                  \
            GB_Global_queue_head_set (Next) ;                               \
            if (Next != NULL)                                               \
            {                                                               \
                Next->queue_prev = NULL ;                                   \
            }                                                               \
            A->queue_next = NULL ;                                          \
            A->enqueued = false ;                                           \
        }                                                                   \
    }
    #include "GB_critical_section.c"
    (*Ahandle) = A ;
    return (ok) ;
}

