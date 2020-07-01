// GB_queue_remove: TODO in 4.0: delete
// DEPRECATED:  all GB_queue_* will be removed when GrB_wait() is gone.

#include "GB.h"

bool GB_queue_remove (GrB_Matrix A)
{
    bool ok = true ;
    if (A->enqueued)
    { 
        #define GB_CRITICAL_SECTION                                         \
        {                                                                   \
            if (A->enqueued)                                                \
            {                                                               \
                GrB_Matrix Prev = (GrB_Matrix) (A->queue_prev) ;            \
                GrB_Matrix Next = (GrB_Matrix) (A->queue_next) ;            \
                if (Prev == NULL)                                           \
                {                                                           \
                    GB_Global_queue_head_set (Next) ;                       \
                }                                                           \
                else                                                        \
                {                                                           \
                    Prev->queue_next = Next ;                               \
                }                                                           \
                if (Next != NULL)                                           \
                {                                                           \
                    Next->queue_prev = Prev ;                               \
                }                                                           \
                A->queue_prev = NULL ;                                      \
                A->queue_next = NULL ;                                      \
                A->enqueued = false ;                                       \
            }                                                               \
        }
        #include "GB_critical_section.c"
    }
    return (ok) ;
}

