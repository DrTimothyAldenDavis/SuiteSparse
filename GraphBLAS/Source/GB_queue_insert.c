// GB_queue_insert:  TODO in 4.0: delete 
// DEPRECATED:  all GB_queue_* will be removed when GrB_wait() is gone.

#include "GB.h"

GB_PUBLIC
bool GB_queue_insert (GrB_Matrix A)
{
    bool ok = true ;
    if ((A->Pending != NULL || A->nzombies > 0) && !(A->enqueued))
    {
        #define GB_CRITICAL_SECTION                                         \
        {                                                                   \
            if ((A->Pending != NULL || A->nzombies > 0) && !(A->enqueued))  \
            {                                                               \
                GrB_Matrix Head = (GrB_Matrix) (GB_Global_queue_head_get ( )) ;\
                A->queue_next = Head ;                                      \
                A->queue_prev = NULL ;                                      \
                A->enqueued = true ;                                        \
                if (Head != NULL)                                           \
                {                                                           \
                    Head->queue_prev = A ;                                  \
                }                                                           \
                GB_Global_queue_head_set (A) ;                              \
            }                                                               \
        }
        #include "GB_critical_section.c"
    }
    return (ok) ;
}

