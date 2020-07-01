// GB_wait.h: DEPRECATED: TODO in 4.0: delete this

#ifndef GB_DEPRECATED_H
#define GB_DEPRECATED_H

bool GB_queue_remove            // remove matrix from queue
(
    GrB_Matrix A                // matrix to remove
) ;
 
GB_PUBLIC
bool GB_queue_insert            // insert matrix at the head of queue
(
    GrB_Matrix A                // matrix to insert
) ;

bool GB_queue_remove_head       // remove matrix at the head of queue
(
    GrB_Matrix *Ahandle         // return matrix or NULL if queue empty
) ;

bool GB_queue_status            // get the queue status of a matrix
(
    GrB_Matrix A,               // matrix to check
    GrB_Matrix *p_head,         // head of the queue
    GrB_Matrix *p_prev,         // prev from A
    GrB_Matrix *p_next,         // next after A
    bool *p_enqd                // true if A is in the queue
) ;

#if defined (USER_POSIX_THREADS)
GB_PUBLIC pthread_mutex_t GB_sync ;
#endif

#endif

