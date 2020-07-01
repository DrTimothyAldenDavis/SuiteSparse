// GrB_wait ( with no inputs ): DEPRECATED: TODO in 4.0: delete this
// DEPRECATED: This will be removed in SuiteSparse:GraphBLAS v4.0.

#include "GB.h"

#define GB_FREE_ALL ;

#if defined (USER_POSIX_THREADS)
pthread_mutex_t GB_sync ;
#endif

GrB_Info GrB_wait ( )       // DEPRECATED.  Do *not* use this function.
{
    GrB_Info info ;
    GB_WHERE ("GrB_wait (with no inputs) DEPRECATED ") ;
    GB_BURBLE_START ("GrB_wait (DEPRECATED: USE GrB_*_wait(object) instead) ") ;
    GrB_Matrix A = NULL ;
    while (true)
    {
        if (!GB_queue_remove_head (&A)) GB_PANIC ;
        if (A == NULL) break ;
        GB_MATRIX_WAIT (A) ;
    }
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

