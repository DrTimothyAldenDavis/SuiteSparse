// Source/Template/GB_critical_section: TODO in 4.0: delete
// DEPRECATED: This critical section is only used to protect the global queue
// of matrices with pending operations, for GrB_wait ( ).  It will be removed
// in v4.0.

{
    #if defined (USER_POSIX_THREADS)
    {
        ok = (pthread_mutex_lock (&GB_sync) == 0) ;
        GB_CRITICAL_SECTION ;
        ok = ok && (pthread_mutex_unlock (&GB_sync) == 0) ;
    }
    #else
    { 
        #pragma omp critical(GB_critical_section)
        GB_CRITICAL_SECTION ;
    }
    #endif
}

#undef GB_CRITICAL_SECTION

