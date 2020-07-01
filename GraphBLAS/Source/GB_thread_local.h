//------------------------------------------------------------------------------
// GB_thread_local.h: definitions for thread local storage
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Thread local storage is created by GrB_init or GxB_init (via GB_init),
// and then accessed by the error logging mechanism (GB_error), and the
// error reporting function GrB_error.

#ifndef GB_THREAD_LOCAL_H
#define GB_THREAD_LOCAL_H

#include "GB.h"

#if defined ( USER_POSIX_THREADS )
// use POSIX for thread-local storage
extern pthread_key_t GB_thread_local_key ;
#else
// use OpenMP for thread-local storage
extern char GB_thread_local_report [GB_RLEN+1] ;
#endif

bool GB_thread_local_init               // intialize thread-local storage
(
    void (* free_function) (void *)
) ;

char *GB_thread_local_get (void) ;      // get pointer to thread-local storage

#endif
