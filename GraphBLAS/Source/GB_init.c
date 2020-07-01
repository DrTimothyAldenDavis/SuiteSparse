//------------------------------------------------------------------------------
// GB_init: initialize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GrB_init, GxB_init, or GxB_cuda_init must called before any other GraphBLAS
// operation; all three rely on this internal function.  If GraphBLAS is used
// by multiple user threads, only one can call GrB_init, GxB_init or
// GxB_cuda_init.

// Result are undefined if multiple user threads simultaneously
// call GrB_init, GxB_init, or GxB_cuda_init.

// GrB_finalize must be called as the last GraphBLAS operation.

// GrB_init, GxB_init, or GxB_cuda_init define the mode that GraphBLAS will
// use:  blocking or non-blocking.  With blocking mode, all operations finish
// before returning to the user application.  With non-blocking mode,
// operations can be left pending, and are computed only when needed.

// GxB_init is the same as GrB_init except that it also defines the
// malloc/calloc/realloc/free functions to use.

// GxB_cuda_init is the same as GrB_init, except that it passes in
// caller_is_GxB_cuda_init as true to this function.  GxB_init and GrB_init
// both pass this flag in as false.

#include "GB.h"
#include "GB_thread_local.h"
#include "GB_mkl.h"

//------------------------------------------------------------------------------
// GB_init
//------------------------------------------------------------------------------

GrB_Info GB_init            // start up GraphBLAS
(
    const GrB_Mode mode,    // blocking or non-blocking mode

    // pointers to memory management functions.  Must be non-NULL.
    void * (* malloc_function  ) (size_t),
    void * (* calloc_function  ) (size_t, size_t),
    void * (* realloc_function ) (void *, size_t),
    void   (* free_function    ) (void *),
    bool malloc_is_thread_safe,

    bool caller_is_GxB_cuda_init,       // true for GxB_cuda_init only

    GB_Context Context      // from GrB_init or GxB_init
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // Do not log the error for GrB_error, since it might not be initialized.

    if (GB_Global_GrB_init_called_get ( ))
    { 
        // GrB_init can only be called once
        return (GrB_PANIC) ;
    }

    GB_Global_GrB_init_called_set (true) ;

    if (! (mode == GrB_BLOCKING || mode == GrB_NONBLOCKING))
    { 
        // invalid mode
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // establish malloc/calloc/realloc/free
    //--------------------------------------------------------------------------

    // GrB_init passes in the ANSI C11 malloc/calloc/realloc/free
    // GxB_cuda_init passes in NULL pointers; they are now defined below.

    if (caller_is_GxB_cuda_init)
    {
        #if defined ( GBCUDA )
        // CUDA is available at compile time, and requested at run time via
        // GxB_cuda_init.  Use CUDA unified memory management functions.
        malloc_function = GxB_cuda_malloc ;
        calloc_function = GxB_cuda_calloc ;
        realloc_function = NULL ;
        free_function = GxB_cuda_free ;
        #else
        // CUDA not available at compile time.  Use ANSI C memory managment
        // functions instead, even though the caller is GxB_cuda_init.
        // No GPUs will be used.
        malloc_function = malloc ;
        calloc_function = calloc ;
        realloc_function = realloc ;
        free_function = free ;
        #endif
    }

    GB_Global_malloc_function_set  (malloc_function ) ;
    GB_Global_calloc_function_set  (calloc_function ) ;
    GB_Global_realloc_function_set (realloc_function) ;
    GB_Global_free_function_set    (free_function   ) ;
    GB_Global_malloc_is_thread_safe_set (malloc_is_thread_safe) ;

    #if GB_HAS_MKL_GRAPH
    printf ("MKL version: %d\n", GB_INTEL_MKL_VERSION) ;
    // also set the MKL allocator functions
    i_malloc  = malloc_function ;
    i_calloc  = calloc_function ;
    i_realloc = realloc_function ;
    i_free    = free_function ;
    #endif

    //--------------------------------------------------------------------------
    // max number of threads
    //--------------------------------------------------------------------------

    // Maximum number of threads for internal parallelization.
    // SuiteSparse:GraphBLAS requires OpenMP to use parallelization within
    // calls to GraphBLAS.  The user application may also call GraphBLAS in
    // parallel, from multiple user threads.  The user threads can use OpenMP,
    // or POSIX pthreads.

    GB_Global_nthreads_max_set (GB_Global_omp_get_max_threads ( )) ;
    GB_Global_chunk_set (GB_CHUNK_DEFAULT) ;

    //--------------------------------------------------------------------------
    // control usage of Intel MKL
    //--------------------------------------------------------------------------

    GB_Global_use_mkl_set (false) ;

    //--------------------------------------------------------------------------
    // initialize thread-local storage
    //--------------------------------------------------------------------------

    if (!GB_thread_local_init (free_function)) GB_PANIC ;

    #if defined (USER_POSIX_THREADS)
    {
        // TODO in 4.0: delete
        bool ok = (pthread_mutex_init (&GB_sync, NULL) == 0) ;
        if (!ok) GB_PANIC ;
    }
    #endif

    //--------------------------------------------------------------------------
    // initialize the blocking/nonblocking mode
    //--------------------------------------------------------------------------

    GB_Global_queue_head_set (NULL) ;   // TODO in 4.0: delete

    // set the mode: blocking or nonblocking
    GB_Global_mode_set (mode) ;

    //--------------------------------------------------------------------------
    // set the global default format
    //--------------------------------------------------------------------------

    // set the default hypersparsity ratio and CSR/CSC format;  any thread
    // can do this later as well, so there is no race condition danger.

    GB_Global_hyper_ratio_set (GB_HYPER_DEFAULT) ;
    GB_Global_is_csc_set (GB_FORMAT_DEFAULT != GxB_BY_ROW) ;

    //--------------------------------------------------------------------------
    // initialize malloc tracking (testing and debugging only)
    //--------------------------------------------------------------------------

    GB_Global_malloc_tracking_set (false) ;
    GB_Global_nmalloc_clear ( ) ;
    GB_Global_malloc_debug_set (false) ;
    GB_Global_malloc_debug_count_set (0) ;

    //--------------------------------------------------------------------------
    // development use only; controls diagnostic output
    //--------------------------------------------------------------------------

    GB_Global_burble_set (false) ;

    //--------------------------------------------------------------------------
    // CUDA initializations
    //--------------------------------------------------------------------------

    // If CUDA exists (#define GBCUDA) and if the caller is GxB_cuda_init, then
    // query the system for the # of GPUs available, their memory sizes, SM
    // counts, and other capabilities.  Unified Memory support is assumed.
    // Then warmup each GPU.

    #if defined ( GBCUDA )
    if (caller_is_GxB_cuda_init)
    {
        // query the system for the # of GPUs
        GB_Global_gpu_control_set (GxB_DEFAULT) ;
        if (!GB_Global_gpu_count_set (true)) GB_PANIC ;
        int gpu_count = GB_Global_gpu_count_get ( ) ;
        fprintf (stderr, "gpu_count: %d\n", gpu_count) ;
        for (int device = 0 ; device < gpu_count ; device++)
        {
            // query the GPU and then warm it up
            if (!GB_Global_gpu_device_properties_get (device)) GB_PANIC ;
            if (!GB_cuda_warmup (device)) GB_PANIC ;
            fprintf (stderr, "gpu %d memory %g Gbytes, %d SMs\n", device,
                ((double) GB_Global_gpu_memorysize_get (device)) / 1e9,
                GB_Global_gpu_sm_get (device)) ;
        }
        // TODO for GPU: check for jit cache
    }
    else
    #endif
    {
        // CUDA not available at compile-time, or available but not requested.
        GB_Global_gpu_control_set (GxB_GPU_NEVER) ;
        GB_Global_gpu_count_set (0) ;
    }

    GB_Global_gpu_chunk_set (GxB_DEFAULT) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

