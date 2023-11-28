// =============================================================================
// === GPUQREngine/Source/GPUQREngine_Internal.cpp =============================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
//
// GPUQREngine_Internal is the call-down from the dense and sparse polymorphic
// wrappers. This code is responsible for maintaining the Scheduler and
// coordinating the factorization in a main loop.
//
// =============================================================================

#ifdef SUITESPARSE_CUDA

#include "GPUQREngine_Internal.hpp"
#include "GPUQREngine_Scheduler.hpp"
#include "GPUQREngine_Stats.hpp"

template <typename Int>
QREngineResultCode GPUQREngine_Internal
(
    size_t gpuMemorySize,   // The total available GPU memory size in bytes
    Front <Int> *fronts,          // The list of fronts to factorize
    Int numFronts,          // The number of fronts to factorize
    Int *Parent,            // The front-to-parent mapping
    Int *Childp,            // Front-to-child column pointers
    Int *Child,             // Child permutation
                            // (Child[Childp[f]] to Child[Childp[f+1]] are all
                            // the front identifiers for front "f"'s children.
    QREngineStats <Int> *stats    // An optional parameter. If present, statistics
                            // are collected and passed back to the caller
                            // via this struct
)
{
    bool ok = true;

    /* Create the scheduler. */
    Scheduler <Int> *scheduler = (Scheduler <Int> *) SuiteSparse_calloc(1,sizeof(Scheduler <Int>));
    if (scheduler == NULL)
    {
        return QRENGINE_OUTOFMEMORY;
    }

    new (scheduler) Scheduler <Int> (fronts, numFronts, gpuMemorySize);
    ok = scheduler->memory_ok && scheduler->cuda_ok;

    /* If we encountered problems initializing the scheduler: */
    if(!ok)
    {
        bool memory_ok = scheduler->memory_ok ;
        bool cuda_ok = scheduler->cuda_ok ;
        if(scheduler)
        {
            scheduler->~Scheduler();
            scheduler = (Scheduler <Int> *) SuiteSparse_free (scheduler) ;
        }
        if(!memory_ok) return QRENGINE_OUTOFMEMORY;
        if(!cuda_ok) return QRENGINE_GPUERROR;
    }

    bool completed = false;
    while(!completed)
    {
//      #ifdef GPUQRENGINE_RENDER
//      scheduler->render();
//      #endif
        scheduler->fillWorkQueue();
        scheduler->transferData();

        // Launch the kernel and break out of the loop if we encountered
        // a cuda error.
        scheduler->launchKernel();
        if(!scheduler->cuda_ok) break;

        completed = scheduler->postProcess();
        scheduler->toggleQueue();
    }

    /* Report metrics back to the caller. */
    if(stats)
    {
        stats->kernelTime = scheduler->kernelTime;
        stats->numLaunches = scheduler->numKernelLaunches;
        stats->flopsActual = scheduler->gpuFlops;
    }

    /* Explicitly invoke the destructor */
    scheduler->~Scheduler();
    scheduler = (Scheduler <Int>*) SuiteSparse_free(scheduler);

    return QRENGINE_SUCCESS;
}

template QREngineResultCode GPUQREngine_Internal
(
    size_t gpuMemorySize,   // The total available GPU memory size in bytes
    Front <int32_t> *fronts,          // The list of fronts to factorize
    int32_t numFronts,          // The number of fronts to factorize
    int32_t *Parent,            // The front-to-parent mapping
    int32_t *Childp,            // Front-to-child column pointers
    int32_t *Child,             // Child permutation
                            // (Child[Childp[f]] to Child[Childp[f+1]] are all
                            // the front identifiers for front "f"'s children.
    QREngineStats <int32_t> *stats    // An optional parameter. If present, statistics
                            // are collected and passed back to the caller
                            // via this struct
) ;
template QREngineResultCode GPUQREngine_Internal
(
    size_t gpuMemorySize,   // The total available GPU memory size in bytes
    Front <int64_t> *fronts,          // The list of fronts to factorize
    int64_t numFronts,          // The number of fronts to factorize
    int64_t *Parent,            // The front-to-parent mapping
    int64_t *Childp,            // Front-to-child column pointers
    int64_t *Child,             // Child permutation
                            // (Child[Childp[f]] to Child[Childp[f+1]] are all
                            // the front identifiers for front "f"'s children.
    QREngineStats <int64_t> *stats    // An optional parameter. If present, statistics
                            // are collected and passed back to the caller
                            // via this struct
) ;

template class BucketList<int32_t>;
template class BucketList<int64_t>;

template class LLBundle<int32_t>;
template class LLBundle<int64_t>;

template class Scheduler<int32_t>;
template class Scheduler<int64_t>;

#endif

