// =============================================================================
// === GPUQREngine/Include/GPUQREngine_LLBundle.hpp ============================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
//
// LLBundle is a principal class in the GPUQREngine.
//
// This class is responsible for maintaining the CPU's view of state information
// during the factorization process.
//
// LLBundles are manipulated by its hosting BucketList and are:
//   1) Advanced
//   2) Created
//   3) Grown       (in the context of pipelining)
//   4) Operated on (participating in Apply, Factorize, or ApplyFactorize tasks)
//
// =============================================================================

#ifndef GPUQRENGINE_LLBUNDLE_HPP
#define GPUQRENGINE_LLBUNDLE_HPP

#include "GPUQREngine_Common.hpp"
#include "GPUQREngine_TaskDescriptor.hpp"

struct TaskDescriptor;
template <typename Int>
class BucketList;


template <typename Int = int64_t> class LLBundle
{
public:
    BucketList <Int> *Buckets; // A back pointer to the hosting BucketList

    Int NativeBucket;   // The column bucket the bundle belongs "is native" to

    Int Shadow;         // A memento for the factorized First tile.
                        // The CPU needs to know

    Int First;          // The tile with the smallest rowtile index.
                        // For factorize tasks, this tile is made upper
                        // triangular.

    Int Last;           // The index of the last filled slot in the bundle.

    Int Delta;          // The index of where the Delta starts
                        // Delta is used in pipelining when we attach a
                        // factorize task to a finishing apply.

    Int SecondMin;      // The index of where First's replacement is

    Int Max;            // The index of the largest element (by rowtile)

    Int PanelSize;
    Int ApplyCount; // # tiles participating in an APPLY, including the Shadow.
    Int Count;      // # tiles in the Bundle (Slots+Delta), but not the Shadow.

    double *VT[2];  // Pointers to VT tiles.
                    // When performing a pipelined task (ApplyFactorize),
                    // memory must be reserved for two separate VT tiles:
                    //   1) For the HH vectors involved in the Apply
                    //   2) For the HH vectors resulting from the factorization

    bool IsFull
    (
        void
    )
    {
        return (Count == PanelSize);
    }

    TaskType CurrentTask;

    void *operator new(long unsigned int, LLBundle <Int>* p){ return p; }
    //------------------------------------------------------------------------------
    //
    // This file contains the constructor and destructor for the LLBundle class.
    // The constructor will attempt to reserve a VT tile automatically, since newly
    // created bundles are immediately slated for factorization.
    //
    // =============================================================================
    LLBundle
    (
        BucketList <Int> *buckets,
        Int panelSize,
        Int nativeBucket
    )
    {
        Buckets = buckets;
        PanelSize = panelSize;
        NativeBucket = nativeBucket;
        SecondMin = Shadow = First = Delta = Max = EMPTY;
        Count = ApplyCount = 0;
        VT[0] = VT[1] = NULL;
    
        /* Create the factorize task and allocate a VT block. */
        CurrentTask = TASKTYPE_GenericFactorize;
        VT[0] = buckets->allocateVT();
    }
    // LLBundle destructor:
    ~LLBundle()
    {
    }
    
    #ifdef GPUQRENGINE_PIPELINING
    void AddTileToDelta(Int rowTile);
    #endif

    void AddTileToSlots(Int rowTile);

    // Advance: returns T/F if the bundle is still around after being advanced.
    bool Advance();

    void PipelinedRearrange();

    void UpdateSecondMinIndex();
    void UpdateMax();
    

    void gpuPack(TaskDescriptor *cpuTask);
};

extern template class LLBundle<int32_t>;
extern template class LLBundle<int64_t>;
#endif
