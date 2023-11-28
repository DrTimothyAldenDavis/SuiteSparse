// =============================================================================
// === GPUQREngine/Source/LLBundle_UpdateSecondMinIndex.cpp ====================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
//
// This file contains two functions that perform a scan through an LLBundle
// instance to discover the SecondMin rowtile and the Max, respectively.
//
// =============================================================================

// -----------------------------------------------------------------------------
// LLBundle::UpdateSecondMinIndex
// -----------------------------------------------------------------------------
#include "GPUQREngine_LLBundle.hpp"
#include "GPUQREngine_BucketList.hpp"
template <typename Int>
void LLBundle <Int>::UpdateSecondMinIndex
(
    void
)
{
    /* If we don't have a First, return early. */
    if(First == EMPTY) return;

    Int *next = Buckets->next;

    /* Scan to find the next second min index. */
    int inspect = SecondMin = next[First];
    while (inspect != EMPTY)
    {
        SecondMin = MIN(SecondMin, inspect);
        inspect = next[inspect];
    }
}
template void LLBundle <int32_t>::UpdateSecondMinIndex
(
    void
) ;
template void LLBundle <int64_t>::UpdateSecondMinIndex
(
    void
) ;

// -----------------------------------------------------------------------------
// LLBundle::UpdateMax
// -----------------------------------------------------------------------------
template <typename Int>
void LLBundle <Int>::UpdateMax
(
    void
)
{
    Int *next = Buckets->next;

    /* Scan to find the max. */
    Max = Shadow;
    for(Int tile=First; tile!=EMPTY; tile=next[tile]) Max = MAX(Max, tile);
}

template void LLBundle <int32_t>::UpdateMax
(
    void
) ;
template void LLBundle <int64_t>::UpdateMax
(
    void
) ;
