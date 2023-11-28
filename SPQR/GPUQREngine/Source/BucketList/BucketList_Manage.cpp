// =============================================================================
// === GPUQREngine/Source/BucketList_Manage.cpp ================================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
//
// This file contains management logic for the BucketList data structure.
// Constant time insertions and removals are possible because of the
// doubly-linked nature of the BucketList. Additional routines include
// allocating and releasing VT tiles and removing a head tile, for use in
// pipelined factorization.
//
// =============================================================================
#include "GPUQREngine_BucketList.hpp"
template <typename Int>
void BucketList<Int>::Insert
(
    Int tile,
    Int bucket,
    bool upperTriangular
)
{
    Int temp = head[bucket];
    head[bucket] = tile;
    next[tile] = temp;
    if (temp != EMPTY) prev[temp] = tile;
    prev[tile] = EMPTY;

    /* Set whether this tile is upper triangular. */
    triu[tile] = upperTriangular;

    idleTileCount[bucket]++;
    numIdleTiles++;

    /* Keep track of the last bucket. */
    LastBucket = MAX(LastBucket, bucket);
}
template void BucketList<int32_t>::Insert
(
    int32_t tile,
    int32_t bucket,
    bool upperTriangular
) ;
template void BucketList<int64_t>::Insert
(
    int64_t tile,
    int64_t bucket,
    bool upperTriangular
) ;

template <typename Int>
void BucketList<Int>::Remove
(
    Int tile,
    Int bucket
)
{
    if (tile == EMPTY) return;

    Int ptile = prev[tile];
    Int ntile = next[tile];
    if (ptile != EMPTY) next[ptile] = ntile;
    if (ntile != EMPTY) prev[ntile] = ptile;

    /* If we removed the head entry, update head. */
    if (ptile == EMPTY) head[bucket] = ntile;

    /* "tile" isn't in a bucket, so there's no pinv to keep track of. */
    prev[tile] = EMPTY;
    next[tile] = EMPTY;

    idleTileCount[bucket]--;
    numIdleTiles--;
}

template void BucketList<int32_t>::Remove
(
    int32_t tile,
    int32_t bucket
) ;
template void BucketList<int64_t>::Remove
(
    int64_t tile,
    int64_t bucket
) ;

#ifdef GPUQRENGINE_PIPELINING
template <typename Int>
Int BucketList<Int>::RemoveHead
(
    Int bucket                  // The bucket number
)
{
    Int tile = head[bucket];
    Remove(tile, bucket);
    return tile;
}
template int32_t BucketList<int32_t>::RemoveHead
(
    int32_t bucket                  // The bucket number
) ;
template int64_t BucketList<int64_t>::RemoveHead
(
    int64_t bucket                  // The bucket number
) ;
#endif
