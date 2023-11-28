// =============================================================================
// === GPUQREngine/Source/BucketList_CreateBundles.cpp =========================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
//
// CreateBundles selects rowtiles up to PANELSIZE and creates a new bundle
// ready for factorization.
//
// =============================================================================

// CreateBundles selects rowtiles up to PANELSIZE and creates a new bundle
// ready for factorization.
#include "GPUQREngine_BucketList.hpp"
template <typename Int>
void BucketList<Int>::CreateBundles
(
    void
)
{
    /* Look for idle tiles to extract into fresh bundles. */
    int colBucket = Wavefront;
    while (colBucket <= LastBucket)
    {
        // Get the tile from the bucket and skip this colBucket if it's empty.
        int tile = head[colBucket];
        if (SkipBundleCreation(tile, colBucket)){ colBucket++; continue; }

        /* At this point we know we're going to add a bundle. */
        LLBundle <Int> candidate(this, PanelSize, colBucket);
        for (int i=0; i<PanelSize && tile != EMPTY; i++)
        {
            /* Remove the node from the bucket lists. */
            Int nextNode = next[tile];
            Remove(tile, colBucket);

            /* Add the tile to the bundle and go to the next tile. */
            candidate.AddTileToSlots(tile);
            tile = nextNode;
        }

        Bundles[numBundles++] = candidate;
        bundleCount[colBucket]++;

        /* If we're at the wavefront: */
        if(Wavefront == colBucket)
        {
            /* If we have one bundle with one tile, advance the wavefront. */
            if(bundleCount[colBucket] == 1 && candidate.Count == 1)
            {
                Wavefront++;
            }
        }
    }
}

template void BucketList<int32_t>::CreateBundles
(
    void
) ;
template void BucketList<int64_t>::CreateBundles
(
    void
) ;

// SkipBundleCreation determines whether we should skip creating a new
// bundle for the specified tile in the specified column bucket.
template <typename Int>
bool BucketList<Int>::SkipBundleCreation
(
    Int tile,           // The tile to consider
    Int colBucket       // The column bucket it sits in
)
{
    if (tile == EMPTY) return true;

    /* We can skip creating the bundle if there's only one tile
       in the bucket and it either isn't native to the bucket or
       if it's already upper triangular
       (in which case don't refactorize it). */
    bool onlyOneTile = (next[tile] == EMPTY);
    bool isNative = (tile == colBucket);
    bool isTriu = triu[tile];
    if(onlyOneTile)
    {
        if(!isNative || isTriu) return true;
    }

    return false;
}

template bool BucketList<int32_t>::SkipBundleCreation
(
    int32_t tile,           // The tile to consider
    int32_t colBucket       // The column bucket it sits in
) ;
template bool BucketList<int64_t>::SkipBundleCreation
(
    int64_t tile,           // The tile to consider
    int64_t colBucket       // The column bucket it sits in
) ;

// IsInternal determines whether a tile is completely within the bounds
// of the front because if it isn't then we will need to use the special
// edge case kernels.
template <typename Int>
bool BucketList<Int>::IsInternal
(
    LLBundle <Int>& Bundle,
    int jLast
)
{
    /* Find the last row for the bundle. */
#if 0
    // We can play it safe and find the max on every call
    Int iTile = Bundle.Shadow;
    for(Int tile = Bundle.First; tile != EMPTY; tile = next[tile])
    {
        iTile = MAX(iTile, tile);
    }
    if(iTile != Bundle.Max) printf("%d vs %d\n", iTile, Bundle.Max);
    assert(iTile == Bundle.Max);
#else
    // We find the max as we construct the bundles
    Int iTile = Bundle.Max;
#endif

    /* We're internal if the last row and column is in bounds. */
    Int iLast = TILESIZE * (iTile+1) - 1;
    return(iLast < front->fm && jLast < front->fn);
}

template bool BucketList<int32_t>::IsInternal
(
    LLBundle <int32_t>& Bundle,
    int jLast
) ;
template bool BucketList<int64_t>::IsInternal
(
    LLBundle <int64_t>& Bundle,
    int jLast
) ;
