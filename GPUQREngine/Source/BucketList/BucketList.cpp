// =============================================================================
// === GPUQREngine/Source/BucketList.cpp =======================================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
//
// This file contains logic to construct and destroy a BucketList.
//
// To support debugging and code coverage tests, we use placement new in order
// to trap and exercise out-of-memory conditions within the operating system
// memory manager.
//
// =============================================================================
// The pattern in use in this file is the memory allocation self-contained
// within the constructor with concrete initialization codes appearing in the
// initializer.This practice is common in OO languages, such as Java in which
// the constructor is responsible for memory management AND initialization.
// =============================================================================

#define FREE_EVERYTHING_BUCKET \
    head = (Int *) SuiteSparse_free(head); \
    idleTileCount = (Int *) SuiteSparse_free(idleTileCount); \
    bundleCount = (Int *) SuiteSparse_free(bundleCount); \
    prev = (Int *) SuiteSparse_free(prev); \
    next = (Int *) SuiteSparse_free(next); \
    triu = (bool *) SuiteSparse_free(triu); \
    Bundles = (LLBundle <Int> *) SuiteSparse_free(Bundles); \
    gpuVT = (double **) SuiteSparse_free(gpuVT); \
    wsMongoVT = Workspace::destroy(wsMongoVT);
#include "GPUQREngine_BucketList.hpp"
template <typename Int>
BucketList<Int>::BucketList
(
    Front <Int> *F,
    Int minApplyGranularity
)
{
    this->useFlag = true;
    this->memory_ok = true;

    this->front = F;
    int fm = front->fm;
    int fn = front->fn;
    Int *Stair = front->Stair;

    numRowTiles = CEIL(fm, TILESIZE);
    numColTiles = CEIL(fn, TILESIZE);
    numBuckets = numColTiles;
    numIdleTiles = numBundles = 0;
    PanelSize = PANELSIZE;
    TileSize = TILESIZE;
    Wavefront = LastBucket = 0;
    VThead = 0;
    ApplyGranularity = minApplyGranularity;

    // FUTURE: malloc space once for all fronts in a stage

    /* Allocate Memory */
    head = (Int*) SuiteSparse_calloc(numBuckets, sizeof(Int));
    idleTileCount = (Int*) SuiteSparse_calloc(numBuckets, sizeof(Int));
    bundleCount = (Int*) SuiteSparse_calloc(numBuckets, sizeof(Int));
    next = (Int*) SuiteSparse_calloc(numRowTiles, sizeof(Int));
    prev = (Int*) SuiteSparse_calloc(numRowTiles, sizeof(Int));
    triu = (bool*) SuiteSparse_calloc(numRowTiles, sizeof(bool));
    Bundles = (LLBundle <Int>*) SuiteSparse_calloc(numRowTiles, sizeof(LLBundle <Int>));
    gpuVT = (double**) SuiteSparse_calloc(numRowTiles, sizeof(double*));

    // malloc wsMongoVT on the GPU
    wsMongoVT = Workspace::allocate (numRowTiles*(TILESIZE+1)*TILESIZE, // GPU
        sizeof(double), false, false, true, false) ;

    /* If we failed to allocate memory, return. */
    if(!head || !idleTileCount || !bundleCount || !next || !prev || !triu
       || !Bundles || !gpuVT || !wsMongoVT)
    {
        FREE_EVERYTHING_BUCKET ;
        memory_ok = false ;
        return;
    }

    /* Initialize data structures */

    /* Initialize buckets */
    for (int i = 0; i < numBuckets; i++)
    {
        head[i] = EMPTY;
        idleTileCount[i] = bundleCount[i] = 0;
    }
    for (int i = 0; i < numRowTiles; i++)
    {
        next[i] = prev[i] = EMPTY;
        triu[i] = false;
    }

    /* Initialize VT structure */
    for(int i=0; i<numRowTiles; i++)
    {
        gpuVT[i] = (double*) wsMongoVT->gpu() + 33*32*i; // base + offset
    }
}

template BucketList<int32_t>::BucketList
(
    Front <int32_t> *F,
    int32_t minApplyGranularity
) ;
template BucketList<int64_t>::BucketList
(
    Front <int64_t> *F,
    int64_t minApplyGranularity
) ;

template <typename Int>
BucketList<Int>::~BucketList()
{
    FREE_EVERYTHING_BUCKET ;

}

template BucketList<int32_t>::~BucketList() ;
template BucketList<int64_t>::~BucketList() ;


template <typename Int>
void BucketList<Int>::Initialize()
{
    int fm = front->fm;
    int fn = front->fn;
    Int *Stair = front->Stair;

    int rowtile = 0;
    for (int colBucket = 0;
        colBucket < numBuckets && numIdleTiles < numRowTiles; colBucket++)
    {
        int lastcol = MIN(fn - 1, colBucket * TileSize + (TileSize - 1));
        int row = MIN(Stair[lastcol], fm - 1);
        row = MAX(row, lastcol); // Handle structural rank deficiency.

        int itile = row / TileSize;
        if (itile >= rowtile)
        {
            for (int i = itile; i >= rowtile; i--) { Insert(i, colBucket); }
            rowtile = itile + 1;
        }
    }
}

template void BucketList<int32_t>::Initialize() ;
template void BucketList<int64_t>::Initialize() ;
