// =============================================================================
// === GPUQREngine/Include/GPUQREngine_GraphVisHelper.hpp ======================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
//
// GraphVisHelper wraps logic to render the contents of a bucket list.
//   This is used primarilly in debugging efforts.
//
// =============================================================================

#ifndef GPUQRENGINE_GRAPHVIZHELPER_HPP
#define GPUQRENGINE_GRAPHVIZHELPER_HPP

#ifdef GPUQRENGINE_RENDER
#include "GPUQREngine_BucketList.hpp"

void GPUQREngine_RenderBuckets(BucketList *buckets);

#endif
#endif
