//------------------------------------------------------------------------------
// SPQR/Include/spqrgpu.hpp
//------------------------------------------------------------------------------

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#ifndef SPQRGPU_HPP_
#define SPQRGPU_HPP_

#include "GPUQREngine_SuiteSparse.hpp"

template <typename Entry, typename Int = int64_t> void spqrgpu_kernel
(
    spqr_blob <Entry, Int> *Blob    // contains the entire problem input/output
) ;

void spqrgpu_computeFrontStaging
(
    // inputs, not modified on output
    int64_t numFronts,     // total number of fronts (nf in caller)
    int64_t *Parent,       // size nf+1, assembly tree (f=nf is placeholder)
    int64_t *Childp,       // size nf+2, children of f are
                        //      Child [Childp [f] ... Childp [f+1]-1]
    int64_t *Child,        // size nf+1.

    int64_t *Fm,           // size nf+1, front f has Fm [f] rows
    int64_t *Cm,           // size nf+1, front f has Cm [f] rows in contrib
    int64_t *Rp,           // size nf+1, Rj[Rp[f]...Rp[f+1]-1] are the cols in f
    int64_t *Sp,           // size m+1, row pointers for sparse matrix S
    int64_t *Sleft,        // size n+2, see spqr_stranspose for description
    int64_t *Super,        // size nf+1, front f pivotal cols are
                        //      Super[f]..Super[f+1]-1
    int64_t *Post,         // size nf+1, front f is kth, if f = Post [k]

    int64_t RimapSize,     // scalar, size of Rimap on the GPU (# of int's)
    int64_t RjmapSize,     // scalar, size of Rjmap on the GPU (# of int's)

    // output, not defined on input:
    bool *feasible,     // scalar, true if feasible, false if GPU memory too low
    int64_t *numStages,    // scalar, number of stages
    int64_t *Stagingp,     // size nf+2, fronts are in the list
                        //      Post [Stagingp [stage]...Stagingp[stage+1]-1]
    int64_t *StageMap,     // size nf, front f is in stage StageMap [f]

    size_t *FSize,      // size nf+1, FSize[stage]: size in bytes of MongoF
    size_t *RSize,      // size nf+1, Rsize[stage]: size in bytes of MongoR
    size_t *SSize,      // size nf+1, Ssize[stage]: size in bytes of S
    int64_t *FOffsets,     // size nf, front f in MondoF [FOffsets[f]...] on GPU
    int64_t *ROffsets,     // size nf, R block in MondoR [Roffsets[f]...] on CPU
    int64_t *SOffsets,     // size nf, S entries for front f are in
                        //      wsS [SOffsets[f]...]

    // input/output:
    cholmod_common *cc
);

void spqrgpu_buildAssemblyMaps
(
    int64_t numFronts,
    int64_t n,
    int64_t *Fmap,
    int64_t *Post,
    int64_t *Super,
    int64_t *Rp,
    int64_t *Rj,
    int64_t *Sleft,
    int64_t *Sp,
    int64_t *Sj,
    double *Sx,
    int64_t *Fm,
    int64_t *Cm,
    int64_t *Childp,
    int64_t *Child,
    int64_t *CompleteStair,
    int *CompleteRjmap,
    int64_t *RjmapOffsets,
    int *CompleteRimap,
    int64_t *RimapOffsets,
    SEntry *cpuS
);

#endif
