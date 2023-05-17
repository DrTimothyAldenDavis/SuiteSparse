//------------------------------------------------------------------------------
// SPQR/Include/spqrgpu.hpp
//------------------------------------------------------------------------------

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#ifndef SPQRGPU_HPP_
#define SPQRGPU_HPP_

#include "GPUQREngine_SuiteSparse.hpp"

template <typename Int = int64_t> void spqrgpu_kernel
(
    spqr_blob <double, Int> *Blob    // contains the entire problem input/output
) ;


template <typename Int = int64_t>
void spqrgpu_kernel
(
    spqr_blob <Complex, Int> *Blob
) ;

template <typename Int>
void spqrgpu_computeFrontStaging
(
    // inputs, not modified on output
    Int numFronts,     // total number of fronts (nf in caller)
    Int *Parent,       // size nf+1, assembly tree (f=nf is placeholder)
    Int *Childp,       // size nf+2, children of f are
                        //      Child [Childp [f] ... Childp [f+1]-1]
    Int *Child,        // size nf+1.

    Int *Fm,           // size nf+1, front f has Fm [f] rows
    Int *Cm,           // size nf+1, front f has Cm [f] rows in contrib
    Int *Rp,           // size nf+1, Rj[Rp[f]...Rp[f+1]-1] are the cols in f
    Int *Sp,           // size m+1, row pointers for sparse matrix S
    Int *Sleft,        // size n+2, see spqr_stranspose for description
    Int *Super,        // size nf+1, front f pivotal cols are
                        //      Super[f]..Super[f+1]-1
    Int *Post,         // size nf+1, front f is kth, if f = Post [k]

    Int RimapSize,     // scalar, size of Rimap on the GPU (# of int's)
    Int RjmapSize,     // scalar, size of Rjmap on the GPU (# of int's)

    // output, not defined on input:
    bool *feasible,     // scalar, true if feasible, false if GPU memory too low
    Int *numStages,    // scalar, number of stages
    Int *Stagingp,     // size nf+2, fronts are in the list
                        //      Post [Stagingp [stage]...Stagingp[stage+1]-1]
    Int *StageMap,     // size nf, front f is in stage StageMap [f]

    size_t *FSize,      // size nf+1, FSize[stage]: size in bytes of MongoF
    size_t *RSize,      // size nf+1, Rsize[stage]: size in bytes of MongoR
    size_t *SSize,      // size nf+1, Ssize[stage]: size in bytes of S
    Int *FOffsets,     // size nf, front f in MondoF [FOffsets[f]...] on GPU
    Int *ROffsets,     // size nf, R block in MondoR [Roffsets[f]...] on CPU
    Int *SOffsets,     // size nf, S entries for front f are in
                        //      wsS [SOffsets[f]...]

    // input/output:
    cholmod_common *cc
);

template <typename Int>
void spqrgpu_buildAssemblyMaps
(
    Int numFronts,
    Int n,
    Int *Fmap,
    Int *Post,
    Int *Super,
    Int *Rp,
    Int *Rj,
    Int *Sleft,
    Int *Sp,
    Int *Sj,
    double *Sx,
    Int *Fm,
    Int *Cm,
    Int *Childp,
    Int *Child,
    Int *CompleteStair,
    int *CompleteRjmap,
    Int *RjmapOffsets,
    int *CompleteRimap,
    Int *RimapOffsets,
    SEntry *cpuS
);

#endif
