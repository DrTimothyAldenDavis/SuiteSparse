// =============================================================================
// === GPUQREngine/Include/Kernel/Assemble/sAssemble.cu ========================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

__device__ void sassemble ( )
{
    double *F = myTask.F;
    SEntry *S = (SEntry*) myTask.AuxAddress[0];
//  int Scount = myTask.extra[0];
    int pstart = myTask.extra[1];
    int pend = myTask.extra[2];

    /* Unpack the S entries and shove into their proper locations. */
    for(int p=pstart+threadIdx.x; p<pend; p+=blockDim.x)
    {
        SEntry e = S[p];
        F[e.findex] = e.value;
    }
}
