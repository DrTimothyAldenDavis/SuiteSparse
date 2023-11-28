// =============================================================================
// === GPUQREngine/Source/GPUQREngine_UberKernel.cu ============================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
// 
// This is the actual concrete kernel invocation, transfering control flow to
// the GPU accelerator briefly. We actually launch kernels using alternating
// streams to overlap communication with computation, so the launch is actually
// asynchronous in nature. We use the CUDA events and streams model througout
// the Scheduler to coordinate asynchronous launch behavior.
// 
// =============================================================================

#ifdef SUITESPARSE_CUDA

#define CUDA_INCLUDE
#include "Kernel/uberKernel.cu"


void GPUQREngine_UberKernel
(
    cudaStream_t kernelStream,      // The stream on which to launch the kernel
    TaskDescriptor *gpuWorkQueue,   // The list of work items for the GPU
    int numTasks                    // The # of items in the work list
)
{
    /* Set the standard launch configuration. */
    dim3 threads(NUMTHREADS, 1);
    dim3 grid(numTasks, 1);

    /* Launch the kernel */
    qrKernel<<<grid, threads, 0, kernelStream>>>(gpuWorkQueue, numTasks);    
}

#endif

