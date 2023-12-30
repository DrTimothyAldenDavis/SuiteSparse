// =============================================================================
// === GPUQREngine/Include/Kernel/Factorize/factorize_3_by_1.cu ================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
// 96-by-32 factorize, no VT or tiles, edge case.  384 threads   WHOLE FRONT
//------------------------------------------------------------------------------

#define FACTORIZE   factorize_96_by_32
#define M           (96)
#define N           (32)
#define BITTYROWS   (8)
#define WHOLE_FRONT
#include "Kernel/Factorize/factorize_vt.cu"
