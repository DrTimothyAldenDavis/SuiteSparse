// =============================================================================
// === GPUQREngine/Include/Kernel/Apply/block_apply_2.cu =======================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
// block_apply_2: handles all edge cases and any number of column tiles
//------------------------------------------------------------------------------

#define BLOCK_APPLY block_apply_2
#define ROW_PANELSIZE 2
#define COL_PANELSIZE 2
#include "block_apply.cu"
