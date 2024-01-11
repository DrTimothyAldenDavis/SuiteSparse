//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/LAGraph/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>

#include "LAGraph.h"

int main (void)
{
    int version [3];
    char msg [LAGRAPH_MSG_LEN], verstring [LAGRAPH_MSG_LEN];
    LAGraph_Init (msg);
    LAGraph_Version (version, verstring, msg);
    LAGraph_Finalize (msg);

    return 0;
}
