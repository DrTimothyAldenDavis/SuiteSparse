//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/GraphBLAS/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>

#include "GraphBLAS.h"

int main (void)
{
    int version [3];
    GrB_init (GrB_NONBLOCKING);
    GxB_Global_Option_get (GxB_LIBRARY_VERSION, version);
    GrB_finalize ();

    return 0;
}
