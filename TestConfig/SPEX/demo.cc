//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/SPEX/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>

#include "SPEX.h"

int main (void)
{
    SPEX_initialize ( );
    SPEX_finalize ( );

    return 0;
}
