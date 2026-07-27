//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/SuiteSparse_config/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>

#include "SuiteSparse_config.h"

int main (void)
{
    int version [3];
    SuiteSparse_version (version);
    std::cout << "SuiteSparse_config version: " << version[0] << "."
        << version[1] << "." << version[2] << std::endl;

    return 0;
}
