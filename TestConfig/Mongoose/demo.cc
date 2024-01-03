//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/Mongoose/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>

#include "Mongoose.hpp"

int main (void)
{
    // FIXME: Do something more meaningful
    std::cout << "Mongoose::mongoose_version(): " <<
        Mongoose::mongoose_version ( ) << std::endl;

    return 0;
}
