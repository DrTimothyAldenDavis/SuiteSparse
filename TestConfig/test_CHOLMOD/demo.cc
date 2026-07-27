//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/CHOLMOD/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>

#include "cholmod.h"

int main (void)
{
    cholmod_common cc;
    cholmod_l_start (&cc);
    cholmod_l_finish (&cc);

    return 0;
}
