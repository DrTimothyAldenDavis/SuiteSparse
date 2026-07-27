//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/ParU/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>

#include "ParU.h"

int main (void)
{
    int ver [3] ;
    char date [128] ;
    ParU_Version (ver, date) ;
    printf ("ParU: %d.%d.%d, %s\n", ver [0], ver [1], ver [2], date) ;
    return 0;
}
