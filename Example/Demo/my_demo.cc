//------------------------------------------------------------------------------
// SuiteSparse/Example/Demo/my_demo.c
//------------------------------------------------------------------------------

// Copyright (c) 2022-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// Example user program

#include <iostream>

#include "my.h"

int main (void)
{
    std::cout << "My demo" << std::endl ;
    int version [3] ;
    char date [128] ;
    my_version (version, date) ;
    std::cout << "Date from #include 'my.h':  " << MY_DATE << std::endl ;
    std::cout << "Date from compiled library: " << date << std::endl ;
    std::cout << "version from #include 'my.h.': " <<
        MY_MAJOR_VERSION << '.' << MY_MINOR_VERSION << '.' << MY_PATCH_VERSION << std::endl ;
    std::cout << "version from compiled library: " <<
        version [0] << '.' << version [1] << '.' << version [2] << std::endl ;

    int result = my_function ( ) ;
    return (result) ;
}

