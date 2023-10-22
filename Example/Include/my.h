//------------------------------------------------------------------------------
// SuiteSparse/Example/[Config or Include]/my.h
//------------------------------------------------------------------------------

// Copyright (c) 2022-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// Example include file for a user library.  Do not edit the Include/my.h
// file, since it is constructed from Config/my.h.in by cmake.

// version and date for example user library
#define MY_DATE "Oct 23, 2023"
#define MY_MAJOR_VERSION 1
#define MY_MINOR_VERSION 4
#define MY_PATCH_VERSION 3

#ifdef __cplusplus
extern "C" {
#endif

void my_library (int version [3], char date [128]) ;
void my_function (void) ;

#ifdef __cplusplus
}
#endif
