//------------------------------------------------------------------------------
// SuiteSparse/Example/[Config or Include]/my.h
//------------------------------------------------------------------------------

// Copyright (c) 2022-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// Example include file for a user library.  Do not edit the Include/my.h
// file, since it is constructed from Config/my.h.in by cmake.

// version and date for example user library
#define MY_DATE "Jan 12, 2024"
#define MY_MAJOR_VERSION 1
#define MY_MINOR_VERSION 6
#define MY_PATCH_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

void my_version (int version [3], char date [128]) ;
int my_function (void) ;
int my_check_version (const char *package, int major, int minor, int patch,
    const char *date, int version [3], long long unsigned int vercode) ;

#ifdef __cplusplus
}
#endif

