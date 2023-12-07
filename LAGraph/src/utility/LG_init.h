//------------------------------------------------------------------------------
// LG_init.h: include file for use within LAGraph itself
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2023 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// These definitions are not meant for the user application that relies on
// LAGraph and/or GraphBLAS.  LG_* methods are for internal use in LAGraph.

#ifndef LG_INIT_H
#define LG_INIT_H

//------------------------------------------------------------------------------
// definitions used in LAGr_Init.c and for testing
//------------------------------------------------------------------------------

LAGRAPH_PUBLIC void LG_set_LAGr_Init_has_been_called (bool setting) ;
LAGRAPH_PUBLIC bool LG_get_LAGr_Init_has_been_called (void) ;

#endif

