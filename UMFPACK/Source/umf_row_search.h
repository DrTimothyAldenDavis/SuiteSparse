//------------------------------------------------------------------------------
// UMFPACK/Source/umf_row_search.h
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

Int UMF_row_search
(
    NumericType *Numeric,
    WorkType *Work,
    SymbolicType *Symbolic,
    Int cdeg0,
    Int cdeg1,
    const Int Pattern [ ],
    const Int Pos [ ],
    Int pivrow [2],
    Int rdeg [2],
    Int W_i [ ],
    Int W_o [ ],
    Int prior_pivrow [2],
    const Entry Wxy [ ],
    Int pivcol,
    Int freebie [2]
) ;

#define IN 0
#define OUT 1

#define IN_IN 0
#define IN_OUT 1
#define OUT_IN 2
#define OUT_OUT 3
