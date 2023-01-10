//------------------------------------------------------------------------------
// UMFPACK/Source/umf_kernel.h
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

Int UMF_kernel
(
    const Int Ap [ ],
    const Int Ai [ ],
    const double Ax [ ],
#ifdef COMPLEX
    const double Az [ ],
#endif
    NumericType *Numeric,
    WorkType *Work,
    SymbolicType *Symbolic
) ;
