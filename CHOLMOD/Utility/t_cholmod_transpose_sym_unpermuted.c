//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_transpose_sym_unpermuted
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------
// C = A' where A and C are symmetric
//------------------------------------------------------------------------------

// The including file must define or undef PACKED, NUMERIC, and LO.
// define PACKED: if A->packed is true, undefine if A->packed is false
// define LO: if A is symmetric lower, undefine it if A is upper
// define NUMERIC: if computing values and pattern of C, undefine it if 
//      computing just the column counts of C.

//------------------------------------------------------------------------------

{

    for (Int j = 0 ; j < n ; j++)
    {
        Int pa = Ap [j] ;
        #ifdef PACKED
        Int paend = Ap [j+1] ;
        #else
        Int paend = pa + Anz [j] ;
        #endif
        for ( ; pa < paend ; pa++)
        {
            // get A(i,j)
            Int i = Ai [pa] ;
            #ifdef LO
            // A is symmetric lower, C is symmetric upper
            if (i < j) continue ;
            #else
            // A is symmetric upper, C is symmetric lower
            if (i > j) continue ;
            #endif
            // C(j,i) = conj (A(i,j))
            Int pc = Wi [i]++ ;
            #ifdef NUMERIC
            ASSIGN_CONJ_OR_NCONJ (Cx, Cz, pc, Ax, Az, pa) ;
            Ci [pc] = j ;
            #endif
        }
    }
}

#undef PACKED
#undef LO

