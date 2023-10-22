//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_transpose_sym_permuted
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------
// C = A(p,p)' where A and C are symmetric
//------------------------------------------------------------------------------

// The including file must define or undef PACKED, NUMERIC, and LO.
// define PACKED: if A->packed is true, undefine if A->packed is false
// define LO: if A is symmetric lower, undefine it if A is upper
// define NUMERIC: if computing values and pattern of C, undefine it if 
//      computing just the column counts of C.

//------------------------------------------------------------------------------

{
    for (Int jold = 0 ; jold < n ; jold++)
    {
        Int jnew = Pinv [jold] ;
        Int pa = Ap [jold] ;
        #ifdef PACKED
        Int paend = Ap [jold+1] ;
        #else
        Int paend = pa + Anz [jold] ;
        #endif
        for ( ; pa < paend ; pa++)
        {
            // get A(iold,jold)
            Int iold = Ai [pa] ;
            Int inew = Pinv [iold] ;
            #ifdef LO
            // A is symmetric lower, C is symmetric upper
            if (iold < jold) continue ;
            if (inew > jnew)
            #else
            // A is symmetric upper, C is symmetric lower
            if (iold > jold) continue ;
            if (inew < jnew)
            #endif
            {
                // C(jnew,inew) = conj (A(iold,jold))
                Int pc = Wi [inew]++ ;
                #ifdef NUMERIC
                ASSIGN_CONJ_OR_NCONJ (Cx, Cz, pc, Ax, Az, pa) ; 
                Ci [pc] = jnew ;
                #endif
            }
            else
            {
                // C(inew,jnew) = A(iold,jold)
                Int pc = Wi [jnew]++ ;
                #ifdef NUMERIC
                ASSIGN (Cx, Cz, pc, Ax, Az, pa) ; 
                Ci [pc] = inew ;
                #endif
            }
        }
    }
}

#undef PACKED
#undef LO

