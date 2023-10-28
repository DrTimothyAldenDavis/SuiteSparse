//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_transpose_unsym_template
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------
// C = A', C = A(:,f)', C = A(p,:)', or C = A(p,f)'; A and C are unsymmetric
//------------------------------------------------------------------------------

// The including file must define or undef PACKED, NUMERIC, and FSET

// define PACKED: if A->packed is true, undefine if A->packed is false

// define NUMERIC: if computing values and pattern of C, undefine it if 
//      computing just the column counts of C.

//------------------------------------------------------------------------------

{

    #undef  N
    #ifdef FSET
    #define N nf
    #else
    #define N ncol
    #endif

    for (Int k = 0 ; k < N ; k++)
    {

        //----------------------------------------------------------------------
        // get the fset, if present
        //----------------------------------------------------------------------

        #ifdef FSET
	Int j = fset [k] ;
        #else
        Int j = k ;
        #endif

        //----------------------------------------------------------------------
        // get A(:,k)
        //----------------------------------------------------------------------

	Int p = Ap [j] ;
        #ifdef PACKED
	Int pend = Ap [j+1] ;
        #else
	Int pend = p + Anz [j] ;
        #endif

        //----------------------------------------------------------------------
        // scan entries in A(:,k)
        //----------------------------------------------------------------------

	for ( ; p < pend ; p++)
	{
            // get A(i,j) and count it or get its place in C
	    Int pc = Wi [Ai [p]]++ ;
            #ifdef NUMERIC
            // C(j,i) = conj (A(i,j))
	    ASSIGN_CONJ_OR_NCONJ (Cx, Cz, pc, Ax, Az, p) ;
	    Ci [pc] = j ;
            #endif
	}
    }
}

#undef FSET
#undef PACKED

