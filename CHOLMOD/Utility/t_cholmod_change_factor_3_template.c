//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_change_factor_3_template
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

{
    Int p = 0 ;     // current position in simplicial L

    for (Int s = 0 ; s < nsuper ; s++)
    {

        //------------------------------------------------------------------
        // get the Supernode (0:nsrow-1, 0:nscol-1)
        //------------------------------------------------------------------

        Int k1 = Super [s] ;    // 1st column of supernode s
        Int k2 = Super [s+1] ;  // k2-1 is last column of supernode s
        Int psi = Lpi [s] ;     // Ls [psi:psend-1]: pattern of s
        Int psend = Lpi [s+1] ;
        Int psx = Lpx [s] ;     // Lx [psx:..]: values of supernode s
        Int nsrow = psend - psi ;   // # of rows in 1st column of s
        Int nscol = k2 - k1 ;       // # of cols in the supernode

        //------------------------------------------------------------------
        // convert the Supernode to the simplicial L (k1,k2-1)
        //------------------------------------------------------------------

        for (Int jj = 0 ; jj < nscol ; jj++)
        {

            //--------------------------------------------------------------
            // convert L(:,j) from Supernode (:,jj)
            //--------------------------------------------------------------

            Int j = jj + k1 ;           // L(:,j) is Supernode column jj
            Lnz [j] = nsrow - jj ;      // # entries in L(:,j)

            //--------------------------------------------------------------
            // log the start of L(:,j) in the simplicial form of L
            //--------------------------------------------------------------

            Int ii = jj ;
            Int q = psx + jj + jj*nsrow ;
            Lp [j] = TO_PACKED ? p:q ;

            //--------------------------------------------------------------
            // handle the diagonal for LDL'
            //--------------------------------------------------------------

            Real ljj [1] ;
            if (!TO_LL)
            {
                // create the row index for L(j,j)
                Li [TO_PACKED ? p:q] = j ;

                // compute D(j,j)
                ASSIGN_REAL (ljj, 0, Lx, q) ;
                if (ljj [0] <= 0)
                {
                    // not positive definite: values of L(:,j) not changed
                    ASSIGN_REAL (Lx, TO_PACKED ? p:q, ljj, 0) ;
                    CLEAR_IMAG (Lx, , TO_PACKED ? p:q) ;
                    ljj [0] = 1 ;
                }
                else
                {
                    // D(j,j) = L (j,j)^2
                    Real djj [1] ;
                    djj [0] = ljj [0] * ljj [0] ;
                    ASSIGN_REAL (Lx, TO_PACKED ? p:q, djj, 0) ;
                    CLEAR_IMAG (Lx, , TO_PACKED ? p:q) ;
                }

                // advance to the first off-diagonal entry of L(:,j)
                ii++ ;
                p++ ;
            }

            //--------------------------------------------------------------
            // convert the remainder of L(:,j)
            //--------------------------------------------------------------

            for ( ; ii < nsrow ; ii++, p++)
            {
                // create the row index for L(i,j)
                q = psx + ii + jj*nsrow ;
                Li [TO_PACKED ? p:q] = Ls [psi + ii] ;
                // revise the numeric value of L(i,j)
                if (TO_LL)
                {
                    // for LL': L(i,j) is unchanged; move to new position
                    if (TO_PACKED)
                    {
                        ASSIGN (Lx, , p, Lx, , q) ;
                    }
                }
                else
                {
                    // for LDL': L(i,j) = L(i,j) / L(j,j)
                    DIV_REAL (Lx, , TO_PACKED ? p:q, Lx, , q, ljj, 0) ;
                }
            }
        }
    }

    // log the end of the last column in L
    Lp [n] = TO_PACKED ? p : Lpx [nsuper] ;
}

#undef TO_PACKED
#undef TO_LL

