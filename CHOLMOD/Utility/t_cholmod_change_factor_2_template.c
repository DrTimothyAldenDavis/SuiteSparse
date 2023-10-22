//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_change_factor_2_template
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Converts a simplicial numeric factor: to/from LL' and LDL', either
// out-of-place or in-place.

#ifdef OUT_OF_PLACE
    // the new L is created in new space: Li2, Lx2, Lz2
    #define Li_NEW Li2
    #define Lx_NEW Lx2
    #define Lz_NEW Lx2
#else
    // L is modified in its existing space: Li, Lx, and Lz
    #define Li_NEW Li
    #define Lx_NEW Lx
    #define Lz_NEW Lz
#endif

#ifdef IN_PLACE
    // Li is not modified.  Entries in Lx and Lz are changed but not moved
    // to a different place in Lx and Lz
    #define p_NEW p
#else
    #define p_NEW pnew
#endif

//------------------------------------------------------------------------------

{
    Int pnew = 0 ;

    if (make_ll)
    {

        //----------------------------------------------------------------------
        // convert an LDL' factorization to LL'
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < n ; j++)
        {

            //------------------------------------------------------------------
            // get column L(:,j)
            //------------------------------------------------------------------

            Int p = Lp [j] ;
            Int len = Lnz [j] ;

            // get the diagonal entry, djj = D(j,j)
            Real djj [1] ;
            ASSIGN_REAL (djj, 0, Lx, p) ;

            if (djj [0] <= 0)
            {

                //--------------------------------------------------------------
                // the matrix is not positive definite
                //--------------------------------------------------------------

                // The matrix is not postive-definite and cannot be converted
                // to LL'.  The column L(:,j) is moved to its new space but not
                // numerically modified so it can be converted back to a valid
                // LDL' factorization.

                if (L->minor == (size_t) n)
                {
                    ERROR (CHOLMOD_NOT_POSDEF, "L not positive definite") ;
                    L->minor = j ;
                }

                #ifndef IN_PLACE
                for (Int k = 0 ; k < len ; k++)
                {
                    Li_NEW [p_NEW+k] = Li [p+k] ;
                    ASSIGN (Lx_NEW, Lz_NEW, p_NEW+k, Lx, Lz, p+k) ;
                }
                #endif

            }
            else
            {

                //--------------------------------------------------------------
                // convert L(:,j) from an LDL' factorization to LL'
                //--------------------------------------------------------------

                // L(j,j) = sqrt (D(j,j))
                Real ljj [1] ;
                ljj [0] = sqrt (djj [0]) ;
                #ifndef IN_PLACE
                Li_NEW [p_NEW] = j ;
                #endif
                ASSIGN_REAL (Lx_NEW, p_NEW, ljj, 0) ;
                CLEAR_IMAG (Lx_NEW, Lz_NEW, p_NEW) ;

                // L(j+1:n,j) = L(j+1:n,j) * L(j,j)
                for (Int k = 1 ; k < len ; k++)
                {
                    #ifndef IN_PLACE
                    Li_NEW [p_NEW+k] = Li [p+k] ;
                    #endif
                    MULT_REAL (Lx_NEW, Lz_NEW, p_NEW+k, Lx, Lz, p+k, ljj, 0) ;
                }
            }

            //------------------------------------------------------------------
            // grow column L(:,j) if requested, and advance to column j+1
            //------------------------------------------------------------------

            #ifndef IN_PLACE
            Lp [j] = p_NEW ;
            #ifdef OUT_OF_PLACE
            if (grow) len = grow_column (len, grow1, grow2, n-j) ;
            #endif
            p_NEW += len ;
            #endif
        }

        // log the end of the last column
        #ifndef IN_PLACE
        Lp [n] = p_NEW ;
        #endif

    }
    else if (make_ldl)
    {

        //----------------------------------------------------------------------
        // convert an LL' factorization to LDL'
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < n ; j++)
        {

            //------------------------------------------------------------------
            // get column L(:,j)
            //------------------------------------------------------------------

            Int len = Lnz [j] ;
            Int p = Lp [j] ;

            // get the diagonal entry, ljj = L(j,j)
            Real ljj [1] ;
            ASSIGN_REAL (ljj, 0, Lx, p) ;

            if (ljj [0] <= 0)
            {

                //--------------------------------------------------------------
                // the matrix is not positive definite
                //--------------------------------------------------------------

                // do not modify L(:,j), just copy it to its new place

                #ifndef IN_PLACE
                for (Int k = 0 ; k < len ; k++)
                {
                    Li_NEW [p_NEW+k] = Li [p+k] ;
                    ASSIGN (Lx_NEW, Lz_NEW, p_NEW+k, Lx, Lz, p+k) ;
                }
                #endif

            }
            else
            {

                //--------------------------------------------------------------
                // convert L(:,j) from an LL' factorization to LDL'
                //--------------------------------------------------------------

                // D(j,j) = L(j,j)^2
                #ifndef IN_PLACE
                Li_NEW [p_NEW] = j ;
                #endif
                Real djj [1] ;
                djj [0] = ljj [0] * ljj [0] ;
                ASSIGN_REAL (Lx_NEW, p_NEW, djj, 0) ;
                CLEAR_IMAG (Lx_NEW, Lz_NEW, p_NEW) ;

                // L(j+1:n) = L(j+1:n) / L(j,j)
                for (Int k = 1 ; k < len ; k++)
                {
                    #ifndef IN_PLACE
                    Li_NEW [p_NEW+k] = Li [p+k] ;
                    #endif
                    DIV_REAL (Lx_NEW, Lz_NEW, p_NEW+k, Lx, Lz, p+k, ljj, 0) ;
                }
            }

            //------------------------------------------------------------------
            // grow column L(:,j) if requested, and advance to column j+1
            //------------------------------------------------------------------

            #ifndef IN_PLACE
            Lp [j] = p_NEW ;
            #ifdef OUT_OF_PLACE
            if (grow) len = grow_column (len, grow1, grow2, n-j) ;
            #endif
            p_NEW += len ;
            #endif
        }

        // log the end of the last column
        #ifndef IN_PLACE
        Lp [n] = p_NEW ;
        #endif

    }
    else
    {

        //----------------------------------------------------------------------
        // factorization remains as is, but space may be revised
        //----------------------------------------------------------------------

        #ifndef IN_PLACE
        for (Int j = 0 ; j < n ; j++)
        {

            //------------------------------------------------------------------
            // get column L(:,j)
            //------------------------------------------------------------------

            Int len = Lnz [j] ;
            Int p = Lp [j] ;

            //------------------------------------------------------------------
            // move L(:,j) to its new position
            //------------------------------------------------------------------

            #ifdef TO_PACKED
            if (p_NEW < p)
            #endif
            {
                for (Int k = 0 ; k < len ; k++)
                {
                    Li_NEW [p_NEW+k] = Li [p+k] ;
                    ASSIGN (Lx_NEW, Lz_NEW, p_NEW+k, Lx, Lz, p+k) ;
                }
                Lp [j] = p_NEW ;
            }

            //------------------------------------------------------------------
            // grow column L(:,j) if requested, and advance to column j+1
            //------------------------------------------------------------------

            #ifdef OUT_OF_PLACE
            if (grow) len = grow_column (len, grow1, grow2, n-j) ;
            #endif
            p_NEW += len ;
        }

        // log the end of the last column
        Lp [n] = p_NEW ;
        #endif
    }
}

#undef OUT_OF_PLACE
#undef TO_PACKED
#undef IN_PLACE
#undef Li_NEW
#undef Lx_NEW
#undef Lz_NEW
#undef p_NEW

