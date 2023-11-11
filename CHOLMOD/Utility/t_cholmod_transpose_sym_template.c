//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_transpose_sym_template:
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------
// C = A' or A(p,p)', either symbolic or numeric phase
//------------------------------------------------------------------------------

// The including file must define or undef NUMERIC.
// define NUMERIC: if computing values and pattern of C, undefine it if
//      computing just the column counts of C.

//------------------------------------------------------------------------------

{
    if (Pinv == NULL)
    {

        //----------------------------------------------------------------------
        // C = A', unpermuted transpose
        //----------------------------------------------------------------------

        if (A->stype < 0)
        {

            //------------------------------------------------------------------
            // C = A', A is symmetric lower, C will be symmetric upper
            //------------------------------------------------------------------

            if (A->packed)
            {
                // A is packed
                #define PACKED
                #define LO
                #include "t_cholmod_transpose_sym_unpermuted.c"
            }
            else
            {
                // A is unpacked
                #define LO
                #include "t_cholmod_transpose_sym_unpermuted.c"
            }

        }
        else
        {

            //------------------------------------------------------------------
            // C = A', A is symmetric upper, C will be symmetric lower
            //------------------------------------------------------------------

            if (A->packed)
            {
                // A is packed
                #define PACKED
                #include "t_cholmod_transpose_sym_unpermuted.c"
            }
            else
            {
                // A is unpacked
                #include "t_cholmod_transpose_sym_unpermuted.c"
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // C = A(p,p)', permuted transpose
        //----------------------------------------------------------------------

        if (A->stype < 0)
        {

            //------------------------------------------------------------------
            // C = A(p,p)', A is symmetric lower, C will be symmetric upper
            //------------------------------------------------------------------

            if (A->packed)
            {
                // A is packed
                #define PACKED
                #define LO
                #include "t_cholmod_transpose_sym_permuted.c"
            }
            else
            {
                // A is unpacked
                #define LO
                #include "t_cholmod_transpose_sym_permuted.c"
            }

        }
        else
        {

            //------------------------------------------------------------------
            // C = A(p,p)', A is symmetric upper, C will be symmetric lower
            //------------------------------------------------------------------

            if (A->packed)
            {
                // A is packed
                #define PACKED
                #include "t_cholmod_transpose_sym_permuted.c"
            }
            else
            {
                // A is unpacked
                #include "t_cholmod_transpose_sym_permuted.c"
            }
        }
    }
}

#undef NUMERIC

