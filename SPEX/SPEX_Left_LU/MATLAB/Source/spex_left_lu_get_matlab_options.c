//------------------------------------------------------------------------------
// SPEX_Left_LU/MATLAB/spex_get_matlab_options: Set factorization options 
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "SPEX_Left_LU_mex.h"

// Purpose: This function reads in the necessary information from the options
// struct for MATLAB.

#define MATCH(s,t) (strcmp (s,t) == 0)

void spex_left_lu_get_matlab_options
(
    SPEX_options* option,           // Control parameters
    spex_mex_options *mexoptions,   // MATLAB-specific options
    const mxArray* input            // options struct, may be NULL
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    mxArray *field ;
    #define LEN 256
    char string [LEN+1] ;

    // true if input options struct is present 
    bool present = (input != NULL) && !mxIsEmpty (input) && mxIsStruct (input) ;

    //--------------------------------------------------------------------------
    // Get the column ordering
    //--------------------------------------------------------------------------

    option->order = SPEX_COLAMD ;     // default: COLAMD ordering
    field = present ? mxGetField (input, 0, "order") : NULL ;
    if (field != NULL)
    {
        if (!mxIsChar (field)) spex_left_lu_mex_error (1, "option.order must be a string") ;
        mxGetString (field, string, LEN) ;
        if (MATCH (string, "none"))
        {
            option->order = SPEX_NO_ORDERING ;  // None: A is factorized as-is
        }
        else if (MATCH (string, "colamd"))
        {
            option->order = SPEX_COLAMD ;       // COLAMD: Default
        }
        else if (MATCH (string, "amd"))
        {
            option->order = SPEX_AMD ;          // AMD
        }
        else
        {
            spex_left_lu_mex_error (1, "unknown option.order") ;
        }
    }

    //--------------------------------------------------------------------------
    // Get the row pivoting scheme
    //--------------------------------------------------------------------------

    option->pivot = SPEX_TOL_SMALLEST ;     // default: diag pivoting with tol
    field = present ? mxGetField (input, 0, "pivot") : NULL ;
    if (field != NULL)
    {
        if (!mxIsChar (field)) spex_left_lu_mex_error (1, "option.pivot must be a string") ;
        mxGetString (field, string, LEN) ;
        if (MATCH (string, "smallest"))
        {
            option->pivot = SPEX_SMALLEST ;         // Smallest pivot
        }
        else if (MATCH (string, "diagonal"))
        {
            option->pivot = SPEX_DIAGONAL ;         // Diagonal pivoting
        }
        else if (MATCH (string, "first"))
        {
            option->pivot = SPEX_FIRST_NONZERO ;    // First nonzero in column
        }
        else if (MATCH (string, "tol smallest"))
        {
            option->pivot = SPEX_TOL_SMALLEST ;     // diag pivoting with tol
                                                    // for smallest pivot
        }
        else if (MATCH (string, "tol largest"))
        {
            option->pivot = SPEX_TOL_LARGEST ;      // diag pivoting with tol
                                                    // for largest pivot.
        }
        else if (MATCH (string, "largest"))
        {
            option->pivot = SPEX_LARGEST ;          // Largest pivot
        }
        else
        {
            spex_left_lu_mex_error (1, "unknown option.pivot") ;
        }
    }

    //--------------------------------------------------------------------------
    // tolerance for row partial pivoting
    //--------------------------------------------------------------------------

    option->tol = 0.1 ;     // default tolerance is 0.1
    if (option->pivot == SPEX_TOL_SMALLEST || option->pivot == SPEX_TOL_LARGEST)
    {
        field = present ? mxGetField (input, 0, "tol") : NULL ;
        if (field != NULL)
        {
            option->tol = mxGetScalar (field) ;
            if (option->tol > 1 || option->tol <= 0)
            {
                spex_left_lu_mex_error (1, "invalid option.tol, "
                    "must be > 0 and <= 1") ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // Get the solution option
    //--------------------------------------------------------------------------

    mexoptions->solution = SPEX_SOLUTION_DOUBLE ;     // default x is double
    field = present ? mxGetField (input, 0, "solution") : NULL ;
    if (field != NULL)
    {
        mxGetString (field, string, LEN) ;
        if (MATCH (string, "vpa"))
        {
            mexoptions->solution = SPEX_SOLUTION_VPA ;  // return x as vpa
        }
        else if (MATCH (string, "char"))
        {
            mexoptions->solution = SPEX_SOLUTION_CHAR ;  // x as cell strings
        }
        else if (MATCH (string, "double"))
        {
            mexoptions->solution = SPEX_SOLUTION_DOUBLE ;  // x as double
        }
        else
        {
            spex_left_lu_mex_error (1, "unknown option.solution") ;
        }
    }

    //--------------------------------------------------------------------------
    // Get the digits option
    //--------------------------------------------------------------------------

    mexoptions->digits = 100 ;     // same as the MATLAB vpa default
    field = present ? mxGetField (input, 0, "digits") : NULL ;
    if (field != NULL)
    {
        double d = mxGetScalar (field) ;
        if (d != trunc (d) || d < 2 || d > (1 << 29))
        {
            // the MATLAB vpa requires digits between 2 and 2^29
            spex_left_lu_mex_error (1, "options.digits must be an integer "
                "between 2 and 2^29") ;
        }
        mexoptions->digits = (int32_t) d ;
    }

    //--------------------------------------------------------------------------
    // Get the print level
    //--------------------------------------------------------------------------

    option->print_level = 0 ;       // default is no printing
    field = present ? mxGetField (input, 0, "print") : NULL ;
    if (field != NULL)
    {
        // silently convert to an integer 0, 1, 2, or 3
        option->print_level = (int) mxGetScalar (field) ;
        option->print_level = SPEX_MIN (option->print_level, 3) ;
        option->print_level = SPEX_MAX (option->print_level, 0) ;
    }
}

