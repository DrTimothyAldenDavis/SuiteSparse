//------------------------------------------------------------------------------
// SPEX/MATLAB/spex_mex_get_matlab_options.c: Get command options from user
//------------------------------------------------------------------------------

// SPEX: (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function reads in the necessary information from the options
 * struct for MATLAB.
 * 
 * Note that default values for the SPEX_options struct are already set in the 
 * caller and thus default values only need to be set for the MATLAB-specific
 * options.
 */


#include "SPEX_mex.h"

#define MATCH(s,t) (strcmp (s,t) == 0)
#define SPEX_MIN(a,b) ((a) < (b) ? (a) : (b))
#define SPEX_MAX(a,b) ((a) > (b) ? (a) : (b))

void spex_mex_get_matlab_options
(
    SPEX_options option,            // Control parameters
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
    bool present = (input != NULL) && !mxIsEmpty (input) && mxIsStruct (input);

    //--------------------------------------------------------------------------
    // Get the column ordering
    //--------------------------------------------------------------------------

    // If the field is present, overwrite the default with the user input.
    field = present ? mxGetField (input, 0, "order") : NULL ;
    if (field != NULL)
    {
        if (!mxIsChar (field))
        {
            spex_mex_error (1, "option.order must be a string");
        }
        mxGetString (field, string, LEN);
        if (MATCH (string, "none"))
        {
            option->order = SPEX_NO_ORDERING ;  // None: A is factorized as-is
        }
        else if (MATCH (string, "colamd"))
        {
            option->order = SPEX_COLAMD ;       // COLAMD
        }
        else if (MATCH (string, "amd"))
        {
            option->order = SPEX_AMD ;          // AMD
        }
        else if (MATCH (string, "default"))
        {
            // COLAMD for LU; AMD otherwise
            option->order = SPEX_DEFAULT_ORDERING ;
        }
        else
        {
            spex_mex_error (1, "unknown option.order");
        }
    }

    //--------------------------------------------------------------------------
    // Get the row pivoting scheme
    //--------------------------------------------------------------------------

    // If the field is present, overwrite the default with the user input.
    field = present ? mxGetField (input, 0, "pivot") : NULL ;
    if (field != NULL)
    {
        if (!mxIsChar (field))
        {
            spex_mex_error (1, "option.pivot must be a string");
        }
        mxGetString (field, string, LEN);
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
            spex_mex_error (1, "unknown option.pivot");
        }
    }

    //--------------------------------------------------------------------------
    // tolerance for row partial pivoting
    //--------------------------------------------------------------------------

    // If we are utilizing tolerance based pivoting and the field is present
    // overwrite the default with the user input.
    if (option->pivot == SPEX_TOL_SMALLEST || option->pivot == SPEX_TOL_LARGEST)
    {
        field = present ? mxGetField (input, 0, "tol") : NULL ;
        if (field != NULL)
        {
            option->tol = mxGetScalar (field);
            if (option->tol > 1 || option->tol <= 0)
            {
                spex_mex_error (1, "invalid option.tol, "
                    "must be > 0 and <= 1");
            }
        }
    }

    //--------------------------------------------------------------------------
    // Get the solution option
    //--------------------------------------------------------------------------

    // By default, matlab will return a double solution unless specified
    // otherwise
    mexoptions->solution = SPEX_SOLUTION_DOUBLE ;
    field = present ? mxGetField (input, 0, "solution") : NULL ;
    if (field != NULL)
    {
        mxGetString (field, string, LEN);
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
            spex_mex_error (1, "unknown option.solution");
        }
    }

    //--------------------------------------------------------------------------
    // Get the digits option
    //--------------------------------------------------------------------------

    // MATLAB default for vpa
    mexoptions->digits = 100 ;
    field = present ? mxGetField (input, 0, "digits") : NULL ;
    if (field != NULL)
    {
        double d = mxGetScalar (field);
        if (d != trunc (d) || d < 2 || d > (1 << 29))
        {
            // the MATLAB vpa requires digits between 2 and 2^29
            spex_mex_error (1, "options.digits must be an integer "
                "between 2 and 2^29");
        }
        mexoptions->digits = (int32_t) d ;
    }

    //--------------------------------------------------------------------------
    // Get the print level
    //--------------------------------------------------------------------------

    // If the field is present, overwrite the default with the user input.
    field = present ? mxGetField (input, 0, "print") : NULL ;
    if (field != NULL)
    {
        // silently convert to an integer 0, 1, 2, or 3
        option->print_level = (int) mxGetScalar (field);
        option->print_level = SPEX_MIN (option->print_level, 3);
        option->print_level = SPEX_MAX (option->print_level, 0);
    }
}

