//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_error: CHOLMOD error handling
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

//------------------------------------------------------------------------------
// MESSAGE macro: print a message using the given printf function pointer
//------------------------------------------------------------------------------

#define MESSAGE(kind)                                           \
{                                                               \
    printf_function ("CHOLMOD " kind ":") ;                     \
    if (message != NULL) printf_function (" %s.", message) ;    \
    if (file    != NULL) printf_function (" file: %s", file) ;  \
    if (line    >  0   ) printf_function (" line: %d", line) ;  \
    printf_function ("\n") ;                                    \
    fflush (stdout) ;                                           \
    fflush (stderr) ;                                           \
}

//------------------------------------------------------------------------------
// cholmod_error
//------------------------------------------------------------------------------

int CHOLMOD(error)
(
    // input:
    int status,             // Common->status
    const char *file,       // source file where error occurred
    int line,               // line number where error occurred
    const char *message,    // error message to print
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;

    //--------------------------------------------------------------------------
    // set the error status
    //--------------------------------------------------------------------------

    Common->status = status ;

    //--------------------------------------------------------------------------
    // handle the error, unless we're inside a CHOLMOD try/catch block
    //--------------------------------------------------------------------------

    if (!(Common->try_catch))
    {

        //----------------------------------------------------------------------
        // print the error message, if permitted
        //----------------------------------------------------------------------

        #ifndef NPRINT
        int (*printf_function) (const char *, ...) ;
        printf_function = SuiteSparse_config_printf_func_get ( ) ;
        if (printf_function != NULL)
        {
            if (status > 0 && Common->print > 1)
            {
                // print a warning message
                MESSAGE ("warning") ;
            }
            else if (Common->print > 0)
            {
                // print an error message
                MESSAGE ("error") ;
            }
        }
        #endif

        //----------------------------------------------------------------------
        // call the user error handler, if present
        //----------------------------------------------------------------------

        if (Common->error_handler != NULL)
        {
            Common->error_handler (status, file, line, message) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (TRUE) ;
}

