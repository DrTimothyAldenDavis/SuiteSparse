//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_change_xdtype_template: change xtype and/or dtype
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Changes the xtype and/or dtype of X and Z.
// X and Z are arrays of size 0, nz, 2*nz, and are modified (in/out).

#include "cholmod_internal.h"

#define TRY(statement)                                                  \
    {                                                                   \
        statement ;                                                     \
        if (Common->status < CHOLMOD_OK)                                \
        {                                                               \
            CHOLMOD(free) (nz, exout, X_output, Common) ;               \
            CHOLMOD(free) (nz, ezout, Z_output, Common) ;               \
            return (FALSE) ;                                            \
        }                                                               \
    }

//------------------------------------------------------------------------------

static int CHANGE_XDTYPE2
(
    Int nz,                 // # of entries in X and/or Z

    int *input_xtype,       // changed to output_xtype on output
            // CHOLMOD_PATTERN: X and Z are NULL on input
            // CHOLMOD_REAL:    X is size nz, and Z is NULL on input
            // CHOLMOD_COMPLEX: X is size 2*nz, and Z is NULL on input
            // CHOLMOD_ZOMPLEX: X and Z are size nz on input

    int output_xtype,
            // CHOLMOD_PATTERN: X and Z are NULL on output
            // CHOLMOD_REAL:    X is size nz, and Z is NULL on output
            // CHOLMOD_COMPLEX: X is size 2*nz, and Z is NULL on output
            // CHOLMOD_ZOMPLEX: X and Z are size nz on output

    int *input_dtype,       // changed to output_dtype on output
    int output_dtype,

    // input/output:
    void **X,               // &X on input, reallocated on output
    void **Z,               // &Z on input, reallocated on output

    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    Common->status = CHOLMOD_OK ;
    if (*input_xtype == output_xtype && sizeof (Real_IN) == sizeof (Real_OUT))
    {
        // nothing to do
        return (TRUE) ;
    }

    //--------------------------------------------------------------------------
    // get the entry sizes of the input and output
    //--------------------------------------------------------------------------

    size_t ein   = sizeof (Real_IN) ;
    size_t exin  = ein  * (((*input_xtype) == CHOLMOD_PATTERN) ? 0 :
                          (((*input_xtype) == CHOLMOD_COMPLEX) ? 2 : 1)) ;
    size_t ezin  = ein  * (((*input_xtype) == CHOLMOD_ZOMPLEX) ? 1 : 0) ;

    size_t eout  = sizeof (Real_OUT) ;
    size_t exout = eout * (((output_xtype) == CHOLMOD_PATTERN) ? 0 :
                          (((output_xtype) == CHOLMOD_COMPLEX) ? 2 : 1)) ;
    size_t ezout = eout * (((output_xtype) == CHOLMOD_ZOMPLEX) ? 1 : 0) ;

    //--------------------------------------------------------------------------
    // convert the array
    //--------------------------------------------------------------------------

    Real_IN  *X_input  = (Real_IN *) *X ;
    Real_IN  *Z_input  = (Real_IN *) *Z ;
    Real_OUT *X_output = NULL ;
    Real_OUT *Z_output = NULL ;

    switch (output_xtype)
    {

        //----------------------------------------------------------------------
        case CHOLMOD_PATTERN: // convert any xtype to pattern
        //----------------------------------------------------------------------

            break ;

        //----------------------------------------------------------------------
        case CHOLMOD_REAL: // convert any xtype to real
        //----------------------------------------------------------------------

            TRY (X_output = CHOLMOD(malloc) (nz, exout, Common)) ;

            switch (*input_xtype)
            {

                case CHOLMOD_PATTERN:   // pattern to real

                    for (Int k = 0 ; k < nz ; k++)
                    {
                        X_output [k] = 1 ;
                    }
                    break ;

                case CHOLMOD_REAL:      // real to real

                    for (Int k = 0 ; k < nz ; k++)
                    {
                        X_output [k] = (Real_OUT) X_input [k] ;
                    }
                    break ;

                case CHOLMOD_COMPLEX:   // complex to real

                    for (Int k = 0 ; k < nz ; k++)
                    {
                        X_output [k] = (Real_OUT) X_input [2*k] ;
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:   // zomplex to real

                    for (Int k = 0 ; k < nz ; k++)
                    {
                        X_output [k] = (Real_OUT) X_input [k] ;
                    }
                    break ;

            }
            break ;

        //----------------------------------------------------------------------
        case CHOLMOD_COMPLEX:   // convert any xtype to complex
        //----------------------------------------------------------------------

            TRY (X_output = CHOLMOD(malloc) (nz, exout, Common)) ;

            switch (*input_xtype)
            {

                case CHOLMOD_PATTERN:   // pattern to complex

                    for (Int k = 0 ; k < nz ; k++)
                    {
                        X_output [2*k  ] = 1 ;
                        X_output [2*k+1] = 0 ;
                    }
                    break ;

                case CHOLMOD_REAL:      // real to complex

                    for (Int k = 0 ; k < nz ; k++)
                    {
                        X_output [2*k  ] = (Real_OUT) X_input [k] ;
                        X_output [2*k+1] = 0 ;
                    }
                    break ;

                case CHOLMOD_COMPLEX:   // complex to complex

                    for (Int k = 0 ; k < 2*nz ; k++)
                    {
                        X_output [k] = (Real_OUT) X_input [k] ;
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:   // zomplex to complex

                    for (Int k = 0 ; k < nz ; k++)
                    {
                        X_output [2*k  ] = (Real_OUT) X_input [k] ;
                        X_output [2*k+1] = (Real_OUT) Z_input [k] ;
                    }
                    break ;
            }
            break ;

        //----------------------------------------------------------------------
        case CHOLMOD_ZOMPLEX:   // convert any xtype to zomplex
        //----------------------------------------------------------------------

            TRY (X_output = CHOLMOD(malloc) (nz, exout, Common)) ;
            TRY (Z_output = CHOLMOD(malloc) (nz, ezout, Common)) ;

            switch (*input_xtype)
            {
                case CHOLMOD_PATTERN:   // pattern to zomplex

                    for (Int k = 0 ; k < nz ; k++)
                    {
                        X_output [k] = 1 ;
                        Z_output [k] = 0 ;
                    }
                    break ;

                case CHOLMOD_REAL:      // real to zomplex

                    for (Int k = 0 ; k < nz ; k++)
                    {
                        X_output [k] = (Real_OUT) X_input [k] ;
                        Z_output [k] = 0 ;
                    }
                    break ;

                case CHOLMOD_COMPLEX:   // complex to zomplex

                    for (Int k = 0 ; k < nz ; k++)
                    {
                        X_output [k] = (Real_OUT) X_input [2*k  ] ;
                        Z_output [k] = (Real_OUT) X_input [2*k+1] ;
                    }
                    break ;

                case CHOLMOD_ZOMPLEX:   // zomplex to zomplex

                    for (Int k = 0 ; k < nz ; k++)
                    {
                        X_output [k] = (Real_OUT) X_input [k] ;
                        Z_output [k] = (Real_OUT) Z_input [k] ;
                    }
                    break ;

            }
            break ;

    }

    //--------------------------------------------------------------------------
    // free the input arrays and return result
    //--------------------------------------------------------------------------

    CHOLMOD(free) (nz, exin, X_input, Common) ;
    CHOLMOD(free) (nz, ezin, Z_input, Common) ;
    (*X) = X_output ;
    (*Z) = Z_output ;
    (*input_xtype) = output_xtype ;
    (*input_dtype) = output_dtype ;
    return (TRUE) ;
}

#undef CHANGE_XDTYPE2
#undef Real_IN
#undef Real_OUT

