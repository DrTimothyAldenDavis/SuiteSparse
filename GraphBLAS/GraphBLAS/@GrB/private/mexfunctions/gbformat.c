//------------------------------------------------------------------------------
// gbformat: get/set the matrix format to use in GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Usage

// f = gbformat ;
// f = gbformat (f) ;

#include "gb_matlab.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin <= 1 && nargout <= 1,
        "usage: f = GrB.format or GrB.format (f)") ;

    //--------------------------------------------------------------------------
    // get/set the format
    //--------------------------------------------------------------------------

    GxB_Format_Value fmt ;

    if (nargin == 0)
    { 

        //----------------------------------------------------------------------
        // format = GrB.format
        //----------------------------------------------------------------------

        // get the global format
        OK (GxB_Global_Option_get (GxB_FORMAT, &fmt)) ;

    }
    else // if (nargin == 1)
    {

        if (mxIsChar (pargin [0]))
        { 

            //------------------------------------------------------------------
            // GrB.format (format)
            //------------------------------------------------------------------

            // set the global format
            fmt = gb_mxstring_to_format (pargin [0]) ;
            OK (GxB_Global_Option_set (GxB_FORMAT, fmt)) ;

        }
        else
        { 

            //------------------------------------------------------------------
            // GrB.format (G)
            //------------------------------------------------------------------

            // get the format of the input matrix G
            mxArray *opaque = mxGetField (pargin [0], 0, "s") ;
            CHECK_ERROR (opaque == NULL, "invalid GraphBLAS struct") ;
            int64_t *s = mxGetInt64s (opaque) ;
            bool is_csc = (bool) (s [6]) ;
            fmt = (is_csc) ? GxB_BY_COL : GxB_BY_ROW ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    if (fmt == GxB_BY_ROW)
    { 
        pargout [0] = mxCreateString ("by row") ;
    }
    else
    { 
        pargout [0] = mxCreateString ("by col") ;
    }
    GB_WRAPUP ;
}

