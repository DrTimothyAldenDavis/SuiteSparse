//------------------------------------------------------------------------------
// gbdegree: number of entries in each vector of a GraphBLAS matrix struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The input may be either a GraphBLAS matrix struct or a standard MATLAB
// sparse matrix.

//  gbdegree (X, 'row')     row degree
//  gbdegree (X, 'col')     column degree
//  gbdegree (X, true)      native (get degree of each vector):
//                          row degree if X is held by row,
//                          col degree if X is held by col.
//  gbdegree (X, false)     non-native (sum across vectors):
//                          col degree if X is held by row,
//                          row degree if X is held by col.

#include "gb_matlab.h"

#define USAGE "usage: degree = gbdegree (X, dim)"

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

    gb_usage (nargin == 2 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the inputs 
    //--------------------------------------------------------------------------

    GrB_Matrix X = gb_get_shallow (pargin [0]) ;
    GxB_Format_Value fmt ;
    OK (GxB_Matrix_Option_get (X, GxB_FORMAT, &fmt)) ;

    bool native ;
    if (mxIsChar (pargin [1]))
    {
        #define LEN 256
        char dim_string [LEN+2] ;
        gb_mxstring_to_string (dim_string, LEN, pargin [1], "dim") ;
        if (MATCH (dim_string, "row"))
        { 
            native = (fmt == GxB_BY_ROW) ;
        }
        else // if (MATCH (dim_string, "col"))
        { 
            native = (fmt == GxB_BY_COL) ;
        }
    }
    else
    { 
        native = (mxGetScalar (pargin [1]) != 0) ;
    }

    //--------------------------------------------------------------------------
    // get the degree of each row or column of X
    //--------------------------------------------------------------------------

    int64_t *degree = NULL ;
    GrB_Index *list = NULL, nvec = 0 ;
    GrB_Vector d = NULL ;

    if (native)
    { 

        //----------------------------------------------------------------------
        // get the degree of each vector of X
        //----------------------------------------------------------------------

        if (!GB_matlab_helper9 (X, &degree, &list, &nvec))
        {
            ERROR ("out of memory") ;
        }
        OK (GxB_Vector_import (&d, GrB_INT64, X->vdim, nvec, &list, &degree,
            NULL)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // get the degree of each index of X
        //----------------------------------------------------------------------

        GrB_Index nvals, nrows, ncols ;
        OK (GrB_Matrix_nvals (&nvals, X)) ;
        OK (GrB_Matrix_nrows (&nrows, X)) ;
        OK (GrB_Matrix_ncols (&ncols, X)) ;
        GrB_Vector y = NULL ;

        if (fmt == GxB_BY_COL)
        {

            //------------------------------------------------------------------
            // compute the degree of each row of X, where X is held by column
            //------------------------------------------------------------------

            if (nvals < ncols / 16 && ncols > 256)
            { 
                // X is hypersparse, or might as well be, so let y be the
                // pattern of nonempty columns of X.
                if (!GB_matlab_helper9 (X, &degree, &list, &nvec))
                {
                    ERROR ("out of memory") ;
                }
                OK (GxB_Vector_import (&y, GrB_INT64, ncols, nvec,
                    &list, &degree, NULL)) ;
            }
            else
            { 
                // y = dense vector of size ncols-by-1; value is not relevant
                OK (GrB_Vector_new (&y, GrB_BOOL, ncols)) ;
                OK (GrB_Vector_assign_BOOL (y, NULL, NULL, false, GrB_ALL,
                    ncols, NULL)) ;
            }

            // d = X*y using the PLUS_PAIR semiring
            OK (GrB_Vector_new (&d, GrB_INT64, nrows)) ;
            OK (GrB_mxv (d, NULL, NULL, GxB_PLUS_PAIR_INT64, X, y, NULL)) ;

        }
        else
        {

            //------------------------------------------------------------------
            // compute the degree of each column of X, where X is held by row
            //------------------------------------------------------------------

            if (nvals < nrows / 16 && nrows > 256)
            { 
                // X is hypersparse, or might as well be, so let y be the
                // pattern of nonempty rows of X.
                if (!GB_matlab_helper9 (X, &degree, &list, &nvec))
                {
                    ERROR ("out of memory") ;
                }
                OK (GxB_Vector_import (&y, GrB_INT64, nrows, nvec,
                    &list, &degree, NULL)) ;
            }
            else
            { 
                // y = dense vector of size nrows-by-1; value is not relevant
                OK (GrB_Vector_new (&y, GrB_BOOL, nrows)) ;
                OK (GrB_Vector_assign_BOOL (y, NULL, NULL, false, GrB_ALL,
                    nrows, NULL)) ;
            }

            // d = y*X using the PLUS_PAIR semiring
            OK (GrB_Vector_new (&d, GrB_INT64, ncols)) ;
            OK (GrB_vxm (d, NULL, NULL, GxB_PLUS_PAIR_INT64, y, X, NULL)) ;
        }

        OK (GrB_Vector_free (&y)) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&X)) ;
    pargout [0] = gb_export (&d, KIND_GRB) ;
    GB_WRAPUP ;
}

