//------------------------------------------------------------------------------
// GB_mex_cast: cast a MATLAB array using C-style casting rules
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Usage: C = GB_mex_cast (X, type) casts the dense array X to given type using
// C-style typecasting rules instead of MATLAB's rules.

#include "GB_mex.h"

#define USAGE "C = GB_mex_cast (X, type, cover)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    // do not get coverage counts unless the 3rd arg is present
    bool do_cover = (nargin == 3) ;
    bool malloc_debug = GB_mx_get_global (do_cover) ;

    // check inputs
    GB_WHERE (USAGE) ;
    if (nargout > 2 || nargin < 1 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    if (mxIsSparse (pargin [0]))
    {
        mexErrMsgTxt ("X must be dense") ;
    }

    // get X
    GB_void *X ;
    int64_t nrows, ncols ;
    GrB_Type xtype ;
    GB_mx_mxArray_to_array (pargin [0], &X, &nrows, &ncols, &xtype) ;
    if (xtype == NULL)
    {
        mexErrMsgTxt ("X must be numeric") ;
    }

    // get the type for C, default is same as X
    GrB_Type ctype = GB_mx_string_to_Type (PARGIN (1), xtype) ;
    if (ctype == NULL)
    {
        mexErrMsgTxt ("C must be numeric") ;
    }

    // create C
    pargout [0] = GB_mx_create_full (nrows, ncols, ctype) ;
    if (xtype == Complex) xtype = GxB_FC64 ;
    GB_void *C = mxGetData (pargout [0]) ;

//  GxB_print (xtype, 3) ;
//  GxB_print (ctype, 3) ;
//  printf ("\nGB_mex_cast from %d to %d: size %ld\n", xtype->code,
//      ctype->code, nrows*ncols) ;

//  printf ("X input:\n") ;
//  for (int k = 0 ; k < nrows*ncols ; k++)
//  {
//      printf ("X [%d] = ", k) ;
//      GB_code_check (xtype->code, X + k*(xtype->size), 3, NULL, Context) ;
//      printf ("\n") ;
//  }

    // cast the data from X to C
    GB_cast_array (C, ctype->code, X, xtype->code, xtype->size, nrows*ncols, 1) ;

//  printf ("X input again:\n") ;
//  for (int k = 0 ; k < nrows*ncols ; k++)
//  {
//      printf ("X [%d] = ", k) ;
//      GB_code_check (xtype->code, X + k*(xtype->size), 3, NULL, Context) ;
//      printf ("\n") ;
//  }

//  printf ("C output:\n") ;
//  for (int k = 0 ; k < nrows*ncols ; k++)
//  {
//      printf ("C [%d] = ", k) ;
//      GB_code_check (ctype->code, C + k*(ctype->size), 3, NULL, Context) ;
//      printf ("\n") ;
//  }

    GB_mx_put_global (do_cover, 0) ;
}

