/* ========================================================================== */
/* === umfpack_copy_numeric ========================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

int umfpack_di_copy_numeric
(
    void **Numeric,
    void *Original
) ;

SuiteSparse_long umfpack_dl_copy_numeric
(
    void **Numeric,
    void *Original
) ;

int umfpack_zi_copy_numeric
(
    void **Numeric,
    void *Original
) ;

SuiteSparse_long umfpack_zl_copy_numeric
(
    void **Numeric,
    void *Original
) ;

/*
double int Syntax:

    #include "umfpack.h"
    int status ;
    void *Original ;
    void *Numeric ;
    status = umfpack_di_copy_numeric (&Numeric, Original) ;

double SuiteSparse_long Syntax:

    #include "umfpack.h"
    SuiteSparse_long status ;
    void *Original ;
    void *Numeric ;
    status = umfpack_dl_copy_numeric (&Numeric, Original) ;

complex int Syntax:

    #include "umfpack.h"
    int status ;
    void *Original ;
    void *Numeric ;
    status = umfpack_zi_copy_numeric (&Numeric, Original) ;

complex SuiteSparse_long Syntax:

    #include "umfpack.h"
    SuiteSparse_long status ;
    void *Original ;
    void *Numeric ;
    status = umfpack_zl_copy_numeric (&Numeric, Original) ;

Purpose:

    Copies a Numeric object. 
    The numeric handle passed to this routine is overwritten with the new object.
    If that object exists prior to calling this routine, a memory leak will
    occur.  The contents of numeric are ignored on input.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_out_of_memory if not enough memory is available.
    UMFPACK_ERROR_file_IO if an I/O error occurred.

Arguments:

    void **Numeric ;	    Output argument.

	**Numeric is the address of a (void *) pointer variable in the user's
	calling routine (see Syntax, above).  On input, the contents of this
	variable are not defined.  On output, this variable holds a (void *)
	pointer to the numeric object (if successful), or (void *) NULL if
	a failure occurred.

    void *Original ;	    Input argument, not modified.

	The Numeric object to be copied.
*/
