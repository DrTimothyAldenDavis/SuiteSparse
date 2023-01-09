/* ========================================================================== */
/* === umfpack_deserialize_numeric ========================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

int umfpack_di_deserialize_numeric
(
    void **Numeric,
    char *buffer
) ;

SuiteSparse_long umfpack_dl_deserialize_numeric
(
    void **Numeric,
    char *buffer
) ;

int umfpack_zi_deserialize_numeric
(
    void **Numeric,
    char *buffer
) ;

SuiteSparse_long umfpack_zl_deserialize_numeric
(
    void **Numeric,
    char *buffer
) ;

/*
double int Syntax:

    #include "umfpack.h"
    int status ;
    char *buffer ;
    void *numeric ;
    status = umfpack_di_deserialize_numeric (&numeric, buffer) ;

double SuiteSparse_long Syntax:

    #include "umfpack.h"
    SuiteSparse_long status ;
    char *buffer ;
    void *numeric ;
    status = umfpack_dl_deserialize_numeric (&numeric, buffer) ;

complex int Syntax:

    #include "umfpack.h"
    int status ;
    char *buffer ;
    void *numeric ;
    status = umfpack_zi_deserialize_numeric (&numeric, buffer) ;

complex SuiteSparse_long Syntax:

    #include "umfpack.h"
    SuiteSparse_long status ;
    char *buffer ;
    void *numeric ;
    status = umfpack_zl_deserialize_numeric (&numeric, buffer) ;

Purpose:

    Loads a numeric object from a buffer created by umfpack_*_serialize_numeric. 
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

    char *buffer ;	    Input argument, not modified.

	The buffer from which to read the numeric object.
*/
