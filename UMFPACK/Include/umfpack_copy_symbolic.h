/* ========================================================================== */
/* === umfpack_copy_symbolic ========================================= */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

int umfpack_di_copy_symbolic
(
    void **Symbolic,
    void *Original
) ;

SuiteSparse_long umfpack_dl_copy_symbolic
(
    void **Symbolic,
    void *Original
) ;

int umfpack_zi_copy_symbolic
(
    void **Symbolic,
    void *Original
) ;

SuiteSparse_long umfpack_zl_copy_symbolic
(
    void **Symbolic,
    void *Original
) ;

/*
double int Syntax:

    #include "umfpack.h"
    int status ;
    void *Original ;
    void *Symbolic ;
    status = umfpack_di_copy_symbolic (&Symbolic, Original) ;

double SuiteSparse_long Syntax:

    #include "umfpack.h"
    SuiteSparse_long status ;
    void *Original ;
    void *Symbolic ;
    status = umfpack_dl_copy_symbolic (&Symbolic, Original) ;

complex int Syntax:

    #include "umfpack.h"
    int status ;
    void *Original ;
    void *Symbolic ;
    status = umfpack_zi_copy_symbolic (&Symbolic, Original) ;

complex SuiteSparse_long Syntax:

    #include "umfpack.h"
    SuiteSparse_long status ;
    void *Original ;
    void *Symbolic ;
    status = umfpack_zl_copy_symbolic (&Symbolic, Original) ;

Purpose:

    Copies a Symbolic object. 
    The Symbolic handle passed to this routine is overwritten with the new object.
    If that object exists prior to calling this routine, a memory leak will
    occur.  The contents of Symbolic are ignored on input.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_out_of_memory if not enough memory is available.
    UMFPACK_ERROR_file_IO if an I/O error occurred.

Arguments:

    void **Symbolic ;	    Output argument.

	**Symbolic is the address of a (void *) pointer variable in the user's
	calling routine (see Syntax, above).  On input, the contents of this
	variable are not defined.  On output, this variable holds a (void *)
	pointer to the Symbolic object (if successful), or (void *) NULL if
	a failure occurred.

    void *Original ;	    Input argument, not modified.

	The original Symbolic object to be copied.
*/
