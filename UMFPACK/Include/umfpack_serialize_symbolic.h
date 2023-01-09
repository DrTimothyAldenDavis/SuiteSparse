/* ========================================================================== */
/* === umfpack_serialize_symbolic================================================= */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

int umfpack_di_serialize_symbolic_size
(
    Int *size,
    void *Symbolic
) ;

SuiteSparse_long umfpack_dl_serialize_symbolic_size
(
    Int *size,
    void *Symbolic
) ;

int umfpack_zi_serialize_symbolic_size
(
    Int *size,
    void *Symbolic
) ;

SuiteSparse_long umfpack_zl_serialize_symbolic_size
(
    Int *size,
    void *Symbolic
) ;

int umfpack_di_serialize_symbolic
(
    void *Symbolic,
    void *buffer
) ;

SuiteSparse_long umfpack_dl_serialize_symbolic
(
    void *Symbolic,
    void *buffer
) ;

int umfpack_zi_serialize_symbolic
(
    void *Symbolic,
    void *buffer
) ;

SuiteSparse_long umfpack_zl_serialize_symbolic
(
    void *Symbolic,
    void *buffer
) ;

/*
double int Syntax:

    #include "umfpack.h"
    int status ;
    char *buffer ;
    void *Symbolic ;
    status = umfpack_di_serialize_symbolic (Symbolic, buffer) ;

double SuiteSparse_long Syntax:

    #include "umfpack.h"
    SuiteSparse_long status ;
    char *buffer ;
    void *Symbolic ;
    status = umfpack_dl_serialize_symbolic (Symbolic, buffer) ;

complex int Syntax:

    #include "umfpack.h"
    int status ;
    char *buffer ;
    void *Symbolic ;
    status = umfpack_zi_serialize_symbolic (Symbolic, buffer) ;

complex SuiteSparse_long Syntax:

    #include "umfpack.h"
    SuiteSparse_long status ;
    char *buffer ;
    void *Symbolic ;
    status = umfpack_zl_serialize_symbolic (Symbolic, buffer) ;

Purpose:

    Saves a Symbolic object to a file, which can later be read by
    umfpack_*_load_symbolic.  The Symbolic object is not modified.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_invalid_Symbolic_object if Symbolic is not valid.
    UMFPACK_ERROR_file_IO if an I/O error occurred.

Arguments:

    void *Symbolic ;	    Input argument, not modified.

	Symbolic must point to a valid Symbolic object, computed by
	umfpack_*_symbolic or loaded by umfpack_*_load_symbolic.

    char *buffer ;	    Input argument, not modified.

	The buffer to which the Symbolic object is written.
*/
