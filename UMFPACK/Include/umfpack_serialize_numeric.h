/* ========================================================================== */
/* === umfpack_serialize_numeric============================================= */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

int umfpack_di_serialize_numeric_size
(
    Int *size,
    void *Numeric
) ;

SuiteSparse_long umfpack_dl_serialize_numeric_size
(
    Int *size,
    void *Numeric
) ;

int umfpack_zi_serialize_numeric_size
(
    Int *size,
    void *Numeric
) ;

SuiteSparse_long umfpack_zl_serialize_numeric_size
(
    Int *size,
    void *Numeric
) ;

int umfpack_di_serialize_numeric
(
    void *Numeric,
    void *buffer
) ;

SuiteSparse_long umfpack_dl_serialize_numeric
(
    void *Numeric,
    void *buffer
) ;

int umfpack_zi_serialize_numeric
(
    void *Numeric,
    void *buffer
) ;

SuiteSparse_long umfpack_zl_serialize_numeric
(
    void *Numeric,
    void *buffer
) ;

/*
double int Syntax:

    #include "umfpack.h"
    int status ;
    char *buffer ;
    void *Numeric ;
    status = umfpack_di_serialize_numeric (Numeric, buffer) ;

double SuiteSparse_long Syntax:

    #include "umfpack.h"
    SuiteSparse_long status ;
    char *buffer ;
    void *Numeric ;
    status = umfpack_dl_serialize_numeric (Numeric, buffer) ;

complex int Syntax:

    #include "umfpack.h"
    int status ;
    char *buffer ;
    void *Numeric ;
    status = umfpack_zi_serialize_numeric (Numeric, buffer) ;

complex SuiteSparse_long Syntax:

    #include "umfpack.h"
    SuiteSparse_long status ;
    char *buffer ;
    void *Numeric ;
    status = umfpack_zl_serialize_numeric (Numeric, buffer) ;

Purpose:

    Saves a Numeric object to a buffer, which can later be read by
    umfpack_*_deserialize_numeric.  
    The Numeric object is not modified.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_invalid_Numeric_object if Numeric is not valid.
    UMFPACK_ERROR_file_IO if an I/O error occurred.

Arguments:

    void *Numeric ;	    Input argument, not modified.

	Numeric must point to a valid Numeric object, computed by
	umfpack_*_numeric or loaded by umfpack_*_deserialize_numeric.

    char *buffer ;	    Input argument, not modified.

	The buffer to which the Numeric object is written.
*/
