//------------------------------------------------------------------------------
// UMFPACK/Demo/umf4_f77wrapper: Fortran interface for UMFPACK
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2022, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/* FORTRAN interface for the C-callable UMFPACK library (double / int version
 * only and double / int64_t versions only).  This is HIGHLY
 * non-portable.  You will need to modify this depending on how your FORTRAN
 * and C compilers behave.  This has been tested in Linux, Sun Solaris, SGI
 * IRIX, and IBM AIX, with various compilers.  It has not been exhaustively
 * tested on all possible combinations of C and FORTRAN compilers.  The
 * int64_t version works on Solaris, SGI IRIX, and IBM AIX when the
 * UMFPACK library is compiled in 64-bit mode.
 *
 * Only a subset of UMFPACK's capabilities are provided.  Refer to the UMFPACK
 * User Guide for details.
 *
 * For some C and FORTRAN compilers, the FORTRAN compiler appends a single
 * underscore ("_") after each routine name.  C doesn't do this, so the
 * translation is made here.  Other FORTRAN compilers treat underscores
 * differently.  For example, a FORTRAN call to a_b gets translated to a call
 * to a_b__ by g77, and to a_b_ by most other FORTRAN compilers.  Thus, the
 * FORTRAN names here do not use underscores.  The xlf compiler in IBM AIX
 * doesn't add an underscore.
 *
 * The matrix A is passed to UMFPACK in compressed column form, with 0-based
 * indices.  In FORTRAN, for an m-by-n matrix A with nz entries, the row
 * indices of the first column (column 1) are in Ai (Ap (1) + 1 ... Ap (2)),
 * with values in Ax (Ap (1) + 1 ... Ap (2)).  The last column (column n) is
 * in Ai (Ap (n) + 1 ... Ap (n+1)) and Ax (Ap (n) + 1 ... Ap (n+1)).  The row
 * indices in Ai are in the range 0 to m-1.  They must be sorted, with no
 * duplicate entries allowed.  Refer to umfpack_di_triplet_to_col for a more
 * flexible format for the input matrix.  The following defintions apply
 * for each of the routines in this file:
 *
 *	integer m, n, Ap (n+1), Ai (nz), symbolic, numeric, filenum, status
 *	double precision Ax (nz), control (20), info (90), x (n), b (n)
 *
 * UMFPACK's status is returned in either a status argument, or in info (1).
 * It is zero if everything is OK, 1 if the matrix is singular (this is a
 * warning, not an error), and negative if an error occurred.  See umfpack.h
 * for more details on the contents of the control and info arrays, and the
 * value of the sys argument.
 *
 * For the Numeric and Symbolic handles, it's probably safe to assume that a
 * FORTRAN integer is sufficient to store a C pointer.  If that doesn't work,
 * try defining numeric and symbolic as integer arrays of size 2, or as
 * integer*8, in the FORTRAN routine that calls these wrapper routines.
 * The latter is required on Solaris, SGI IRIX, and IBM AIX when UMFPACK is
 * compiled in 64-bit mode.
 */

#include "umfpack.h"
#include <ctype.h>
#include <stdio.h>
#ifdef NULL
#undef NULL
#endif
#define NULL 0
#define LEN 200

/* -------------------------------------------------------------------------- */
/* integer type: int32_t or int64_t */
/* -------------------------------------------------------------------------- */

#if defined (DLONG)

#define Int int64_t
#define UMFPACK_defaults	 umfpack_dl_defaults
#define UMFPACK_free_numeric	 umfpack_dl_free_numeric
#define UMFPACK_free_symbolic	 umfpack_dl_free_symbolic
#define UMFPACK_numeric		 umfpack_dl_numeric
#define UMFPACK_report_control	 umfpack_dl_report_control
#define UMFPACK_report_info	 umfpack_dl_report_info
#define UMFPACK_save_numeric	 umfpack_dl_save_numeric
#define UMFPACK_save_symbolic	 umfpack_dl_save_symbolic
#define UMFPACK_load_numeric	 umfpack_dl_load_numeric
#define UMFPACK_load_symbolic	 umfpack_dl_load_symbolic
#define UMFPACK_scale		 umfpack_dl_scale
#define UMFPACK_solve		 umfpack_dl_solve
#define UMFPACK_symbolic	 umfpack_dl_symbolic

#else

#define Int int
#define UMFPACK_defaults	 umfpack_di_defaults
#define UMFPACK_free_numeric	 umfpack_di_free_numeric
#define UMFPACK_free_symbolic	 umfpack_di_free_symbolic
#define UMFPACK_numeric		 umfpack_di_numeric
#define UMFPACK_report_control	 umfpack_di_report_control
#define UMFPACK_report_info	 umfpack_di_report_info
#define UMFPACK_save_numeric	 umfpack_di_save_numeric
#define UMFPACK_save_symbolic	 umfpack_di_save_symbolic
#define UMFPACK_load_numeric	 umfpack_di_load_numeric
#define UMFPACK_load_symbolic	 umfpack_di_load_symbolic
#define UMFPACK_scale		 umfpack_di_scale
#define UMFPACK_solve		 umfpack_di_solve
#define UMFPACK_symbolic	 umfpack_di_symbolic

#endif

/* -------------------------------------------------------------------------- */
/* construct a file name from a file number (not user-callable) */
/* -------------------------------------------------------------------------- */

static void make_filename (Int filenum, char *prefix, char *filename)
{
    char *psrc, *pdst ;
#ifdef DLONG
    sprintf (filename, "%s%"PRId64".umf", prefix, filenum) ;
#else
    sprintf (filename, "%s%d.umf", prefix, filenum) ;
#endif
    /* remove any spaces in the filename */
    pdst = filename ;
    for (psrc = filename ; *psrc ; psrc++)
    {
	if (!isspace (*psrc)) *pdst++ = *psrc ;
    }
    *pdst = '\0' ;
}

/* ========================================================================== */
/* === with underscore ====================================================== */
/* ========================================================================== */

/* Solaris, Linux, and SGI IRIX.  Probably Compaq Alpha as well. */

#define umf4def_FORTRAN  umf4def_
#define umf4pcon_FORTRAN umf4pcon_
#define umf4sym_FORTRAN  umf4sym_
#define umf4num_FORTRAN  umf4num_
#define umf4solr_FORTRAN umf4solr_
#define umf4sol_FORTRAN  umf4sol_
#define umf4scal_FORTRAN umf4scal_
#define umf4pinf_FORTRAN umf4pinf_
#define umf4fnum_FORTRAN umf4fnum_
#define umf4fsym_FORTRAN umf4fsym_
#define umf4snum_FORTRAN umf4snum_
#define umf4ssym_FORTRAN umf4ssym_
#define umf4lnum_FORTRAN umf4lnum_
#define umf4lsym_FORTRAN umf4lsym_

#include "umf4_f77wrapper_methods.h"

/* ========================================================================== */
/* === with no underscore =================================================== */
/* ========================================================================== */

/* IBM AIX and HP Unix */

#define umf4def_FORTRAN  umf4def
#define umf4pcon_FORTRAN umf4pcon
#define umf4sym_FORTRAN  umf4sym
#define umf4num_FORTRAN  umf4num
#define umf4solr_FORTRAN umf4solr
#define umf4sol_FORTRAN  umf4sol
#define umf4scal_FORTRAN umf4scal
#define umf4pinf_FORTRAN umf4pinf
#define umf4fnum_FORTRAN umf4fnum
#define umf4fsym_FORTRAN umf4fsym
#define umf4snum_FORTRAN umf4snum
#define umf4ssym_FORTRAN umf4ssym
#define umf4lnum_FORTRAN umf4lnum
#define umf4lsym_FORTRAN umf4lsym

#include "umf4_f77wrapper_methods.h"

/* ========================================================================== */
/* === upper case, no underscore ============================================ */
/* ========================================================================== */

/* Microsoft Windows */

#define umf4def_FORTRAN  UMF4DEF
#define umf4pcon_FORTRAN UMF4PCON
#define umf4sym_FORTRAN  UMF4SYM
#define umf4num_FORTRAN  UMF4NUM
#define umf4solr_FORTRAN UMF4SOLR
#define umf4sol_FORTRAN  UMF4SOL
#define umf4scal_FORTRAN UMF4SCAL
#define umf4pinf_FORTRAN UMF4PINF
#define umf4fnum_FORTRAN UMF4FNUM
#define umf4fsym_FORTRAN UMF4FSYM
#define umf4snum_FORTRAN UMF4SNUM
#define umf4ssym_FORTRAN UMF4SSYM
#define umf4lnum_FORTRAN UMF4LNUM
#define umf4lsym_FORTRAN UMF4LSYM

#include "umf4_f77wrapper_methods.h"

