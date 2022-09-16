//------------------------------------------------------------------------------
// AMD/Demo/amd_f77wrapper: Fortran interface for AMD
//------------------------------------------------------------------------------

// AMD, Copyright (c) 1996-2022, Timothy A. Davis, Patrick R. Amestoy, and
// Iain S. Duff.  All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

/* Fortran interface for the C-callable AMD library (int32_t only).  This
 * is HIGHLY non-portable.  You will need to modify this depending on how your
 * Fortran and C compilers behave.  Two examples are provided.
 *
 * To avoid using I/O, and to avoid the extra porting step of a Fortran
 * function, the status code is returned as the first entry in P (P [0] in C
 * and P (1) in Fortran) if an error occurs.  The error codes are negative
 * (-1: out of memory, -2: invalid matrix).
 *
 * For some C and Fortran compilers, the Fortran compiler appends a single "_"
 * after each routine name.  C doesn't do this, so the translation is made
 * here.  Some Fortran compilers don't append an underscore (xlf on IBM AIX,
 * for * example).
 */

#include "amd.h"

/* ------------------------------------------------------------------------- */
/* Linux, Solaris, SGI */
/* ------------------------------------------------------------------------- */

void amdorder_ (int32_t *n, const int32_t *Ap, const int32_t *Ai, int32_t *P,
    double *Control, double *Info)
{
    int32_t result = amd_order (*n, Ap, Ai, P, Control, Info) ;
    if (result != AMD_OK && P) P [0] = result ;
}

void amddefaults_ (double *Control)
{
    amd_defaults (Control) ;
}

void amdcontrol_ (double *Control)
{
    fflush (stdout) ;
    amd_control (Control) ;
    fflush (stdout) ;
}

void amdinfo_ (double *Info)
{
    fflush (stdout) ;
    amd_info (Info) ;
    fflush (stdout) ;
}

/* ------------------------------------------------------------------------- */
/* IBM AIX.  Probably Windows, Compaq Alpha, and HP Unix as well. */
/* ------------------------------------------------------------------------- */

void amdorder (int32_t *n, const int32_t *Ap, const int32_t *Ai, int32_t *P,
    double *Control, double *Info)
{
    int32_t result = amd_order (*n, Ap, Ai, P, Control, Info) ;
    if (result != AMD_OK && P) P [0] = result ;
}

void amddefaults (double *Control)
{
    amd_defaults (Control) ;
}

void amdcontrol (double *Control)
{
    fflush (stdout) ;
    amd_control (Control) ;
    fflush (stdout) ;
}

void amdinfo (double *Info)
{
    fflush (stdout) ;
    amd_info (Info) ;
    fflush (stdout) ;
}
