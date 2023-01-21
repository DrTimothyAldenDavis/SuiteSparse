METIS 5.1.0, with minor modifications by Tim Davis
to incorporate it into SuiteSparse.

This copy of METIS is slightly changed from the original METIS v5.1.0
distribution.  The use of METIS in SuiteSparse is optional, but if used, this
revised version is required.  Do not attempt to use this copy of METIS to
build a stand-alone METIS library.

(1) In metis-5.1.0/include/metis.h, the default integer size has been changed
    from 32 bits to 64 (IDXTYPEWIDTH).  METIS 5.1.0 gives this flexility to the
    user, asking the user to modify this file.  That has been done here, and as
    a result, this file is renamed to SuiteSparse_metis.h.  Getting the
    unmodified libmetis.so in a Linux distro (likely with 32-bit integers)
    combined with a modified metis.h (with 64-bit integers) breaks things
    badly.  So the safest thing is to compile the METIS functions as built
    into CHOLMOD, rather than creating a new and different libmetis.so.

(3) The files metis-5.1.0/GKlib/GKLib.h and metis-5.1.0/GKlib/memory.c have
    been modified to disable the signal-handling in METIS, which is used when
    METIS runs out of memory.

(4) The abs and iabs functions in the original metis-5.1.0 give compiler
    warnings when IDXTYPEWIDTH is 64, so they have been replaced with a static
    inline function, SuiteSparse_metis_abs64.

(5) Warnings disabled to avoid compiler warnings of
    misleading indentation (getopt.c, csr.c).

(6) The malloc/calloc/realloc/free functions have been replaced with
    SuiteSparse_config.(malloc/calloc/realloc/free) throughout.  The gkmcore
    feature is disabled since it can conflict with the use of mxMalloc
    in the MATLAB interface to SuiteSparse.  See GKlib/memory.c.

(7) All original files from metis-5.1.0 that have been modified here have been
    placed in ./include/original, ./libmetis/original, or ./GKlib/original

Tim Davis, Jan 19, 2023, Texas A&M University.  Any changes made by Tim Davis
are released to the original copyright holder, under the original Apache-2.0
license of METIS.

METIS, Copyright 1995-2013, Regents of the University of Minnesota.
Author: George Karypis
SPDX-License-identifier: Apache-2.0

